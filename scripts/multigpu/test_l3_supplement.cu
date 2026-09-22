// Small concurrency fixtures calling the production candidate, collector,
// transport and RX helpers. Test gates are kept separate from payload storage.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../SSSP/sssp.cuh"
__device__ unsigned long long g_dq_clamp;
__host__ __device__ node_struct::node_struct(int i,int d):id(i),dist(d){}
__host__ __device__ node_struct::node_struct():id(0),dist(DIST_MAX){}
__device__ VALUE_TYPE node_struct::get_data(){return dist;}
#include "../../SSSP/l3/l3_candidate.cuh"
#include "../../SSSP/l3/l3_transport.cuh"
#include "../../SSSP/l3/l3_bulk.cuh"
#include "../../SSSP/l3/l3_collect.cuh"
#include "../../SSSP/l3/l3_receive.cuh"
#if L3_FAULT_INJECT_ACK_DELAY
__device__ unsigned int g_l3_fault_ack_delay;
__device__ int g_l3_fault_ack_pending;
#endif
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",#x,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(int n){T*p;CHECK(cudaMalloc(&p,n*sizeof(T)));CHECK(cudaMemset(p,0,n*sizeof(T)));return p;}
__device__ void wait_gate(int *p,int value){cuda::atomic_ref<int,cuda::thread_scope_device>a(*p);unsigned long long t=clock64();while(a.load(cuda::memory_order_acquire)<value){assert(clock64()-t<10000000000ull);}}
__device__ void gate(int*p,int value){cuda::atomic_ref<int,cuda::thread_scope_device>(*p).store(value,cuda::memory_order_release);}
// Explicit schedules establish the ownership invariant; they are not weak-memory litmus tests.
__global__ void candidate_schedule(int*c,unsigned*m,unsigned*h,unsigned*h2,int*g,int*out,int mode){
    if(threadIdx.x)return;
    if(blockIdx.x==0){
        l3_record_candidate(c,m,h,h2,0,100,nullptr,nullptr,nullptr,nullptr);gate(g,1);
        wait_gate(g,2);
        l3_record_candidate(c,m,h,h2,0,50,nullptr,nullptr,nullptr,nullptr);gate(g,3);
    }else{
        wait_gate(g,1);
        if(mode==0){gate(g,2);wait_gate(g,3);}
        unsigned bits=atomicAdd(m,0u);assert(bits==1 && atomicCAS(m,bits,0u)==bits);
        if(mode==1){gate(g,2);wait_gate(g,3);}
        out[0]=atomicExch(c+1,DIST_MAX);
        if(mode==2){gate(g,2);wait_gate(g,3);}
        unsigned later=atomicExch(m,0u);out[1]=later?atomicExch(c+1,DIST_MAX):DIST_MAX;
        out[2]=min(out[0],out[1]);
    }
}
__global__ void candidate_race(int*c,unsigned*m,unsigned*h,unsigned*h2,int*done,int*seen,int*stats){
    int lane=threadIdx.x;
    if(blockIdx.x<8){
        for(int t=0;t<128;++t){int id=(blockIdx.x*113+lane*7+t)%1024;int value=100000-t*256-blockIdx.x*32-lane;
            l3_record_candidate(c,m,h,h2,id,value,nullptr,nullptr,nullptr,nullptr);}
        __threadfence();__syncwarp();if(!lane)atomicAdd(done,1);
    }else{
        __shared__ int ids[1024],values[1024];
        for(int pass=0;pass<100000;++pass){
            int count=l3_collect_cooperative(c,m,h,h2,c,1024,1024,ids,values,lane);
            for(int i=lane;i<count;i+=32)atomicMin(seen+ids[i]-1,values[i]);
            // Same authoritative mark fallback as TX: claim first, exchange second.
            unsigned bits=atomicAdd(m+lane,0u);if(bits && atomicCAS(m+lane,bits,0u)!=bits)bits=0;
            while(bits){int b=__ffs(bits)-1;bits&=bits-1;int id=lane*32+b;int value=atomicExch(c+id+1,DIST_MAX);atomicMin(seen+id,value);}
            __syncwarp();
            bool drained=atomicAdd(done,0)==8 && atomicAdd(m+lane,0u)==0;
            if(__all_sync(FULL_MASK,drained)){if(!lane)stats[0]=pass+1;return;}
        }assert(false);
    }
}
struct ReceiverQueue {
    l2_delta_queue<node_struct> q2;
    int *states,*acks,*generations,*distances,*checks;
    __device__ write_status write_through(node_struct *p,int&n,int b,int w,int lane,unsigned*debug){
        if(!lane){
            int reading=(atomicAdd(states,0)==BULK_SLOT_READING)+(atomicAdd(states+1,0)==BULK_SLOT_READING);
            assert(reading==1);assert(distances[p[0].id]==p[0].dist);
            for(int slot=0;slot<2;++slot)if(atomicAdd(states+slot,0)==BULK_SLOT_READING)assert(atomicAdd(acks+slot,0)<atomicAdd(generations+slot,0));
            // RX has applied D and still owns READING before the real DQ commit.
            atomicAdd(checks,1);__nanosleep(1000);
        }__syncwarp();return q2.write(p,n,b,w,lane,debug);
    }
};
struct Transport {node_struct *inbox;int *counts,*epochs,*acks,*states,*gens,*gate,*distances,*checks,*result;};
constexpr int epochs=256,items=32;
__device__ int sysload(int*p){return cuda::atomic_ref<int,cuda::thread_scope_system>(*p).load(cuda::memory_order_acquire);}
__device__ void sysstore(int*p,int v){cuda::atomic_ref<int,cuda::thread_scope_system>(*p).store(v,cuda::memory_order_release);}
__global__ void sender(Transport t){
    int lane=threadIdx.x;__shared__ int ids[items],values[items];ids[lane]=lane+1;
    int retries=0;unsigned long long begin=clock64();
    for(int e=1;e<=epochs;++e){values[lane]=10000-e;__syncwarp();
        while(!bulk_publish_l3_batch(e,lane,0,items,items,ids,values,t.inbox,t.counts,t.epochs,t.acks,t.states,t.gens)){
            ++retries;assert(values[lane]==10000-e && ids[lane]==lane+1);
            if(e==3 && !lane)sysstore(t.gate,1); // release receiver only after proving both slots occupied
            assert(clock64()-begin<30000000000ull);
        }
    }
    if(!lane)t.result[0]=retries;
}
__global__ void receiver(Transport t,ReceiverQueue queue){
    int lane=threadIdx.x,rx=0;__shared__ int journal_storage[64];node_struct *journal=reinterpret_cast<node_struct*>(journal_storage);
    if(!lane){unsigned long long start=clock64();while(!sysload(t.gate))assert(clock64()-start<30000000000ull);}__syncwarp();
    unsigned long long start=clock64();
    while(rx<epochs){
#if L3_FAULT_INJECT_ACK_DELAY
        // Finite receiver-local delay, independent of any future RX action.
        if(rx==1){if(!lane)__nanosleep(100000);__syncwarp();}
        if(!lane)bulk_inbox_retry_delayed_ack(t.states,t.acks);__syncwarp();
#endif
        int old=rx;int wins=l3_receive_to_l2(t.inbox,t.counts,t.epochs,t.acks,t.states,t.gens,rx,t.distances,0,items,journal,queue,nullptr,lane);
        if(rx>old)assert(wins==items);
        assert(clock64()-start<30000000000ull);
    }
    if(!lane)t.result[1]=rx;
}
void candidates(int gpu){CHECK(cudaSetDevice(gpu));int*c=alloc<int>(1025),*g=alloc<int>(1),*out=alloc<int>(3),*done=alloc<int>(1),*seen=alloc<int>(1024),*stats=alloc<int>(1);unsigned*m=alloc<unsigned>(32),*h=alloc<unsigned>(1),*h2=alloc<unsigned>(1);
    std::vector<int>initial(1025,DIST_MAX);int cases=0;
    for(int mode=0;mode<3;++mode)for(int r=0;r<32;++r){CHECK(cudaMemcpy(c,initial.data(),4100,cudaMemcpyHostToDevice));CHECK(cudaMemset(m,0,128));CHECK(cudaMemset(g,0,4));candidate_schedule<<<2,1>>>(c,m,h,h2,g,out,mode);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());int host[3];CHECK(cudaMemcpy(host,out,12,cudaMemcpyDeviceToHost));assert(host[2]==50);assert(host[0]==(mode==2?100:50) && host[1]==(mode==2?50:DIST_MAX));++cases;}
    std::vector<int>expected(1024,DIST_MAX),actual(1024);
    for(int b=0;b<8;++b)for(int l=0;l<32;++l)for(int t=0;t<128;++t){int id=(b*113+l*7+t)%1024;expected[id]=std::min(expected[id],100000-t*256-b*32-l);}
    for(int r=0;r<32;++r){CHECK(cudaMemcpy(c,initial.data(),4100,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(seen,initial.data(),4096,cudaMemcpyHostToDevice));CHECK(cudaMemset(m,0,128));CHECK(cudaMemset(h,0,4));CHECK(cudaMemset(h2,0,4));CHECK(cudaMemset(done,0,4));candidate_race<<<9,32>>>(c,m,h,h2,done,seen,stats);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(actual.data(),seen,4096,cudaMemcpyDeviceToHost));assert(actual==expected);}
    printf("CANDIDATE gpu=%d controlled=%d racing=32 updates_per_race=32768 PASS\n",gpu,cases);
    for(void*p:std::vector<void*>{(void*)c,g,out,done,seen,stats,m,h,h2})CHECK(cudaFree(p));
}
void transport(int source,int target){
    CHECK(cudaSetDevice(target));Transport t{};t.inbox=alloc<node_struct>(66);t.counts=alloc<int>(2);t.epochs=alloc<int>(2);t.acks=alloc<int>(2);t.states=alloc<int>(2);t.gens=alloc<int>(2);t.gate=alloc<int>(1);t.distances=alloc<int>(33);t.checks=alloc<int>(1);t.result=alloc<int>(2);
    std::vector<int>d(33,DIST_MAX);CHECK(cudaMemcpy(t.distances,d.data(),132,cudaMemcpyHostToDevice));
    ReceiverQueue q{};q.states=t.states;q.acks=t.acks;q.generations=t.gens;q.distances=t.distances;q.checks=t.checks;q.q2.bucketNum=1;q.q2.total_size=16384;q.q2.total_block_size=32;q.q2.delta=200000;q.q2.data=alloc<node_struct>(16384);q.q2.block_write_done=alloc<int>(32);q.q2.write_reserve=alloc<int>(1);q.q2.first_pos=alloc<int>(1);q.q2.debug_write_done=alloc<int>(1);
    cudaFuncAttributes receiver_attr{};CHECK(cudaFuncGetAttributes(&receiver_attr,receiver)); // load before concurrent persistent launch
    CHECK(cudaSetDevice(source));cudaFuncAttributes attr{};CHECK(cudaFuncGetAttributes(&attr,sender));
    CHECK(cudaSetDevice(target));receiver<<<1,32>>>(t,q);CHECK(cudaGetLastError());
    CHECK(cudaSetDevice(source));sender<<<1,32>>>(t);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaSetDevice(target));CHECK(cudaDeviceSynchronize());
    int result[2],checks,writes,ack[2];CHECK(cudaMemcpy(result,t.result,8,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&checks,t.checks,4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(&writes,q.q2.write_reserve,4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(ack,t.acks,8,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(d.data(),t.distances,132,cudaMemcpyDeviceToHost));
    assert(result[0]>0 && result[1]==epochs && checks==epochs && writes==epochs*items && ack[epochs%2]==epochs);
    for(int i=1;i<=items;++i)assert(d[i]==10000-epochs);
    std::vector<node_struct>records(writes);CHECK(cudaMemcpy(records.data(),q.q2.data,writes*sizeof(node_struct),cudaMemcpyDeviceToHost));
    for(int e=0;e<epochs;++e)for(int i=0;i<items;++i)assert(records[e*items+i].id==i+1 && records[e*items+i].dist==9999-e);
    unsigned delayed=0;int pending=0;
#if L3_FAULT_INJECT_ACK_DELAY
    CHECK(cudaMemcpyFromSymbol(&delayed,g_l3_fault_ack_delay,sizeof(delayed)));CHECK(cudaMemcpyFromSymbol(&pending,g_l3_fault_ack_pending,sizeof(pending)));assert(delayed==1 && pending==0);
#endif
    printf("TRANSPORT source=%d target=%d epochs=%d backpressure_retries=%d precommit_checks=%d l2_writes=%d delayed_ack_fired=%u pending=%d PASS\n",source,target,epochs,result[0],checks,writes,delayed,pending);
    for(void*p:std::vector<void*>{(void*)t.inbox,t.counts,t.epochs,t.acks,t.states,t.gens,t.gate,t.distances,t.checks,t.result,q.q2.data,q.q2.block_write_done,q.q2.write_reserve,q.q2.first_pos,q.q2.debug_write_done})CHECK(cudaFree(p));
}
int main(){int n;CHECK(cudaGetDeviceCount(&n));assert(n>=2);
    for(int a=0;a<2;++a){cudaDeviceProp p{};CHECK(cudaGetDeviceProperties(&p,a));printf("DEVICE gpu=%d name=%s cc=%d%d sms=%d\n",a,p.name,p.major,p.minor,p.multiProcessorCount);
        int access=0,native=0;CHECK(cudaDeviceCanAccessPeer(&access,a,1-a));CHECK(cudaDeviceGetP2PAttribute(&native,cudaDevP2PAttrNativeAtomicSupported,a,1-a));printf("P2P source=%d target=%d access=%d native_atomics=%d\n",a,1-a,access,native);assert(access && native);CHECK(cudaSetDevice(a));CHECK(cudaDeviceEnablePeerAccess(1-a,0));}
    candidates(0);candidates(1);transport(0,1);transport(1,0);printf("L3_SUPPLEMENT_PRIMITIVES PASS\n");
}
