#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cassert>
#include <cstdio>
#include <vector>
#include "ml_queue.cuh"
#if (DQ_CLAMP_DIAG == true)
__device__ unsigned long long g_dq_clamp;
#endif
struct Item {
    int id, checksum;
    __device__ int get_data() const { return 0; }
};
using Queue=l2_delta_queue<Item>;
#define CHECK(call) do { auto status=(call); if(status!=cudaSuccess) { \
    fprintf(stderr,"%s\n",cudaGetErrorString(status)); return 2; } } while(0)

__global__ void publication(Queue q, int producers, int batch, int *errors) {
    int lane=threadIdx.x;
    if(blockIdx.x < producers) {
        __shared__ Item journal[32];
        int id=blockIdx.x*batch+lane;
        journal[lane]={id,id^0x13579};
        __syncwarp();
        int count=batch;
        auto status=q.write(journal,count,blockIdx.x,0,lane,nullptr);
        assert(status==WRITE_SUCCESS);
    } else {
        int total=producers*batch;
        for(int begin=0;begin<total;begin+=MEM_BLOCK_SIZE) {
            int expected=min(MEM_BLOCK_SIZE,total-begin);
            cuda::atomic_ref<int,cuda::thread_scope_device> published(
                q.block_write_done[begin/MEM_BLOCK_SIZE]);
            while(published.load(cuda::memory_order_acquire)<expected) {}
            for(int i=lane;i<expected;i+=32) {
                volatile Item *entry=q.data+begin+i;
                int id=entry->id, checksum=entry->checksum;
                if(id<0 || id>=total || checksum!=(id^0x13579)) atomicAdd(errors,1);
            }
        }
    }
}

int main() {
    Queue q={};
    q.bucketNum=1; q.total_size=2048; q.total_block_size=4; q.delta=1;
    CHECK(cudaMallocManaged(&q.data,2048*sizeof(Item)));
    CHECK(cudaMallocManaged(&q.block_write_done,4*sizeof(int)));
    CHECK(cudaMallocManaged(&q.write_reserve,sizeof(int)));
    CHECK(cudaMallocManaged(&q.first_pos,sizeof(int)));
    CHECK(cudaMallocManaged(&q.debug_write_done,sizeof(int)));
    int *errors; CHECK(cudaMallocManaged(&errors,sizeof(int)));
    int tests=0;
    for(int base:{0,7}) for(int batch:{1,7,16,32}) for(int repeat=0;repeat<50;++repeat) {
        CHECK(cudaMemset(q.data,0xff,2048*sizeof(Item)));
        CHECK(cudaMemset(q.block_write_done,0,4*sizeof(int)));
        CHECK(cudaMemset(q.write_reserve,0,sizeof(int)));
        CHECK(cudaMemcpy(q.first_pos,&base,sizeof(int),cudaMemcpyHostToDevice));
#if (DQ_CLAMP_DIAG == true)
        unsigned long long clamp_count=0;
        CHECK(cudaMemcpyToSymbol(g_dq_clamp,&clamp_count,sizeof(clamp_count)));
#endif
        CHECK(cudaMemset(q.debug_write_done,0,sizeof(int)));
        CHECK(cudaMemset(errors,0,sizeof(int)));
        publication<<<33,32>>>(q,32,batch,errors);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        assert(*errors==0 && *q.write_reserve==32*batch && *q.debug_write_done==32*batch);
#if (DQ_CLAMP_DIAG == true)
        CHECK(cudaMemcpyFromSymbol(&clamp_count,g_dq_clamp,sizeof(clamp_count)));
        assert(clamp_count==(base ? 32ull*batch : 0ull));
#endif
        std::vector<bool> seen(32*batch,false);
        for(int i=0;i<32*batch;++i) {
            auto item=q.data[i];
            assert(item.id>=0 && item.id<32*batch && !seen[item.id]);
            assert(item.checksum==(item.id^0x13579)); seen[item.id]=true;
        }
        ++tests;
    }
    printf("DQ_PUBLICATION tests=%d clamp_diag=%d PASS\n",tests,int(DQ_CLAMP_DIAG));
    CHECK(cudaFree(q.data)); CHECK(cudaFree(q.block_write_done)); CHECK(cudaFree(q.write_reserve));
    CHECK(cudaFree(q.first_pos)); CHECK(cudaFree(q.debug_write_done)); CHECK(cudaFree(errors));
}
