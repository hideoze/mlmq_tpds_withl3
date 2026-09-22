// Compile against the guarded isolated core. Boundary acceptance and rejection.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "ml_queue.cuh"
__device__ unsigned long long g_dq_clamp;
struct Item{int id,value;__device__ int get_data(){return 0;}};
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;}}while(0)
__global__ void fill(l2_delta_queue<Item>q,int batches){__shared__ Item a[32];int lane=threadIdx.x;
    for(int b=0;b<batches;++b){a[lane]={b*32+lane,123};__syncwarp();int n=32;q.write(a,n,0,0,lane,nullptr);}}
int main(int argc,char**argv){bool overflow=argc>1; l2_delta_queue<Item>q{};q.bucketNum=1;q.total_size=1024;q.total_block_size=2;q.delta=1;
    CHECK(cudaMalloc(&q.data,1024*sizeof(Item)));CHECK(cudaMalloc(&q.block_write_done,8));CHECK(cudaMemset(q.block_write_done,0,8));
    CHECK(cudaMalloc(&q.write_reserve,4));CHECK(cudaMemset(q.write_reserve,0,4));CHECK(cudaMalloc(&q.first_pos,4));CHECK(cudaMemset(q.first_pos,0,4));CHECK(cudaMalloc(&q.debug_write_done,4));CHECK(cudaMemset(q.debug_write_done,0,4));
    fill<<<1,32>>>(q,overflow?33:32);CHECK(cudaGetLastError());auto status=cudaDeviceSynchronize();
    if(overflow){if(status!=cudaErrorAssert)return 3;printf("CAPACITY_BOUNDARY rejected=1056 capacity=1024 expected_device_assert=1 PASS\n");cudaDeviceReset();return 0;}
    if(status!=cudaSuccess)return 4;Item a[1024];CHECK(cudaMemcpy(a,q.data,sizeof(a),cudaMemcpyDeviceToHost));for(int i=0;i<1024;++i)assert(a[i].id==i && a[i].value==123);
    printf("CAPACITY_BOUNDARY accepted=1024 capacity=1024 PASS\n");CHECK(cudaDeviceReset());return 0;
}
