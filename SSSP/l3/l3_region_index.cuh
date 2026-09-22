#pragma once
#if (L3_REGION_RELAX == true)
#if (!defined(TYPE_INT) || USE_DIST_IN_STRUCT == false || WORK_COUNT == true || L3_WORK_DIAG == true || L3_CHAIN_SHORTCUTS == true || L3_CHAIN_PARTITION == true || L3_DIRECT_RX == false || L3_TERM_ONLY_WORKER == false || GHOST_DEPTH > 0 || L3_RX_EXPRESS == true || L3_RX_L2_PULL == true || L3_ADMISSION_BUDGET == true || L0_DIRECT_SMALL == true || L0_SOURCE_SNAPSHOT == true || SEED_EXP == true)
#error "regional relaxation requires integer ordinary term-only BULK and original CSR"
#endif
#include "l3_region_index.h"
struct l3_region_device_index {int *map,*members,*rows,*sources,*weights;};
__device__ l3_region_device_index g_l3_regions={};
static l3_region_device_index l3_region_storage[MAX_GPU]={};
#if (L3_REGION_DIAG == true)
__device__ unsigned long long g_l3_region_counts[4]; // regions, rounds, materialized, external attempts
#endif
void l3_region_release(int gpu) {
    cudaSetDevice(gpu);auto &s=l3_region_storage[gpu];
    for(int*p:{s.map,s.members,s.rows,s.sources,s.weights})if(p)cudaFree(p);
    s={};g_benchmark.require(cudaMemcpyToSymbol(g_l3_regions,&s,sizeof(s))==cudaSuccess,"region index clear");
}
void l3_region_install(int gpu,const l3_region_index &index) {
    l3_region_release(gpu);const double start=mlmq_bench_ms();auto &s=l3_region_storage[gpu];size_t bytes=0;
    auto upload=[&](int **p,const std::vector<int>&v){
        size_t size=v.size()*sizeof(int);bytes+=size;if(!size)return;
        g_benchmark.require(cudaMalloc(p,size)==cudaSuccess,"region alloc");
        g_benchmark.require(cudaMemcpy(*p,v.data(),size,cudaMemcpyHostToDevice)==cudaSuccess,"region upload");
    };
    upload(&s.map,index.map);upload(&s.members,index.members);upload(&s.rows,index.rows);upload(&s.sources,index.sources);upload(&s.weights,index.weights);
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_regions,&s,sizeof(s))==cudaSuccess,"region index bind");
    printf("L3_REGION_UPLOAD gpu=%d bytes=%zu upload_ms=%.6f\n",gpu,bytes,mlmq_bench_ms()-start);
}
#endif
