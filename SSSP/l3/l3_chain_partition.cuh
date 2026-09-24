#pragma once
#if (L3_CHAIN_PARTITION == true)
#if (!defined(TYPE_INT) || USE_DIST_IN_STRUCT == false || WORK_COUNT == true || L3_WORK_DIAG == true || L3_CHAIN_SHORTCUTS == true || L3_DIRECT_RX == false || L3_TERM_ONLY_WORKER == false || GHOST_DEPTH > 0 || L3_RX_EXPRESS == true || L3_RX_L2_PULL == true || L3_ADMISSION_BUDGET == true || L0_DIRECT_SMALL == true || L0_SOURCE_SNAPSHOT == true || SEED_EXP == true)
#error "chain partition prototype requires integer ordinary term-only BULK and original CSR"
#endif
#include "l3_chain_partition.h"
#include <cstdio>
#include <cstdlib>

static inline void l3_chain_cuda_require(cudaError_t status,const char *operation) {
    if(status==cudaSuccess)return;
    std::fprintf(stderr,"L3_CHAIN_CUDA_ERROR operation=%s error=%s\n",
        operation,cudaGetErrorString(status));
    std::exit(2);
}
__device__ uint64_t *g_l3_chain_route=nullptr;
__device__ l3_chain_partition_node *g_l3_chain_nodes=nullptr;
static uint64_t *l3_chain_route_storage[MAX_GPU]={};
static l3_chain_partition_node *l3_chain_node_storage[MAX_GPU]={};
#if (L3_CHAIN_PARTITION_DIAG == true)
// closures, materialized interiors, queued tails, RX-derived sources,
// route reads, unique segments, repeated closures, max visits to one segment
__device__ unsigned long long g_l3_chain_partition_counts[8];
__device__ unsigned *g_l3_chain_segment_visits=nullptr;
static unsigned *l3_chain_segment_visit_storage[MAX_GPU]={};
static size_t l3_chain_segment_visit_count[MAX_GPU]={};
#endif
void l3_chain_partition_release(int gpu) {
    l3_chain_cuda_require(cudaSetDevice(gpu),"set device for release");
    if(l3_chain_route_storage[gpu])
        l3_chain_cuda_require(cudaFree(l3_chain_route_storage[gpu]),"chain route free");
    if(l3_chain_node_storage[gpu])
        l3_chain_cuda_require(cudaFree(l3_chain_node_storage[gpu]),"chain nodes free");
#if (L3_CHAIN_PARTITION_DIAG == true)
    if(l3_chain_segment_visit_storage[gpu])
        l3_chain_cuda_require(cudaFree(l3_chain_segment_visit_storage[gpu]),"chain visits free");
    l3_chain_segment_visit_storage[gpu]=nullptr;l3_chain_segment_visit_count[gpu]=0;
#endif
    l3_chain_route_storage[gpu]=nullptr;l3_chain_node_storage[gpu]=nullptr;
    l3_chain_cuda_require(cudaMemcpyToSymbol(g_l3_chain_route,&l3_chain_route_storage[gpu],sizeof(uint64_t*)),"chain route clear");
    l3_chain_cuda_require(cudaMemcpyToSymbol(g_l3_chain_nodes,&l3_chain_node_storage[gpu],sizeof(l3_chain_partition_node*)),"chain nodes clear");
#if (L3_CHAIN_PARTITION_DIAG == true)
    l3_chain_cuda_require(cudaMemcpyToSymbol(g_l3_chain_segment_visits,&l3_chain_segment_visit_storage[gpu],sizeof(unsigned*)),"chain visits clear");
#endif
}
void l3_chain_partition_install(int gpu,const l3_chain_partition_index &index) {
    l3_chain_partition_release(gpu);
    const double start=mlmq_bench_ms();
    const size_t route_bytes=index.route.size()*sizeof(uint64_t),node_bytes=index.nodes.size()*sizeof(l3_chain_partition_node);
    l3_chain_cuda_require(cudaMalloc(&l3_chain_route_storage[gpu],route_bytes),"chain route alloc");
    l3_chain_cuda_require(cudaMemcpy(l3_chain_route_storage[gpu],index.route.data(),route_bytes,cudaMemcpyHostToDevice),"chain route upload");
    if(node_bytes){
        l3_chain_cuda_require(cudaMalloc(&l3_chain_node_storage[gpu],node_bytes),"chain nodes alloc");
        l3_chain_cuda_require(cudaMemcpy(l3_chain_node_storage[gpu],index.nodes.data(),node_bytes,cudaMemcpyHostToDevice),"chain nodes upload");
#if (L3_CHAIN_PARTITION_DIAG == true)
        l3_chain_segment_visit_count[gpu]=index.nodes.size();
        l3_chain_cuda_require(cudaMalloc(&l3_chain_segment_visit_storage[gpu],index.nodes.size()*sizeof(unsigned)),"chain visits alloc");
        l3_chain_cuda_require(cudaMemset(l3_chain_segment_visit_storage[gpu],0,index.nodes.size()*sizeof(unsigned)),"chain visits initialize");
#endif
    }
    l3_chain_cuda_require(cudaMemcpyToSymbol(g_l3_chain_route,&l3_chain_route_storage[gpu],sizeof(uint64_t*)),"chain route bind");
    l3_chain_cuda_require(cudaMemcpyToSymbol(g_l3_chain_nodes,&l3_chain_node_storage[gpu],sizeof(l3_chain_partition_node*)),"chain nodes bind");
#if (L3_CHAIN_PARTITION_DIAG == true)
    l3_chain_cuda_require(cudaMemcpyToSymbol(g_l3_chain_segment_visits,&l3_chain_segment_visit_storage[gpu],sizeof(unsigned*)),"chain visits bind");
    printf("L3_CHAIN_PARTITION_DIAG_MEMORY gpu=%d visit_bytes=%zu\n",gpu,
        index.nodes.size()*sizeof(unsigned));
#endif
    printf("L3_CHAIN_PARTITION_UPLOAD gpu=%d bytes=%zu upload_ms=%.6f\n",gpu,route_bytes+node_bytes,mlmq_bench_ms()-start);
}
#if (L3_CHAIN_PARTITION_DIAG == true)
void l3_chain_partition_diag_reset(int gpu) {
    l3_chain_cuda_require(cudaSetDevice(gpu),"set device for chain visits reset");
    if(l3_chain_segment_visit_storage[gpu])
        l3_chain_cuda_require(cudaMemset(l3_chain_segment_visit_storage[gpu],0,l3_chain_segment_visit_count[gpu]*sizeof(unsigned)),"chain visits reset");
}
#endif
// All lanes use the SAME source snapshot for a closure. Thus each internal
// outgoing relaxation is dominated by another target of this closure. The
// original input L2 credit covers publication/materialization by all lanes.
__device__ __forceinline__ void l3_chain_partition_target(
    int base,int offset,int source,int snapshot,int &vertex,int &value,bool &interior) {
    int index=base+offset;index+=index>=source; // omit source itself
    const auto target=g_l3_chain_nodes[index],origin=g_l3_chain_nodes[source];
    long long distance=(long long)snapshot+(index>source?target.forward-origin.forward:origin.reverse-target.reverse);
    vertex=target.id;value=distance>=DIST_MAX?DIST_MAX:int(distance);
    interior=index>base&&index<base+g_l3_chain_nodes[base].length-1;
}
#endif
