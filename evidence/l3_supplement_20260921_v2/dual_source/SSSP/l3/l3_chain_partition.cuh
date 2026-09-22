#pragma once
#if (L3_CHAIN_PARTITION == true)
#if (!defined(TYPE_INT) || USE_DIST_IN_STRUCT == false || WORK_COUNT == true || L3_WORK_DIAG == true || L3_CHAIN_SHORTCUTS == true || L3_DIRECT_RX == false || L3_TERM_ONLY_WORKER == false || GHOST_DEPTH > 0 || L3_RX_EXPRESS == true || L3_RX_L2_PULL == true || L3_ADMISSION_BUDGET == true || L0_DIRECT_SMALL == true || L0_SOURCE_SNAPSHOT == true || SEED_EXP == true)
#error "chain partition prototype requires integer ordinary term-only BULK and original CSR"
#endif
#include "l3_chain_partition.h"
__device__ uint64_t *g_l3_chain_route=nullptr;
__device__ l3_chain_partition_node *g_l3_chain_nodes=nullptr;
static uint64_t *l3_chain_route_storage[MAX_GPU]={};
static l3_chain_partition_node *l3_chain_node_storage[MAX_GPU]={};
#if (L3_CHAIN_PARTITION_DIAG == true)
__device__ unsigned long long g_l3_chain_partition_counts[5]; // closures, materialized, tails, sources, route reads
#endif
void l3_chain_partition_release(int gpu) {
    cudaSetDevice(gpu);
    if(l3_chain_route_storage[gpu])cudaFree(l3_chain_route_storage[gpu]);
    if(l3_chain_node_storage[gpu])cudaFree(l3_chain_node_storage[gpu]);
    l3_chain_route_storage[gpu]=nullptr;l3_chain_node_storage[gpu]=nullptr;
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_chain_route,&l3_chain_route_storage[gpu],sizeof(uint64_t*))==cudaSuccess,"chain route clear");
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_chain_nodes,&l3_chain_node_storage[gpu],sizeof(l3_chain_partition_node*))==cudaSuccess,"chain nodes clear");
}
void l3_chain_partition_install(int gpu,const l3_chain_partition_index &index) {
    l3_chain_partition_release(gpu);
    const double start=mlmq_bench_ms();
    const size_t route_bytes=index.route.size()*sizeof(uint64_t),node_bytes=index.nodes.size()*sizeof(l3_chain_partition_node);
    g_benchmark.require(cudaMalloc(&l3_chain_route_storage[gpu],route_bytes)==cudaSuccess,"chain route alloc");
    g_benchmark.require(cudaMemcpy(l3_chain_route_storage[gpu],index.route.data(),route_bytes,cudaMemcpyHostToDevice)==cudaSuccess,"chain route upload");
    if(node_bytes){
        g_benchmark.require(cudaMalloc(&l3_chain_node_storage[gpu],node_bytes)==cudaSuccess,"chain nodes alloc");
        g_benchmark.require(cudaMemcpy(l3_chain_node_storage[gpu],index.nodes.data(),node_bytes,cudaMemcpyHostToDevice)==cudaSuccess,"chain nodes upload");
    }
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_chain_route,&l3_chain_route_storage[gpu],sizeof(uint64_t*))==cudaSuccess,"chain route bind");
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_chain_nodes,&l3_chain_node_storage[gpu],sizeof(l3_chain_partition_node*))==cudaSuccess,"chain nodes bind");
    printf("L3_CHAIN_PARTITION_UPLOAD gpu=%d bytes=%zu upload_ms=%.6f\n",gpu,route_bytes+node_bytes,mlmq_bench_ms()-start);
}
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
