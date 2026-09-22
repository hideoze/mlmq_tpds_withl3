#pragma once
#if (L3_RECOVERY_DOMAIN_DIAG == true)
#if (L3_BOUNDARY_INDEX == false || L3_ACK_SCAN == false || L3_WORKER_RECOVERY == false || L3_COMPLETED_ROWS == true)
#error "recovery domain observation requires fixed-owner boundary indexing and unmodified frozen ACK recovery"
#endif
#include <vector>
#include <cstdio>
#include <cstdlib>

static constexpr int L3_DOMAIN_EXAMPLES=32;
struct l3_domain_example { int local, distance, processed, request; };
struct l3_domain_stats {
    unsigned long long checked, inside, outside, helpers;
    unsigned example_count;
    l3_domain_example examples[L3_DOMAIN_EXAMPLES];
};
__device__ unsigned *g_l3_incoming_words=nullptr;
__device__ l3_domain_stats g_l3_domain_stats={};
static unsigned *l3_incoming_storage[MAX_GPU]={};
static int l3_incoming_sizes[MAX_GPU]={};
static void l3_domain_check(cudaError_t e) {
    if(e!=cudaSuccess) {fprintf(stderr,"recovery domain CUDA error: %s\n",cudaGetErrorString(e));std::exit(2);}
}
static void l3_domain_release(int gpu) {
    l3_domain_check(cudaSetDevice(gpu));
    unsigned *none=nullptr;
    l3_domain_check(cudaMemcpyToSymbol(g_l3_incoming_words,&none,sizeof(none)));
    if(l3_incoming_storage[gpu]) l3_domain_check(cudaFree(l3_incoming_storage[gpu]));
    l3_incoming_storage[gpu]=nullptr;l3_incoming_sizes[gpu]=0;
}
static void l3_domain_reset() {
    l3_domain_stats zero={};
    l3_domain_check(cudaMemcpyToSymbol(g_l3_domain_stats,&zero,sizeof(zero)));
}
static void l3_domain_setup(int gpu,int vertices,const std::vector<int> &incoming) {
    l3_domain_release(gpu);
    const int words=(vertices+31)/32;
    std::vector<unsigned> membership((words+31)/32,0);
    size_t covered=0;
    for(int w:incoming) {
        if(w<0||w>=words) {fprintf(stderr,"incoming word outside owner domain\n");std::exit(2);}
        membership[w>>5]|=1u<<(w&31);
        covered+=std::min(32,vertices-32*w);
    }
    if(!membership.empty()) {
        l3_domain_check(cudaMalloc(&l3_incoming_storage[gpu],membership.size()*sizeof(unsigned)));
        l3_domain_check(cudaMemcpy(l3_incoming_storage[gpu],membership.data(),membership.size()*sizeof(unsigned),cudaMemcpyHostToDevice));
    }
    l3_domain_check(cudaMemcpyToSymbol(g_l3_incoming_words,&l3_incoming_storage[gpu],sizeof(unsigned*)));
    l3_incoming_sizes[gpu]=vertices;l3_domain_reset();
    printf("L3_RECOVERY_DOMAIN_SETUP gpu=%d vertices=%d incoming_words=%zu covered_vertices=%zu bytes=%zu\n",gpu,vertices,incoming.size(),covered,membership.size()*sizeof(unsigned));
}
__device__ __forceinline__ bool l3_incoming_word(int local) {
    const unsigned w=unsigned(local-1)>>5;
    return g_l3_incoming_words && (g_l3_incoming_words[w>>5]&(1u<<(w&31)));
}
// One disjoint pass per helper, before the original full repair. Only called
// after all local producer ACKs. No changes to distances, progress, or dirty.
__device__ __noinline__ void l3_domain_observe(
    const int *distances,const int *processed,int begin,int end,int lane,int request) {
    unsigned long long checked=0,inside=0,outside=0;
    for(int v=begin+lane;v<end;v+=32) {
        const int d=*((volatile const int *)&distances[v]);
        const int p=*((volatile const int *)&processed[v]);
        ++checked;
        if(d>=p) continue;
        if(l3_incoming_word(v)) ++inside;
        else {
            ++outside;
            const unsigned slot=atomicAdd(&g_l3_domain_stats.example_count,1u);
            if(slot<L3_DOMAIN_EXAMPLES)g_l3_domain_stats.examples[slot]={v,d,p,request};
        }
    }
    for(int shift=16;shift;shift>>=1) {
        checked+=__shfl_down_sync(0xffffffffu,checked,shift);
        inside+=__shfl_down_sync(0xffffffffu,inside,shift);
        outside+=__shfl_down_sync(0xffffffffu,outside,shift);
    }
    if(!lane) {
        atomicAdd(&g_l3_domain_stats.checked,checked);
        atomicAdd(&g_l3_domain_stats.inside,inside);
        atomicAdd(&g_l3_domain_stats.outside,outside);
        atomicAdd(&g_l3_domain_stats.helpers,1ull);
    }
}
static void l3_domain_report(int gpu) {
    l3_domain_stats s={};
    l3_domain_check(cudaMemcpyFromSymbol(&s,g_l3_domain_stats,sizeof(s)));
    printf("L3_RECOVERY_DOMAIN gpu=%d vertices=%d checked=%llu inside=%llu outside=%llu helpers=%llu example_count=%u\n",
           gpu,l3_incoming_sizes[gpu],s.checked,s.inside,s.outside,s.helpers,s.example_count);
    for(unsigned i=0;i<s.example_count && i<L3_DOMAIN_EXAMPLES;++i) {
        const auto &e=s.examples[i];
        printf("L3_RECOVERY_OUTSIDE gpu=%d local=%d distance=%d processed=%d request=%d\n",gpu,e.local,e.distance,e.processed,e.request);
    }
}
#endif
