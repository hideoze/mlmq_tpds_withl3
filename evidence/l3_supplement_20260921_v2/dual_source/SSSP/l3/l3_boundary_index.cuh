#pragma once

#if (L3_BOUNDARY_INDEX == true)
#if (BULK_ROUND == false || L3_TERM_HANDSHAKE == false || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || BULK_EPOCH == true || SEED_BARRIER == true || L3_TILE_LOAN == true || GHOST_DEPTH > 0 || SEED_EXP == true)
#error "boundary index requires fixed-owner ordinary BULK without extra edge producers"
#endif
#include <vector>
#include <chrono>
#if (L3_COMPACT_CANDIDATES == true && GLOBAL_ROUND_PEER_CACHE_FEEDBACK == true)
#error "compact candidates do not support peer cache feedback in dense owner IDs"
#endif

// These are immutable for a graph/query family. No hint or event controls
// membership: every target of every local-to-peer edge is included.
__device__ int *g_l3_boundary_words = nullptr;
__device__ int g_l3_boundary_word_count = -1;
static int *l3_boundary_storage[MAX_GPU] = {};
#if (L3_RECOVERY_DOMAIN_DIAG == true)
// Complete sender-to-peer word domain, even if dense scanning was selected.
// After both builds, the opposite sender's list defines this owner's input.
static std::vector<int> l3_boundary_host_words[MAX_GPU];
#endif
#if (L3_COMPACT_CANDIDATES == true)
// Dense peer-local -> compact lookup is used only on crossing edges. Candidate
// arrays retain their old allocation size in this prototype, but only the
// compact prefix is addressed. The reverse lookup is used exactly once before
// a batch becomes a wire-format retained journal.
__device__ int g_l3_candidate_count = -1;
__device__ int *g_l3_candidate_lookup = nullptr;
__device__ int *g_l3_candidate_reverse = nullptr;
static int *l3_candidate_lookup_storage[MAX_GPU] = {};
static int *l3_candidate_reverse_storage[MAX_GPU] = {};
#endif

static void l3_boundary_check(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr,"boundary index CUDA error: %s\n",cudaGetErrorString(e));
        std::exit(2);
    }
}

static void l3_boundary_release(int gpu) {
    l3_boundary_check(cudaSetDevice(gpu));
    int disabled=-1;
    int *empty=nullptr;
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_boundary_word_count,&disabled,sizeof(disabled)));
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_boundary_words,&empty,sizeof(empty)));
    if (l3_boundary_storage[gpu]) l3_boundary_check(cudaFree(l3_boundary_storage[gpu]));
    l3_boundary_storage[gpu]=nullptr;
#if (L3_RECOVERY_DOMAIN_DIAG == true)
    l3_boundary_host_words[gpu].clear();
#endif
#if (L3_COMPACT_CANDIDATES == true)
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_candidate_count,&disabled,sizeof(disabled)));
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_candidate_lookup,&empty,sizeof(empty)));
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_candidate_reverse,&empty,sizeof(empty)));
    if(l3_candidate_lookup_storage[gpu]) l3_boundary_check(cudaFree(l3_candidate_lookup_storage[gpu]));
    if(l3_candidate_reverse_storage[gpu]) l3_boundary_check(cudaFree(l3_candidate_reverse_storage[gpu]));
    l3_candidate_lookup_storage[gpu]=nullptr;
    l3_candidate_reverse_storage[gpu]=nullptr;
#endif
}

static void l3_boundary_build(int gpu, const int *rows, const int *cols,
                              int local_vertices, int peer_begin, int peer_size) {
    const auto start=std::chrono::steady_clock::now();
    l3_boundary_release(gpu);
    int edges=0;
    l3_boundary_check(cudaMemcpy(&edges,rows+local_vertices,sizeof(edges),cudaMemcpyDeviceToHost));
    if (edges<0 || peer_size<0) std::exit(2);
    const int words=(peer_size+31)/32;
    std::vector<unsigned char> seen(words,0);
#if (L3_COMPACT_CANDIDATES == true)
    std::vector<int> lookup(peer_size,-1), reverse;
#endif
    std::vector<int> buffer(1<<20), index;
    for (int begin=0;begin<edges;) {
        int count=std::min(int(buffer.size()),edges-begin);
        l3_boundary_check(cudaMemcpy(buffer.data(),cols+begin,size_t(count)*sizeof(int),cudaMemcpyDeviceToHost));
        for (int j=0;j<count;++j) {
            int local=buffer[j]-peer_begin;
            if (local>=0 && local<peer_size) {
                seen[local>>5]=1;
#if (L3_COMPACT_CANDIDATES == true)
                lookup[local]=0;
#endif
            }
        }
        begin+=count;
    }
    for (int word=0;word<words;++word) if (seen[word]) index.push_back(word);
#if (L3_RECOVERY_DOMAIN_DIAG == true)
    l3_boundary_host_words[gpu]=index;
#endif
    // Dense graphs retain the original contiguous scan, avoiding an extra
    // indirection when indexing would not meaningfully reduce its domain.
    int count=index.size()<size_t(words)/2 ? int(index.size()) : -1;
    if (count>0) {
        l3_boundary_check(cudaMalloc(&l3_boundary_storage[gpu],size_t(count)*sizeof(int)));
        l3_boundary_check(cudaMemcpy(l3_boundary_storage[gpu],index.data(),size_t(count)*sizeof(int),cudaMemcpyHostToDevice));
    }
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_boundary_words,&l3_boundary_storage[gpu],sizeof(int*)));
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_boundary_word_count,&count,sizeof(count)));
#if (L3_COMPACT_CANDIDATES == true)
    for(int v=0;v<peer_size;++v) if(lookup[v]==0) {
        lookup[v]=int(reverse.size())+1;
        reverse.push_back(v+1);
    }
    // Keep ordinary addressing for dense crossing domains, without paying a
    // lookup on every crossing edge for negligible scanning benefit.
    int compact=reverse.size()<size_t(peer_size)/2 ? int(reverse.size()) : -1;
    if(compact>0) {
        l3_boundary_check(cudaMalloc(&l3_candidate_lookup_storage[gpu],lookup.size()*sizeof(int)));
        l3_boundary_check(cudaMemcpy(l3_candidate_lookup_storage[gpu],lookup.data(),lookup.size()*sizeof(int),cudaMemcpyHostToDevice));
        l3_boundary_check(cudaMalloc(&l3_candidate_reverse_storage[gpu],reverse.size()*sizeof(int)));
        l3_boundary_check(cudaMemcpy(l3_candidate_reverse_storage[gpu],reverse.data(),reverse.size()*sizeof(int),cudaMemcpyHostToDevice));
    }
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_candidate_lookup,&l3_candidate_lookup_storage[gpu],sizeof(int*)));
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_candidate_reverse,&l3_candidate_reverse_storage[gpu],sizeof(int*)));
    l3_boundary_check(cudaMemcpyToSymbol(g_l3_candidate_count,&compact,sizeof(compact)));
    printf("L3_COMPACT_CANDIDATES gpu=%d peer_vertices=%d boundary_vertices=%zu compact=%d words=%d\n",gpu,peer_size,reverse.size(),compact>=0,compact>=0?(compact+31)/32:words);
#endif
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    printf("L3_BOUNDARY_INDEX gpu=%d dense_words=%d boundary_words=%zu indexed=%d build_ms=%.6f\n",gpu,words,index.size(),count>=0,ms);
}
#endif

__device__ __forceinline__ int l3_scan_word_count(int dense_words) {
#if (L3_COMPACT_CANDIDATES == true)
    if(g_l3_candidate_count>=0) return (g_l3_candidate_count+31)/32;
#endif
#if (L3_BOUNDARY_INDEX == true)
    return g_l3_boundary_word_count>=0 ? g_l3_boundary_word_count : dense_words;
#else
    return dense_words;
#endif
}
__device__ __forceinline__ int l3_scan_word(int index) {
#if (L3_COMPACT_CANDIDATES == true)
    if(g_l3_candidate_count>=0) return index;
#endif
#if (L3_BOUNDARY_INDEX == true)
    return g_l3_boundary_word_count>=0 ? g_l3_boundary_words[index] : index;
#else
    return index;
#endif
}

__device__ __forceinline__ int l3_candidate_size(int peer_vertices) {
#if (L3_COMPACT_CANDIDATES == true)
    if(g_l3_candidate_count>=0) return g_l3_candidate_count;
#endif
    return peer_vertices;
}

__device__ __forceinline__ int l3_candidate_index(int peer_local_one_based) {
#if (L3_COMPACT_CANDIDATES == true)
    if(g_l3_candidate_count>=0) {
        const int slot=g_l3_candidate_lookup[peer_local_one_based-1];
        assert(slot>0 && slot<=g_l3_candidate_count);
        return slot;
    }
#endif
    return peer_local_one_based;
}

__device__ __forceinline__ int l3_candidate_peer_index(int compact_one_based) {
#if (L3_COMPACT_CANDIDATES == true)
    if(g_l3_candidate_count>=0) {
        assert(compact_one_based>0 && compact_one_based<=g_l3_candidate_count);
        return g_l3_candidate_reverse[compact_one_based-1];
    }
#endif
    return compact_one_based;
}
