#pragma once
#include "l3_mapping_metrics.cuh"

#ifndef L3_ADAPTIVE_WORDS
#define L3_ADAPTIVE_WORDS false
#endif

// Warp-uniform choice: compare the number of collective emission passes.
__device__ __forceinline__ void l3_word_shape(unsigned bits, int &occupied, int &longest)
{
    longest = __popc(bits);
    for (int offset = 16; offset; offset >>= 1) {
        int other = __shfl_xor_sync(FULL_MASK, longest, offset);
        longest = longest > other ? longest : other;
    }
    occupied = __popc(__ballot_sync(FULL_MASK, bits != 0));
}

__device__ __forceinline__ bool l3_use_word_team(unsigned bits)
{
    int occupied, longest;
    l3_word_shape(bits, occupied, longest);
    return occupied < longest;
}

__device__ __forceinline__ int l3_emit_candidate(
    int id, int *candidate, int *cache, int size, int count,
    int *ids, VALUE_TYPE *values, int lane)
{
    int value = id > 0 && id <= size ? atomicExch(candidate + id, DIST_MAX) : DIST_MAX;
    bool valid = id > 0 && id <= size && value != DIST_MAX;
    unsigned winners = __ballot_sync(FULL_MASK, valid);
    int position = __popc(winners & ((1u << lane) - 1u));
    if (valid) {
#if (BULK_NO_CACHE == false)
        atomicMin(cache + id, value);
#endif
        ids[count + position] = id;
        values[count + position] = value;
    }
    return count + __popc(winners);
}

// One warp selects nonempty hint blocks, then shares their 32 mark words.
// Reserve room for a whole 1024-vertex block before claiming any marks.
// Hints remain advisory; the caller retains periodic authoritative full scans.
__device__ __forceinline__ int l3_collect_cooperative(
    int *candidate, unsigned *mark, unsigned *hint, unsigned *hint2,
    int *cache, int size, int capacity, int *ids, VALUE_TYPE *values, int lane)
{
    const int words = (size + 31) / 32;
    const int hints = (words + 31) / 32;
    int count = 0;
    const int summaries = (hints + 31) / 32;
    for (int upper = 0; upper < summaries; upper += 32) {
        bool live = upper + lane < summaries
            && (!hint2 || atomicAdd(hint2 + upper + lane, 0u) != 0);
        unsigned blocks = __ballot_sync(FULL_MASK, live);
        while (blocks) {
        int base = (upper + __ffs(blocks) - 1) * 32;
        unsigned snapshot = base + lane < hints ? atomicAdd(hint + base + lane, 0u) : 0;
        unsigned active = __ballot_sync(FULL_MASK, snapshot != 0);
        while (active) {
            if (capacity - count < 1024) return count;
            int selected = __ffs(active) - 1;
            int h = base + selected;
            unsigned old_hint = __shfl_sync(FULL_MASK, snapshot, selected);
            int word = h * 32 + lane;
#if (L3_MAPPING_DIAG == true)
            unsigned long long mapping_start = clock64();
            int count_before = count;
#endif
            unsigned bits = word < words ? atomicAdd(mark + word, 0u) : 0;
            if (bits && !l3_device_mark_claim(mark + word, bits)) bits = 0;
            // Selection happens only after claiming. Both mappings consume the
            // exact same owned bits, with an unchanged whole-block reservation.
#if (L3_ADAPTIVE_WORDS == true || L3_MAPPING_DIAG == true)
            int k, m;
            l3_word_shape(bits, k, m);
            bool team = L3_ADAPTIVE_WORDS && k < m;
#endif
#if (L3_ADAPTIVE_WORDS == true)
            if (team) {
                unsigned occupied = __ballot_sync(FULL_MASK, bits != 0);
                while (occupied) {
                    int owner = __ffs(occupied) - 1;
                    unsigned owned = __shfl_sync(FULL_MASK, bits, owner);
                    int owned_word = __shfl_sync(FULL_MASK, word, owner);
                    int id = (owned & (1u << lane)) ? owned_word * 32 + lane + 1 : 0;
                    count = l3_emit_candidate(id, candidate, cache, size, count, ids, values, lane);
                    occupied &= occupied - 1;
                }
                bits = 0;
            }
#endif
            // Sparse-word mapping: one candidate per lane per emission pass.
            while (__any_sync(FULL_MASK, bits != 0)) {
                int id = 0;
                if (bits) {
                    int bit = __ffs(bits) - 1;
                    bits &= bits - 1;
                    id = word * 32 + bit + 1;
                }
                count = l3_emit_candidate(id, candidate, cache, size, count, ids, values, lane);
            }
            __syncwarp();
#if (L3_MAPPING_DIAG == true)
            if (!lane) {
                unsigned long long cycles = clock64() - mapping_start;
                auto &d = g_l3_mapping_metrics;
                ++d.blocks;
                d.eligible += k < m;
                d.chosen += team;
                d.static_passes += m;
                d.selected_passes += team ? k : m;
                d.emitted += count - count_before;
                d.block_cycles += cycles;
                (team ? d.team_cycles : d.lane_cycles) += cycles;
                ++d.k_hist[k];
                ++d.m_hist[m];
            }
#endif
            bool empty = word >= words || atomicAdd(mark + word, 0u) == 0;
            bool all_empty = __all_sync(FULL_MASK, empty);
            if (!lane && all_empty) {
                atomicCAS(hint + h, old_hint, 0u);
                if (hint2 && atomicAdd(hint + h, 0u) == 0) {
                    unsigned bit = 1u << (h & 31);
                    atomicAnd(hint2 + (h >> 5), ~bit);
                    if (atomicAdd(hint + h, 0u) != 0)
                        atomicOr(hint2 + (h >> 5), bit);
                }
            }
            __syncwarp();
            active &= active - 1;
        }
        blocks &= blocks - 1;
        }
    }
    return count;
}
