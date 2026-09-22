#pragma once

// L3_RX_L2_PULL diagnostics are intentionally a small device-local counter
// array.  The production policy does not allocate or touch this array unless
// the diagnostic build explicitly enables it.
enum l3_rx_l2_pull_stat_index
{
    L3_RX_L2_PULL_RX_EVENTS = 0,
    L3_RX_L2_PULL_HINT_SEEN,
    L3_RX_L2_PULL_PULL_ATTEMPT,
    L3_RX_L2_PULL_PULL_RECORDS,
    L3_RX_L2_PULL_EMPTY_PULL,
    L3_RX_L2_PULL_ORDINARY_AFTER_PULL,
    // Sum of q2.get_queue_size() observed at eligible pull attempts.  This is
    // the evidence that the pull was extra work available in the old L2 path,
    // rather than a read caused by an empty queue.
    L3_RX_L2_PULL_OLD_INFLIGHT,
    L3_RX_L2_PULL_L2_AVAILABLE,
    // The corresponding pre-read state for pulls that actually returned
    // records.  These counters disambiguate an available L2 queue from an
    // empty pull caused by a racing worker.
    L3_RX_L2_PULL_OLD_INFLIGHT_WITH_RECORD,
    L3_RX_L2_PULL_L2_AVAILABLE_WITH_RECORD,
    L3_RX_L2_PULL_CLAIM_LOST,
    L3_RX_L2_PULL_STATS_COUNT
};

#if (L3_RX_L2_PULL_DIAG == true)
__device__ __forceinline__ void l3_rx_l2_pull_add_stat(
    unsigned long long *stats, int index, unsigned long long value = 1ull)
{
    if (stats != nullptr)
        atomicAdd(stats + index, value);
}
#endif
