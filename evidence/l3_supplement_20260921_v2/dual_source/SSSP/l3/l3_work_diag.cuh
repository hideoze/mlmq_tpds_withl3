#pragma once
#if (L3_WORK_DIAG == true)
// Counters are lane-private until one final warp reduction. Clocks are only
// subtracted within the same worker warp; they are not a GPU-global timeline.
struct l3_work_metrics {
    unsigned long long popped, expanded, edges;
    unsigned long long owner_destination_edges, cross_owner_destination_edges;
    unsigned long long calls, active, first, last, span;
    // Starvation attribution: read attempts that returned nothing while the
    // global queue still had items (warp raced others for sparse buckets) vs
    // iterations with no queue work at all (waiting on peer/termination).
    unsigned long long empty_reads, idle_iters, busy_iters;
};
static constexpr int L3_WORK_SLOTS = 4096;
__device__ l3_work_metrics g_l3_work_metrics[L3_WORK_SLOTS];

// Per-query source expansion accounting.  The array is indexed by local
// owner vertex; repeated_edges counts the CSR row length for every expansion
// after the first, so it measures repeated edge work rather than duplicate
// queue records alone.  This exists only in the diagnostic build.
struct l3_source_expand_metrics {
    unsigned long long unique_sources;
    unsigned long long repeated_sources;
    unsigned long long repeated_edges;
};
__device__ l3_source_expand_metrics g_l3_source_expand_metrics;
#endif
