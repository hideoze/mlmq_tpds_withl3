#pragma once
#ifndef L3_DIAGNOSTICS
#define L3_DIAGNOSTICS false
#endif
struct l3_diagnostics {
    unsigned long long edges, vertices, tx_batches, adjacent_descents;
    unsigned long long remote_attempts, cache_filtered, candidate_updates;
    struct wave { int count, minimum, maximum; } waves[32];
    unsigned long long scans, empty, full, extracted, scan_cycles;
    unsigned long long rx_batches, received, improved;
    unsigned long long budget_sum, budget_changes;
    unsigned long long feedback_drains;
    unsigned long long feedback_batches, feedback_count, feedback_improved, feedback_changes;
    unsigned long long event_skips, event_full;
    unsigned long long injected_publish_retries;
};
#if (L3_DIAGNOSTICS == true)
// TX fields have one lane writer; RX fields have a separate one lane writer.
// edges/vertices use warp-reduced atomicAdd from workers; remote_attempts,
// cache_filtered and candidate_updates use per-attempt atomicAdd (diagnostic
// perturbation). Other fields have a single writer. Host reads after kernels.
__device__ l3_diagnostics g_l3_diagnostics;
#endif
