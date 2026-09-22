#pragma once
#ifndef L3_MAPPING_DIAG
#define L3_MAPPING_DIAG false
#endif
#if (L3_MAPPING_DIAG == true)
// One TX lane writer per device. Host reads only after kernel completion.
// Deliberately independent of worker edge counters in L3_DIAGNOSTICS.
struct l3_mapping_metrics {
    unsigned long long blocks, eligible, chosen;
    unsigned long long static_passes, selected_passes, emitted;
    unsigned long long block_cycles, team_cycles, lane_cycles;
    unsigned long long k_hist[33], m_hist[33];
};
__device__ l3_mapping_metrics g_l3_mapping_metrics;
#endif
