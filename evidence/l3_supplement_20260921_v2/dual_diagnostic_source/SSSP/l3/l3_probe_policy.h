#pragma once
#ifndef L3_DEFER_BUSY_PROBE
#define L3_DEFER_BUSY_PROBE false
#endif
#ifdef __CUDACC__
#define L3_PROBE_HD __host__ __device__
#else
#define L3_PROBE_HD
#endif
// Scheduling only. At most 31 eligible busy observations may defer a full
// check; a quiet observation immediately restores the existing check path.
struct l3_probe_policy {
    unsigned skipped = 0;
    L3_PROBE_HD bool defer(bool busy) {
        if (!busy) { skipped = 0; return false; }
        if (++skipped < 32) return true;
        skipped = 0;
        return false;
    }
};
#undef L3_PROBE_HD
