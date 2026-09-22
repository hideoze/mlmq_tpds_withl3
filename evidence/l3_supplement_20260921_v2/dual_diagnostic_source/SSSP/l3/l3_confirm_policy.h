#pragma once

#ifndef L3_CONFIRM_BACKOFF
#define L3_CONFIRM_BACKOFF false
#endif

// Scheduling-only policy. The caller still performs the periodic authoritative
// backstop and the final frozen producer/mark/transport checks.
struct l3_confirm_policy {
    unsigned skipped = 0;

    __host__ __device__ bool allow(bool pre_ready) {
        if (!pre_ready) {
            skipped = 0;
            return true;
        }
#if (L3_CONFIRM_BACKOFF == true)
        if (++skipped < 32) return false;
        skipped = 0;
#endif
        return true;
    }
};
