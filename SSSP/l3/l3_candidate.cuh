#pragma once

#include "l3_sync.cuh"

// Candidate-side L3 primitive.  This helper owns only the local aggregation
// and event publication; it never performs a peer-memory write or waits for a
// transport operation.  remote_cand/remote_mark remain the authoritative
// state, while hints and counters are optional indexes/telemetry.
__device__ __forceinline__ bool l3_record_candidate(
    VALUE_TYPE *remote_cand, unsigned *remote_mark,
    unsigned *mark_hint, unsigned *mark_hint2,
    int r0, VALUE_TYPE new_dist,
    VALUE_TYPE *old_out, int *remote_pending,
    unsigned long long *mark_signal, int *local_idle)
{
    if (remote_cand == NULL || remote_mark == NULL
        || r0 < 0)
        return false;

    const int lidx = r0 + 1;
    VALUE_TYPE old_dist;
#ifdef TYPE_INT
    old_dist = atomicMin(&remote_cand[lidx], new_dist);
#else
    old_dist = atomicMin_float(&remote_cand[lidx], new_dist);
#endif
    if (old_out != NULL)
        *old_out = old_dist;
    if (new_dist >= old_dist)
        return false;

    const unsigned bit = 1u << (r0 & 31);
    const unsigned old_mark = l3_device_mark_publish(&remote_mark[r0 >> 5], bit);
    if (mark_hint != NULL)
        atomicOr(&mark_hint[r0 >> 10], 1u << ((r0 >> 5) & 31));
    if (mark_hint2 != NULL)
        atomicOr(&mark_hint2[r0 >> 15], 1u << ((r0 >> 10) & 31));

    // pending and signal are wake-up/telemetry aids only.  The mark bitmap
    // remains authoritative if either auxiliary signal is stale.
    if (remote_pending != NULL && !(old_mark & bit))
        atomicAdd(remote_pending, 1);
    if (mark_signal != NULL)
        atomicAdd(mark_signal, 1ull);
    if (local_idle != NULL)
        l3_atomic_store_release<cuda::thread_scope_system>(local_idle, 0);
    return true;
}
