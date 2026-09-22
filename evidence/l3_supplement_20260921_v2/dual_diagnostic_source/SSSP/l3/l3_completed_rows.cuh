#pragma once
#if (L3_COMPLETED_ROWS == true)
#include <cuda/atomic>
extern __device__ VALUE_TYPE *g_l3_completed_rows;

// Separate from last_processed (which ordinary workers write BEFORE edges).
// Call only after all outputs of the certified row have durable responsibility.
__device__ __forceinline__ void l3_publish_completed_row(int local,VALUE_TYPE d)
{
    if(d==DIST_MAX) return;
    cuda::atomic_ref<VALUE_TYPE,cuda::thread_scope_device> a(g_l3_completed_rows[local]);
    VALUE_TYPE old=a.load(cuda::memory_order_relaxed);
    while(d<old && !a.compare_exchange_weak(old,d,cuda::memory_order_release,
                                           cuda::memory_order_relaxed)) {}
}
__device__ __forceinline__ VALUE_TYPE l3_recovery_threshold(int local,VALUE_TYPE ordinary)
{
    if(!g_l3_completed_rows) return ordinary;
    VALUE_TYPE completed=cuda::atomic_ref<VALUE_TYPE,cuda::thread_scope_device>(
        g_l3_completed_rows[local]).load(cuda::memory_order_acquire);
    return min(ordinary,completed);
}
__device__ __forceinline__ bool l3_recovery_needs_work(int local,VALUE_TYPE distance,VALUE_TYPE ordinary)
{
    // Preserve the coalesced ordinary scan. Acquire evidence only on the rare
    // would-be-dirty path; never do an acquire load for every settled vertex.
    if(distance>=ordinary) return false;
    return distance<l3_recovery_threshold(local,ordinary);
}
#endif
