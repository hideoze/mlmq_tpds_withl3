#pragma once
#if (L3_WORKER_RECOVERY == true)
#include <cuda/atomic>
#if (L3_ACK_WIDE_SCAN == true && L3_ACK_SCAN == false)
#error "Wide recovery requires the frozen ACK scan path"
#endif
#if (BULK_ROUND == false || L3_TERM_HANDSHAKE == false || L3_DIRECT_RX == false || L3_RETAIN_TX == false || L3_RECOVERY_MODE != 0 || L3_BOUNDED_RECOVERY == true || L3_CONFIRM_BACKOFF == true || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || BULK_EPOCH == true || SEED_BARRIER == true)
#error "worker recovery requires unmodified default BULK correctness/termination"
#endif

// One manager issues requests on each device. Query completion leaves request
// zero; epoch remains monotonic across workspace reuse. No peer GPU atomics.
struct l3_worker_recovery_control {
    int request, done, found, epoch;
    unsigned *dirty, *hint;
};
__device__ l3_worker_recovery_control g_worker_recovery = {};
using l3_local_atomic = cuda::atomic_ref<int, cuda::thread_scope_device>;

__device__ __forceinline__ bool backstop_scan_range(
    VALUE_TYPE *, VALUE_TYPE *, unsigned *, unsigned *, int, int, int);

// Out-of-line scan keeps scratch registers out of the hot worker expansion.
// Caller is a selected frozen worker; legacy requests select block warp0.
__device__ __noinline__ void l3_worker_recovery_service(
    VALUE_TYPE *distances, VALUE_TYPE *processed, int size,
    int rank, int blocks, int lane, int request)
{
    // Acquire in all participating lanes before reading published pointers.
    if(lane) {
        // The acquire is required even in a build with assertions disabled.
        const int observed=l3_local_atomic(g_worker_recovery.request).load(cuda::memory_order_acquire);
        assert(observed==request);
    }
    auto part=l3_recovery_slice(size,rank,blocks);
#if (L3_RECOVERY_DOMAIN_DIAG == true)
    l3_domain_observe(distances,processed,part.begin,part.end,lane,request);
#endif
    bool found=backstop_scan_range(distances,processed,g_worker_recovery.dirty,
                                  g_worker_recovery.hint,part.begin,part.end,lane);
    unsigned found_mask=__ballot_sync(FULL_MASK,found);
#if (L3_ACK_SCAN_FAULT == true)
    if(rank==blocks-1) {
        const unsigned long long started=clock64();
        while(clock64()-started<250000ull) {}
        if(!lane) ++g_l3_progress.recovery_delays;
    }
#endif
    // Each writer's dirty/hint stores precede the leader's completion publish.
    __threadfence();__syncwarp();
    if(!lane) {
        if(found_mask)l3_local_atomic(g_worker_recovery.found).fetch_or(1,cuda::memory_order_relaxed);
        l3_local_atomic(g_worker_recovery.done).fetch_add(1,cuda::memory_order_acq_rel);
    }
    __syncwarp();
}

// No call-frame/register save on the common no-request path. Keep the cursor
// in the caller, not an address-taken local written by every service poll.
__device__ __forceinline__ void l3_worker_recovery_poll(
    VALUE_TYPE *distances, VALUE_TYPE *processed, int size,
    int rank, int blocks, int lane, int &seen, int local_warp = 0)
{
    int request=0;
    if(!lane)request=l3_local_atomic(g_worker_recovery.request).load(cuda::memory_order_acquire);
    request=__shfl_sync(FULL_MASK,request,0);
    if(!request||request==seen)return;
#if (L3_ACK_WIDE_SCAN == true)
    // Mode is part of the acquired token: a nonparticipant in a legacy
    // request must not read mutable metadata from the next generation.
    if(request & 1) {
        rank=rank*WARP_NUM_PER_BLOCK+local_warp;
        blocks*=WARP_NUM_PER_BLOCK;
    } else {
        if(local_warp!=0)return;
    }
#endif
    // Update before completion is published; a new generation may arrive
    // immediately after this warp finishes its previous slice.
    seen=request;
    l3_worker_recovery_service(distances,processed,size,rank,blocks,lane,request);
}

__device__ __forceinline__ bool l3_worker_recovery_request(
    unsigned *dirty,unsigned *hint,int blocks,int lane,bool wide=false)
{
    if(!lane) {
        assert(blocks>0);
        assert(l3_local_atomic(g_worker_recovery.request).load(cuda::memory_order_acquire)==0);
        assert(g_worker_recovery.epoch<INT_MAX);
        g_worker_recovery.dirty=dirty;g_worker_recovery.hint=hint;
        l3_local_atomic(g_worker_recovery.found).store(0,cuda::memory_order_relaxed);
        l3_local_atomic(g_worker_recovery.done).store(0,cuda::memory_order_relaxed);
#if (L3_ACK_WIDE_SCAN == true)
        assert(g_worker_recovery.epoch<INT_MAX/2);
        const int token=2*(++g_worker_recovery.epoch)+int(wide);
#else
        const int token=++g_worker_recovery.epoch;
#endif
        l3_local_atomic(g_worker_recovery.request).store(token,cuda::memory_order_release);
    }
    __syncwarp();
    bool found=false;
    if(!lane) {
        while(l3_local_atomic(g_worker_recovery.done).load(cuda::memory_order_acquire)<blocks){}
        assert(l3_local_atomic(g_worker_recovery.done).load(cuda::memory_order_acquire)==blocks);
        found=l3_local_atomic(g_worker_recovery.found).load(cuda::memory_order_relaxed)!=0;
        l3_local_atomic(g_worker_recovery.request).store(0,cuda::memory_order_release);
    }
    return __shfl_sync(FULL_MASK,int(found),0)!=0;
}
#endif
