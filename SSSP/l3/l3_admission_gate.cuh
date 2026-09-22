#pragma once
#include <cuda/atomic>

// Prototype primitive, not wired into the production worker yet.
// [epoch:32][closed:1][registered producers:31]. Registration is not q2 debt:
// a producer must publish a durable slot or retain local work before leaving.
// A coordinator may acknowledge quiescence only after BOTH gates are closed,
// registrations are zero, slots are drained, and ordinary work is quiescent.
// Epoch initialization/replacement is host-only after all GPU users join.
// Caller owns one exactly-once leave per successful enter. The aggregate count
// cannot detect duplicate leave while another producer remains registered.
// Epoch zero is invalid; do not wrap/reuse epochs within a live query.
using l3_gate_word = unsigned long long;
static constexpr l3_gate_word L3_GATE_CLOSED = 1ull << 31;
static constexpr l3_gate_word L3_GATE_COUNT = L3_GATE_CLOSED - 1;
static_assert(sizeof(l3_gate_word)==8,"gate requires 64-bit atomics");

__host__ __device__ constexpr l3_gate_word l3_gate_initial(unsigned epoch) {
    return l3_gate_word(epoch) << 32;
}
__device__ __forceinline__ l3_gate_word l3_gate_load(l3_gate_word *p) {
    return cuda::atomic_ref<l3_gate_word,cuda::thread_scope_system>(*p)
        .load(cuda::memory_order_acquire);
}
__device__ __forceinline__ bool l3_gate_enter(l3_gate_word *p,unsigned epoch) {
    auto a=cuda::atomic_ref<l3_gate_word,cuda::thread_scope_system>(*p);
    auto old=a.load(cuda::memory_order_acquire);
    for (;;) {
        if (!epoch || unsigned(old>>32)!=epoch || (old&L3_GATE_CLOSED) ||
            (old&L3_GATE_COUNT)==L3_GATE_COUNT) return false;
        if(a.compare_exchange_weak(old,old+1,cuda::memory_order_acq_rel,
                                   cuda::memory_order_acquire)) return true;
    }
}
__device__ __forceinline__ bool l3_gate_leave(l3_gate_word *p,unsigned epoch) {
    auto a=cuda::atomic_ref<l3_gate_word,cuda::thread_scope_system>(*p);
    auto old=a.load(cuda::memory_order_acquire);
    for (;;) {
        if (!epoch || unsigned(old>>32)!=epoch || !(old&L3_GATE_COUNT)) return false;
        if(a.compare_exchange_weak(old,old-1,cuda::memory_order_acq_rel,
                                   cuda::memory_order_acquire)) return true;
    }
}
__device__ __forceinline__ bool l3_gate_close(l3_gate_word *p,unsigned epoch) {
    auto a=cuda::atomic_ref<l3_gate_word,cuda::thread_scope_system>(*p);
    auto old=a.load(cuda::memory_order_acquire);
    for (;;) {
        if (!epoch || unsigned(old>>32)!=epoch) return false;
        if(old&L3_GATE_CLOSED) return true;
        if(a.compare_exchange_weak(old,old|L3_GATE_CLOSED,cuda::memory_order_acq_rel,
                                   cuda::memory_order_acquire)) return true;
    }
}
__device__ __forceinline__ bool l3_gate_drained(l3_gate_word *p,unsigned epoch) {
    return epoch && l3_gate_load(p)==(l3_gate_initial(epoch)|L3_GATE_CLOSED);
}

// Gates must remain closed for this query. Inspect registrations BEFORE slots:
// after both counts drain, no producer can publish a previously unseen slot.
// This only certifies loan drain; caller still checks ordinary/local work.
__device__ __forceinline__ bool l3_gate_slots_drained(
    l3_gate_word *local,l3_gate_word *peer,unsigned epoch,
    int *out_state,int *in_state,int free_state)
{
    if(!l3_gate_drained(local,epoch) || !l3_gate_drained(peer,epoch)) return false;
    auto out=cuda::atomic_ref<int,cuda::thread_scope_system>(*out_state).load(cuda::memory_order_acquire);
    auto in=cuda::atomic_ref<int,cuda::thread_scope_system>(*in_state).load(cuda::memory_order_acquire);
    return out==free_state && in==free_state;
}
