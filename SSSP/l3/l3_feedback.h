#pragma once
#ifdef __CUDACC__
#define L3_FEEDBACK_HD __host__ __device__
#else
#define L3_FEEDBACK_HD
#endif

// Receiver-owned per slot. RX writes before ACK; TX reads while holding
// WRITING for the next generation of that slot, before publishing new work.
struct l3_feedback_record {
    int epoch;
    int count;
    int improved;
    L3_FEEDBACK_HD bool valid(int expected) const {
        return expected > 0 && epoch == expected && count >= 0 &&
               improved >= 0 && improved <= count;
    }
};
#undef L3_FEEDBACK_HD

#if defined(__CUDACC__) && (L3_RX_FEEDBACK_TRACE == true)
// Bounded diagnostic storage; never interleave device printf with host output.
struct l3_feedback_trace_entry {
    int kind, owner, receiver, tx_epoch;
    l3_feedback_record record;
    unsigned long long before, after;
    int mode;
};
constexpr unsigned L3_FEEDBACK_TRACE_CAPACITY = 4096;
__device__ l3_feedback_trace_entry g_l3_feedback_trace[L3_FEEDBACK_TRACE_CAPACITY];
__device__ unsigned g_l3_feedback_trace_head;
__device__ __forceinline__ void l3_trace_feedback(
    int kind, int owner, int receiver, int tx_epoch,
    l3_feedback_record record, unsigned long long before,
    unsigned long long after, int mode)
{
    unsigned index = atomicAdd(&g_l3_feedback_trace_head, 1u);
    if (index < L3_FEEDBACK_TRACE_CAPACITY)
        g_l3_feedback_trace[index] = {kind, owner, receiver, tx_epoch, record, before, after, mode};
}
#endif
