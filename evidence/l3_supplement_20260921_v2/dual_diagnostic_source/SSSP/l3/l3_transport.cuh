#pragma once

// L3 transport contract and shared BULK slot ownership primitives. 具体
// pack/apply 编排仍在 sssp_run.cu，先将 generation/ACK 状态机集中于此。
enum l3_transport_kind
{
    L3_TRANSPORT_BULK = 0,
    L3_TRANSPORT_ASYNC = 1
};

struct l3_transport_token
{
    int epoch;
    int generation;
    int slot;
};

#if (BULK_ROUND == true)
// Shared BULK slot ownership primitives.  Apply/pack orchestration remains in
// sssp_run.cu, but the generation/ACK state machine is defined once here so
// every receive path uses the same publish and reuse contract.
#if (L3_FAULT_INJECT_CLAIM_RETRY == true)
extern __device__ unsigned int g_l3_fault_claim_retry;
#endif
#if (L3_FAULT_INJECT_READY_DELAY == true)
extern __device__ unsigned int g_l3_fault_ready_delay;
#endif
#if (L3_FAULT_INJECT_ACK_DELAY == true)
extern __device__ unsigned int g_l3_fault_ack_delay;
extern __device__ int g_l3_fault_ack_pending;
#endif

__device__ __forceinline__ int bulk_inbox_slot(int epoch)
{
    return epoch % BULK_INBOX_SLOTS;
}

__device__ __forceinline__ bool bulk_inbox_try_acquire_write(
    int *slot_state, int *slot_generation, int *slot_ack,
    int slot, int epoch)
{
    if (slot_state == NULL || slot_generation == NULL || slot_ack == NULL)
        return false;

    const int previous_epoch = epoch - BULK_INBOX_SLOTS;
    int state = atomicAdd(slot_state + slot, 0);
    int generation = atomicAdd(slot_generation + slot, 0);
    int ack = atomicAdd(slot_ack + slot, 0);

    bool reusable = (state == BULK_SLOT_FREE);
    if (previous_epoch >= 1)
        reusable = reusable || (state == BULK_SLOT_DONE
                                && generation <= previous_epoch
                                && ack >= previous_epoch);
    if (!reusable)
        return false;

    if (state == BULK_SLOT_DONE)
        atomicExch(slot_state + slot, BULK_SLOT_FREE);
    __threadfence_system();
    if (atomicExch(slot_state + slot, BULK_SLOT_WRITING)
        != BULK_SLOT_FREE)
        return false;

    *((volatile int *)(slot_generation + slot)) = epoch;
    __threadfence_system();
    return true;
}

__device__ __forceinline__ bool bulk_inbox_claim_read(
    int *slot_state, int *slot_generation, int *slot_epoch,
    int slot, int expected_epoch)
{
    if (slot_state == NULL || slot_generation == NULL || slot_epoch == NULL)
        return false;
    int state = atomicAdd(slot_state + slot, 0);
    int generation = atomicAdd(slot_generation + slot, 0);
    int published = atomicAdd(slot_epoch + slot, 0);
    if (state != BULK_SLOT_READY || generation != expected_epoch
        || published != expected_epoch)
        return false;
#if (L3_FAULT_INJECT_CLAIM_RETRY == true)
    // 有界地模拟一次接收端瞬态竞争：不改变 READY/generation，也不写 ACK，
    // 调用方下一轮必须重新走完整的 ready + claim 流程才能消费该 slot。
    if (atomicCAS(&g_l3_fault_claim_retry, 0u, 1u) == 0u)
        return false;
#endif
    return atomicCAS(slot_state + slot, BULK_SLOT_READY, BULK_SLOT_READING)
           == BULK_SLOT_READY;
}

__device__ __forceinline__ void bulk_inbox_finish_read(
    int *slot_state, int *slot_ack, int slot, int epoch)
{
    __threadfence_system();
#if (L3_FAULT_INJECT_ACK_DELAY == true)
    bool delay_ack = false;
    if (slot_ack != NULL
        && atomicCAS(&g_l3_fault_ack_delay, 0u, 1u) == 0u)
    {
        // pending 先于 DONE 发布；manager 下一轮只在看到 DONE 后重发 ACK。
        atomicExch(&g_l3_fault_ack_pending, epoch);
        delay_ack = true;
    }
    if (slot_ack != NULL && !delay_ack)
        atomicExch(slot_ack + slot, epoch);
#else
    if (slot_ack != NULL)
        atomicExch(slot_ack + slot, epoch);
#endif
    __threadfence_system();
    if (slot_state != NULL)
        atomicExch(slot_state + slot, BULK_SLOT_DONE);
    __threadfence_system();
}

#if (L3_FAULT_INJECT_ACK_DELAY == true)
// 有限 ACK 延迟的恢复点。finish_read 已经把槽置为 DONE，但第一次 ACK 被
// 故意省略；manager 下一轮在继续观察 READY 前补发同一 epoch 的 ACK。
// 发送端因此仍需按原有 generation/ACK 合约等待，不能覆盖未确认槽位。
__device__ __forceinline__ bool bulk_inbox_retry_delayed_ack(
    int *slot_state, int *slot_ack)
{
    if (slot_state == NULL || slot_ack == NULL)
        return false;
    int epoch = atomicAdd(&g_l3_fault_ack_pending, 0);
    if (epoch <= 0)
        return false;
    int slot = bulk_inbox_slot(epoch);
    if (atomicAdd(slot_state + slot, 0) != BULK_SLOT_DONE)
        return false;
    __threadfence_system();
    atomicExch(slot_ack + slot, epoch);
    __threadfence_system();
    atomicCAS(&g_l3_fault_ack_pending, epoch, 0);
    return true;
}
#endif

__device__ __forceinline__ int bulk_inbox_ready_epoch(
    int *inbox_epoch, int *inbox_state, int *inbox_generation, int rx_epoch)
{
    // READY/state/generation 由对端异步发布，若整个 warp 各 lane 独立读取，
    // 发布窗口中可能观察到不同快照并在调用方的 early-return/continue 分叉。
    // 只允许当前 active mask 的 leader 取一次协议快照，再广播统一结果。
    unsigned active_mask = __activemask();
    int active_leader = __ffs(active_mask) - 1;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int ready = -1;
    if (lane_id == active_leader
        && inbox_epoch != NULL && inbox_state != NULL
        && inbox_generation != NULL)
    {
        int next = rx_epoch + 1;
        int slot = bulk_inbox_slot(next);
        int observed_ready = atomicAdd(inbox_epoch + slot, 0);
        int state = atomicAdd(inbox_state + slot, 0);
        int generation = atomicAdd(inbox_generation + slot, 0);
        if (observed_ready == next && generation == next
            && state == BULK_SLOT_READY)
            ready = observed_ready;
    }
    ready = __shfl_sync(active_mask, ready, active_leader);
#if (L3_FAULT_INJECT_READY_DELAY == true)
    // 有界地模拟一次 READY 可见性延迟：不修改 inbox 元数据，调用方下轮
    // 仍会按同一个 rx_epoch+1 观察并领取该 generation。ready helper 可能
    // 由整个 warp 或仅 lane0 调用，必须让当前 active mask 的所有 lane
    // 得到一致返回值，否则外层 has_next 分支会发生 warp divergence。
    unsigned delayed = 0;
    if (ready > rx_epoch && lane_id == active_leader)
        delayed = (atomicCAS(&g_l3_fault_ready_delay, 0u, 1u) == 0u);
    delayed = __shfl_sync(active_mask, delayed, active_leader);
    if (delayed != 0u)
        return -1;
#endif
    return ready;
}

__device__ __forceinline__ bool bulk_inbox_has_next(
    int *inbox_epoch, int *inbox_state, int *inbox_generation, int rx_epoch)
{
    return bulk_inbox_ready_epoch(
               inbox_epoch, inbox_state, inbox_generation, rx_epoch) > rx_epoch;
}
#endif
