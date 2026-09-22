#pragma once

// A bounded, local SPSC batch channel between the existing receive manager
// warp and one existing work warp.  The producer advances tail before the
// payload is made READY; the consumer advances head only after its ordinary
// simple_process has returned.  Thus a published-but-not-ready or held slot
// remains visible to termination as tail != head.
struct l3_rx_express_ring
{
    NODE_TYPE *payload;
    int *count;
    unsigned long long *sequence;
    unsigned long long *head;
    unsigned long long *tail;
#if (L3_RX_EXPRESS_DIAG == true)
    unsigned long long *stats;
#endif
};

enum l3_rx_express_stat_index
{
    L3_RX_EXPRESS_ENQUEUE_RECORDS = 0,
    L3_RX_EXPRESS_DEQUEUE_RECORDS = 1,
    L3_RX_EXPRESS_COMPLETE_RECORDS = 2,
    L3_RX_EXPRESS_FALLBACK_RECORDS = 3,
    L3_RX_EXPRESS_ENQUEUE_BATCHES = 4,
    L3_RX_EXPRESS_DEQUEUE_BATCHES = 5,
    L3_RX_EXPRESS_NORMAL_BATCHES = 6,
    L3_RX_EXPRESS_HIGH_WATER = 7,
    L3_RX_EXPRESS_STAT_COUNT = 8
};

#if (L3_RX_EXPRESS == true)

__device__ __forceinline__ void l3_rx_express_add_stat(
    const l3_rx_express_ring &ring, int index, unsigned long long value)
{
#if (L3_RX_EXPRESS_DIAG == true)
    if (ring.stats != NULL && value != 0)
        atomicAdd(ring.stats + index, value);
#else
    (void)ring;
    (void)index;
    (void)value;
#endif
}

__device__ __forceinline__ void l3_rx_express_note_fallback(
    const l3_rx_express_ring &ring, int count, int lane)
{
    if (lane == 0 && count > 0)
        l3_rx_express_add_stat(ring, L3_RX_EXPRESS_FALLBACK_RECORDS,
                               (unsigned long long)count);
}

__device__ __forceinline__ bool l3_rx_express_empty(
    const l3_rx_express_ring &ring)
{
    if (ring.head == NULL || ring.tail == NULL)
        return true;
    return atomicAdd(ring.head, 0ull) == atomicAdd(ring.tail, 0ull);
}

__device__ __forceinline__ bool l3_rx_express_try_enqueue(
    const l3_rx_express_ring &ring, NODE_TYPE *source, int count, int lane)
{
    if (source == NULL || ring.payload == NULL || ring.count == NULL
        || ring.sequence == NULL || ring.head == NULL || ring.tail == NULL
        || count <= 0 || count > L3_RX_EXPRESS_BATCH)
        return false;

    bool reserved = false;
    unsigned long long ticket = 0;
    if (lane == 0)
    {
        const unsigned long long head = atomicAdd(ring.head, 0ull);
        const unsigned long long tail = atomicAdd(ring.tail, 0ull);
        if (tail - head < (unsigned long long)L3_RX_EXPRESS_SLOTS)
        {
            ticket = tail;
            const int slot = (int)(ticket & (L3_RX_EXPRESS_SLOTS - 1));
            if (atomicAdd(ring.sequence + slot, 0ull) == ticket)
            {
                // There is only one producer.  Advancing tail here makes a
                // partially written slot count as live to termination; the
                // sequence word below is the publication gate for the worker.
                atomicExch(ring.tail, ticket + 1ull);
                reserved = true;
            }
        }
    }
    reserved = __shfl_sync(FULL_MASK, reserved, 0);
    ticket = __shfl_sync(FULL_MASK, ticket, 0);
    if (!reserved)
        return false;

    const int slot = (int)(ticket & (L3_RX_EXPRESS_SLOTS - 1));
    if (lane < count)
        ring.payload[slot * L3_RX_EXPRESS_BATCH + lane] = source[lane];
    __syncwarp();
    if (lane == 0)
    {
        ring.count[slot] = count;
        __threadfence();
        atomicExch(ring.sequence + slot, ticket + 1ull);
        const unsigned long long occupancy =
            (ticket + 1ull) - atomicAdd(ring.head, 0ull);
        l3_rx_express_add_stat(ring, L3_RX_EXPRESS_ENQUEUE_RECORDS,
                               (unsigned long long)count);
        l3_rx_express_add_stat(ring, L3_RX_EXPRESS_ENQUEUE_BATCHES, 1ull);
#if (L3_RX_EXPRESS_DIAG == true)
        if (ring.stats != NULL)
            atomicMax(ring.stats + L3_RX_EXPRESS_HIGH_WATER, occupancy);
#endif
    }
    __syncwarp();
    return true;
}

__device__ __forceinline__ bool l3_rx_express_try_dequeue(
    const l3_rx_express_ring &ring, NODE_TYPE *destination, int &count,
    unsigned long long &ticket, int lane)
{
    count = 0;
    ticket = 0;
    if (destination == NULL || ring.payload == NULL || ring.count == NULL
        || ring.sequence == NULL || ring.head == NULL || ring.tail == NULL)
        return false;

    bool ready = false;
    int n = 0;
    if (lane == 0)
    {
        const unsigned long long head = atomicAdd(ring.head, 0ull);
        const unsigned long long tail = atomicAdd(ring.tail, 0ull);
        if (head < tail)
        {
            const int slot = (int)(head & (L3_RX_EXPRESS_SLOTS - 1));
            if (atomicAdd(ring.sequence + slot, 0ull) == head + 1ull)
            {
                ticket = head;
                n = atomicAdd(ring.count + slot, 0);
                if (n < 0) n = 0;
                if (n > L3_RX_EXPRESS_BATCH) n = L3_RX_EXPRESS_BATCH;
                ready = n > 0;
            }
        }
    }
    ready = __shfl_sync(FULL_MASK, ready, 0);
    n = __shfl_sync(FULL_MASK, n, 0);
    ticket = __shfl_sync(FULL_MASK, ticket, 0);
    if (!ready)
        return false;

    const int slot = (int)(ticket & (L3_RX_EXPRESS_SLOTS - 1));
    if (lane < n)
        destination[lane] = ring.payload[slot * L3_RX_EXPRESS_BATCH + lane];
    __syncwarp();
    count = n;
    if (lane == 0)
    {
        l3_rx_express_add_stat(ring, L3_RX_EXPRESS_DEQUEUE_RECORDS,
                               (unsigned long long)n);
        l3_rx_express_add_stat(ring, L3_RX_EXPRESS_DEQUEUE_BATCHES, 1ull);
    }
    __syncwarp();
    return true;
}

__device__ __forceinline__ void l3_rx_express_release(
    const l3_rx_express_ring &ring, unsigned long long ticket, int count,
    int lane)
{
    if (ring.sequence == NULL || ring.head == NULL || ring.tail == NULL)
        return;
    if (lane == 0)
    {
        const int slot = (int)(ticket & (L3_RX_EXPRESS_SLOTS - 1));
        // All queue/L3 writes caused by simple_process precede the ownership
        // release.  A producer may reuse the slot only after this fence.
        __threadfence();
        atomicExch(ring.sequence + slot,
                   ticket + (unsigned long long)L3_RX_EXPRESS_SLOTS);
        atomicExch(ring.head, ticket + 1ull);
        l3_rx_express_add_stat(ring, L3_RX_EXPRESS_COMPLETE_RECORDS,
                               (unsigned long long)(count > 0 ? count : 0));
    }
    __syncwarp();
}

__device__ __forceinline__ void l3_rx_express_note_normal_batch(
    const l3_rx_express_ring &ring, int count, int global_wid, int lane)
{
    if (global_wid == 0 && lane == 0 && count > 0)
        l3_rx_express_add_stat(ring, L3_RX_EXPRESS_NORMAL_BATCHES, 1ull);
}

#endif  // L3_RX_EXPRESS
