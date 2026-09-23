#pragma once
#include "l3_transport.cuh"
#include "l3_priority_bootstrap.cuh"
#include "l3_rx_lag.cuh"
#include "l3_rx_l2_pull.cuh"

// Single manager warp: READING inbox + shared batch form the winner journal.
// ACK follows L2 commit, not merely authoritative distance application.
#if (L3_DIRECT_RX == true)
template <typename Queue>
__device__ __forceinline__ int l3_receive_to_l2(
    NODE_TYPE *inbox, int *counts, int *epochs, int *acks,
    int *states, int *generations, int &rx_epoch,
    VALUE_TYPE *distances, int begin, int size, NODE_TYPE *journal,
    Queue &queue, unsigned *debug_time, int lane
#if (L3_RX_LAG_DIAG == true)
    , VALUE_TYPE *processed
#endif
#if (L3_RX_PRIORITY_BOOTSTRAP == true)
    , bool remote_start
#endif
#if (L3_RX_FEEDBACK_MODE > 0)
    , l3_feedback_record *feedback
#endif
#if (L3_RX_EXPRESS == true)
    , l3_rx_express_ring rx_express
    , bool rx_express_enabled
#endif
#if (L3_RX_L2_PULL == true)
    , unsigned *rx_commit_seq
#if (L3_RX_L2_PULL_DIAG == true)
    , unsigned long long *rx_l2_pull_stats
#endif
#endif
    )
{
    int epoch = bulk_inbox_ready_epoch(epochs, states, generations, rx_epoch);
    if (epoch <= rx_epoch) return 0;
    int slot = bulk_inbox_slot(epoch);
    bool claimed = false;
    if (lane == 0)
        claimed = bulk_inbox_claim_read(states, generations, epochs, slot, epoch);
    claimed = __shfl_sync(FULL_MASK, claimed, 0);
    if (!claimed) return 0;
    const bool payload_visible = bulk_inbox_confirm_read_lane(states + slot);
    if (!__all_sync(FULL_MASK, payload_visible)) return 0;
#if (L3_PROGRESS_DIAG == true)
    const unsigned long long progress_receive_start = clock64();
    if (!lane) {
        if (!g_l3_progress.batches) g_l3_progress.first_receive = progress_receive_start;
        g_l3_progress.last_receive = progress_receive_start;
        ++g_l3_progress.batches;
    }
#endif
    int count = 0;
    if (lane == 0)
        count = l3_atomic_load_relaxed<cuda::thread_scope_system>(counts + slot);
    count = __shfl_sync(FULL_MASK, count, 0);
    assert(count >= 0 && count <= size);
#if (L3_RX_PRIORITY_BOOTSTRAP == true)
    // Only the first incoming batch on a non-source owner, before applying
    // distances or committing any activation. No offline/reference distances.
    if(remote_start && rx_epoch==0 && count>0) {
        int minimum=DIST_MAX;
        for(int i=lane;i<count;i+=WARP_SIZE)
            minimum=min(minimum,int(inbox[slot*(size+1)+i].get_data()));
        for(int offset=16;offset;offset/=2)
            minimum=min(minimum,__shfl_down_sync(FULL_MASK,minimum,offset));
        minimum=__shfl_sync(FULL_MASK,minimum,0);
        int position=l3_bootstrap_empty_priority(queue.q2,minimum,lane);
#if (L3_PROGRESS_DIAG == true)
        if(!lane)printf("L3_PRIORITY_BOOTSTRAP begin=%d minimum=%d position=%d\n",begin,minimum,position);
#endif
    }
#endif
    int improved = 0;
#if (L3_RX_EXPRESS == false)
    int journal_count = 0;
#endif
    for (int offset = 0; offset < count; offset += WARP_SIZE) {
#if (L3_RX_LAG_DIAG == true)
        long long base=0; int delta=1,buckets=1;
        if(!lane)l3_rx_base_snapshot(queue.q2,base,delta,buckets);
        base=__shfl_sync(FULL_MASK,base,0);
        delta=__shfl_sync(FULL_MASK,delta,0);
        buckets=__shfl_sync(FULL_MASK,buckets,0);
        VALUE_TYPE processed_snapshot=DIST_MAX;
#endif
        int id = 0;
        VALUE_TYPE value = DIST_MAX;
        bool won = false;
        if (offset + lane < count) {
            NODE_TYPE item = inbox[slot * (size + 1) + offset + lane];
            id = item.id;
            int local = id - begin;
            assert(local >= 1 && local <= size);
            value = item.get_data();
#if (L3_RX_LAG_DIAG == true)
            processed_snapshot=*((volatile VALUE_TYPE *)&processed[local]);
#endif
#ifdef TYPE_INT
            won = value < atomicMin(distances + local, value);
#else
            won = value < atomicMin_float(distances + local, value);
#endif
        }
        unsigned winners = __ballot_sync(FULL_MASK, won);
#if (L3_ACK_SCAN_FAULT == true)
        int dropped_lane = -1;
        if(!lane && winners && !g_l3_progress.dropped_notification) {
            dropped_lane = __ffs(winners)-1;
            g_l3_progress.dropped_notification = 1;
        }
        dropped_lane = __shfl_sync(FULL_MASK,dropped_lane,0);
        // Preserve the successful authoritative distance update but suppress
        // exactly one queue notification per GPU/query. Recovery must find it.
        if(lane==dropped_lane) won=false;
        winners = __ballot_sync(FULL_MASK,won);
#endif
#if (L3_RX_LAG_DIAG == true)
        unsigned seen=__ballot_sync(FULL_MASK,won && processed_snapshot!=DIST_MAX && value<processed_snapshot);
        unsigned behind=__ballot_sync(FULL_MASK,won && static_cast<long long>(value)<base);
        unsigned ahead=__ballot_sync(FULL_MASK,won && static_cast<long long>(value)>=base+static_cast<long long>(delta)*buckets);
        unsigned long long lag=won && static_cast<long long>(value)<base ? (base-value)/delta : 0;
        for(int step=16;step;step/=2)lag=max(lag,__shfl_down_sync(FULL_MASK,lag,step));
        if(!lane) {
            g_l3_progress.rx_seen_snapshot+=__popc(seen);
            g_l3_progress.rx_below_base+=__popc(behind);
            g_l3_progress.rx_far_ahead+=__popc(ahead);
            g_l3_progress.rx_max_lag_buckets=max(g_l3_progress.rx_max_lag_buckets,lag);
        }
#endif
        int length = __popc(winners);
        int position = __popc(winners & ((1u << lane) - 1u));
#if (L3_CHAIN_PARTITION == true)
        id |= L3_CHAIN_PARTITION_TAG; // local queue only; transport IDs stay plain
#endif
#if (L3_RX_EXPRESS == false)
        if (won)
            journal[journal_count + position] = node_struct(id, value);
#else
        if (won) journal[position] = node_struct(id, value);
#endif
        __syncwarp();
        if (length) {
#if (L3_PROGRESS_DIAG == true)
            if (!lane && !g_l3_progress.winners && improved == 0)
                g_l3_progress.first_winner = clock64();
#endif
#if (L3_RX_EXPRESS == false)
            journal_count += length;
            improved += length;
            if (journal_count >= L3_RX_COMMIT_BATCH)
            {
                int committed = journal_count;
                int pending = committed;
                write_status status = queue.write_through(
                    journal, pending, 0, 0, lane, debug_time);
                // Restricted to DQ, whose write commits the whole batch without waiting.
                assert(status == WRITE_SUCCESS);
                __threadfence();
                __syncwarp();
                journal_count = 0;
#if (L3_PROGRESS_DIAG == true)
                if (!lane) {
                    g_l3_progress.winners += committed;
                    g_l3_progress.last_commit = clock64();
                }
#endif
            }
#else
#if (L3_RX_EXPRESS == true)
            bool express_enqueued = false;
            if (rx_express_enabled)
                express_enqueued = l3_rx_express_try_enqueue(
                    rx_express, journal, length, lane);
            if (!express_enqueued)
            {
#endif
            int pending = length;
            write_status status = queue.write_through(journal, pending, 0, 0, lane, debug_time);
            // Restricted to DQ, whose write commits the whole batch without waiting.
            assert(status == WRITE_SUCCESS);
            __threadfence();
            __syncwarp();
#if (L3_RX_EXPRESS == true)
                l3_rx_express_note_fallback(rx_express, length, lane);
            }
#endif
            improved += length;
#if (L3_PROGRESS_DIAG == true)
            if (!lane) {
                g_l3_progress.winners += length;
                g_l3_progress.last_commit = clock64();
            }
#endif
#endif
        }
    }
#if (L3_RX_EXPRESS == false)
    if (journal_count > 0) {
        int committed = journal_count;
        int pending = committed;
        write_status status = queue.write_through(
            journal, pending, 0, 0, lane, debug_time);
        // Restricted to DQ, whose write commits the whole batch without waiting.
        assert(status == WRITE_SUCCESS);
        __threadfence();
        __syncwarp();
#if (L3_PROGRESS_DIAG == true)
        if (!lane) {
            g_l3_progress.winners += committed;
            g_l3_progress.last_commit = clock64();
        }
#endif
    }
#endif
    __threadfence();
    __syncwarp();
    if (lane == 0) {
#if (L3_RX_FEEDBACK_MODE > 0)
        assert(feedback != nullptr);
        feedback[slot] = {epoch, count, improved};
#if (L3_RX_FEEDBACK_TRACE == true)
        l3_trace_feedback(0, begin, begin, epoch, {epoch, count, improved}, 0, 0, L3_RX_FEEDBACK_MODE);
#endif
        // finish_read's system fence publishes this record before ACK.
#endif
#if (L3_DIAGNOSTICS == true)
        ++g_l3_diagnostics.rx_batches;
        g_l3_diagnostics.received += count;
        g_l3_diagnostics.improved += improved;
#endif
#if (L3_RX_L2_PULL == true)
        // Publish exactly once per inbox after every winner has been submitted
        // to the ordinary L2 queue, but before finish_read publishes the ACK.
        if (improved > 0)
        {
            atomicAdd(rx_commit_seq, 1u);
#if (L3_RX_L2_PULL_DIAG == true)
            l3_rx_l2_pull_add_stat(rx_l2_pull_stats,
                                   L3_RX_L2_PULL_RX_EVENTS);
#endif
        }
#endif
        bulk_inbox_finish_read(states, acks, slot, epoch);
#if (L3_PROGRESS_DIAG == true)
        g_l3_progress.items += count;
        g_l3_progress.receive_cycles += clock64() - progress_receive_start;
#endif
    }
    __syncwarp();
    rx_epoch = epoch;
    return improved;
}
#endif
