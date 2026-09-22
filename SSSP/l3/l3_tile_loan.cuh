#pragma once

// P2 persistent tile loan.  This header is intentionally included after
// relax_dst/simple_process have been defined.  The whole implementation is
// compile-time isolated: the default build has no loan fields or device code.
#if (L3_TILE_LOAN == true)
#if (L3_CONTINUATION == true)
#include <cuda/atomic>
#endif

__device__ __forceinline__ int l3_loan_state_cas(int *p,int old,int value)
{
#if (L3_CONTINUATION == true)
    cuda::atomic_ref<int,cuda::thread_scope_system>(*p).compare_exchange_strong(
        old,value,cuda::memory_order_acq_rel,cuda::memory_order_acquire);
    return old;
#else
    return atomicCAS(p,old,value);
#endif
}
__device__ __forceinline__ void l3_loan_state_store(int *p,int value)
{
#if (L3_CONTINUATION == true)
    cuda::atomic_ref<int,cuda::thread_scope_system>(*p).store(value,cuda::memory_order_release);
#else
    atomicExch(p,value);
#endif
}

__device__ __forceinline__ int l3_tile_loan_load_state(const int *state)
{
#if (L3_CONTINUATION == true)
    return state == NULL ? L3_TILE_LOAN_FREE :
        cuda::atomic_ref<int,cuda::thread_scope_system>(*(int*)state).load(cuda::memory_order_acquire);
#else
    return state == NULL ? L3_TILE_LOAN_FREE
                         : atomicAdd((int *)state, 0);
#endif
}

__device__ __forceinline__ bool l3_tile_loan_any_active(
    const l3_channel_view &channel, int lane_id)
{
    int active = 0;
    if (!lane_id)
    {
        const int out_state = l3_tile_loan_load_state(channel.loan_out.state);
        const int in_state = l3_tile_loan_load_state(channel.loan_in.state);
        active = (out_state != L3_TILE_LOAN_FREE ||
                  in_state != L3_TILE_LOAN_FREE);
    }
    return __shfl_sync(FULL_MASK, active, 0) != 0;
}

// Only the designated global_wid==0 worker can become a helper.  Other work
// warps remain ordinary workers and therefore do not need to observe the
// single-slot control state.
__device__ __forceinline__ bool l3_tile_loan_worker_idle(
    const l3_channel_view &channel, int global_wid, int lane_id,int enabled=1)
{
    if (global_wid != 0)
        return true;
#if (L3_MULTI_PRODUCER == true)
    if(!enabled) return true;
    int drained=0;
    if(!lane_id) drained=l3_gate_slots_drained(channel.loan_gate,channel.peer_loan_gate,
        channel.loan_epoch,channel.loan_out.state,channel.loan_in.state,L3_TILE_LOAN_FREE);
    return __shfl_sync(FULL_MASK,drained,0)!=0;
#else
    int idle = 0;
    if (!lane_id)
    {
        idle = (l3_tile_loan_load_state(channel.loan_out.state) ==
                    L3_TILE_LOAN_FREE &&
                l3_tile_loan_load_state(channel.loan_in.state) ==
                    L3_TILE_LOAN_FREE);
    }
    return __shfl_sync(FULL_MASK, idle, 0) != 0;
#endif
}

// Borrower-side service.  A successful READY->EXECUTING or APPLIED->ACKED
// transition returns true so the caller can skip the ordinary q2 read for
// this loop iteration.
__device__ __forceinline__ bool l3_tile_loan_service_incoming(
    const l3_channel_view &channel, int lane_id, int &active_generation)
{
    if (channel.loan_in.state == NULL ||
        channel.loan_in.generation == NULL ||
        channel.loan_in.seed_count == NULL ||
        channel.loan_in.result_count == NULL ||
        channel.loan_in.requeue_count == NULL ||
        channel.loan_in.ready == NULL)
        return false;

    int state = L3_TILE_LOAN_FREE;
    if (!lane_id)
        state = l3_tile_loan_load_state(channel.loan_in.state);
    state = __shfl_sync(FULL_MASK, state, 0);

    if (state == L3_TILE_LOAN_READY)
    {
        int claimed = 0;
        if (!lane_id)
        {
            claimed = (atomicCAS(channel.loan_in.state,
                                 L3_TILE_LOAN_READY,
                                 L3_TILE_LOAN_EXECUTING) ==
                       L3_TILE_LOAN_READY);
            if (claimed)
                atomicExch(channel.loan_in.ready, 0);
        }
        claimed = __shfl_sync(FULL_MASK, claimed, 0);
        if (claimed)
        {
            int generation = 0;
            int seed_count = 0;
            int result_count = 0;
            if (!lane_id)
            {
                generation = atomicAdd(channel.loan_in.generation, 0);
                seed_count = atomicAdd(channel.loan_in.seed_count, 0);
                result_count = atomicAdd(channel.loan_in.result_count, 0);
            }
            generation = __shfl_sync(FULL_MASK, generation, 0);
            seed_count = __shfl_sync(FULL_MASK, seed_count, 0);
            result_count = __shfl_sync(FULL_MASK, result_count, 0);
            active_generation = generation;

            bool valid = channel.loan_in.tasks != NULL &&
                         channel.loan_in.task_dists != NULL &&
                         channel.loan_in.results != NULL &&
                         channel.loan_in.result_dists != NULL &&
                         channel.loan_in.generation != NULL &&
                         channel.loan_in.requeue_count != NULL &&
                         channel.loan_in.ready != NULL &&
                         channel.peer_row_ptr != NULL &&
                         channel.peer_col_idx != NULL &&
                         channel.peer_edge_data != NULL &&
                         seed_count > 0 &&
                         seed_count <= L3_TILE_LOAN_MAX_SEEDS &&
                         result_count >= 0 &&
                         result_count <= L3_TILE_LOAN_RESULT_CAP;

            // Validate all task rows before writing any result.  If the home
            // side violated the contract, return a fallback marker instead
            // of leaving the persistent protocol stuck in EXECUTING.
            bool lane_valid = valid;
            if (lane_valid)
            {
                for (int i = lane_id; i < seed_count; i += WARP_SIZE)
                {
                    NODE_TYPE task = channel.loan_in.tasks[i];
                    const int local = task.id - 1 - channel.peer_v_begin;
                    if (local < 0 || local >= channel.peer_v_local)
                    {
                        lane_valid = false;
                        break;
                    }
                }
            }
            const unsigned valid_mask = __ballot_sync(FULL_MASK, lane_valid);
            if (valid_mask != FULL_MASK)
            {
                if (!lane_id)
                    atomicExch(channel.loan_in.requeue_count, -1);
                __threadfence_system();
                if (!lane_id)
                    atomicExch(channel.loan_in.state, L3_TILE_LOAN_RETURNED);
#if (L3_TILE_LOAN_DIAG == true)
                if (!lane_id)
                    printf("L3_LOAN_EXECUTE_FALLBACK peer_begin=%d gen=%d seeds=%d results=%d\n",
                           channel.peer_v_begin, generation, seed_count,
                           result_count);
#endif
                return true;
            }

            for (int i = lane_id; i < seed_count; i += WARP_SIZE)
            {
                NODE_TYPE task = channel.loan_in.tasks[i];
                const int local = task.id - 1 - channel.peer_v_begin;
                const int first = channel.peer_row_ptr[local];
                const int last = channel.peer_row_ptr[local + 1];
                int output_base = 0;
                for (int k = 0; k < i; ++k)
                {
                    NODE_TYPE prior = channel.loan_in.tasks[k];
                    const int prior_local = prior.id - 1 - channel.peer_v_begin;
                    output_base += channel.peer_row_ptr[prior_local + 1] -
                                   channel.peer_row_ptr[prior_local];
                }
                const VALUE_TYPE source_dist = channel.loan_in.task_dists[i];
                for (int e = first; e < last; ++e)
                {
                    const int output = output_base + e - first;
                    channel.loan_in.results[output] =
                        node_struct(channel.peer_col_idx[e] + 1,
                                    source_dist + channel.peer_edge_data[e]);
                    channel.loan_in.result_dists[output] =
                        source_dist + channel.peer_edge_data[e];
                }
            }

            __threadfence_system();
            __syncwarp();
            if (!lane_id)
                atomicExch(channel.loan_in.state, L3_TILE_LOAN_RETURNED);
#if (L3_TILE_LOAN_DIAG == true)
            if (!lane_id)
            {
                int unique_dst = 0;
                for (int i = 0; i < result_count; ++i)
                {
                    bool seen = false;
                    for (int j = 0; j < i; ++j)
                    {
                        if (channel.loan_in.results[j].id ==
                            channel.loan_in.results[i].id)
                        {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen)
                        ++unique_dst;
                }
                printf("L3_LOAN_EXECUTE peer_begin=%d gen=%d seeds=%d results=%d unique_dst=%d dup_dst=%d\n",
                       channel.peer_v_begin, generation, seed_count,
                       result_count, unique_dst, result_count - unique_dst);
            }
#endif
            return true;
        }
    }

    if (state == L3_TILE_LOAN_APPLIED)
    {
        int generation = 0;
        if (!lane_id)
            generation = atomicAdd(channel.loan_in.generation, 0);
        generation = __shfl_sync(FULL_MASK, generation, 0);
        if (active_generation == generation && generation != 0)
        {
            if (!lane_id)
                atomicExch(channel.loan_in.state, L3_TILE_LOAN_ACKED);
#if (L3_TILE_LOAN_DIAG == true)
            if (!lane_id)
                printf("L3_LOAN_ACK peer_begin=%d gen=%d\n",
                       channel.peer_v_begin, generation);
#endif
            return true;
        }
    }

    return false;
}

// Home-side result application.  The loan's q2 credit is returned only after
// every result has gone through the same relax_dst semantics as normal work.
template <typename QUEUE_TYPE>
__device__ __forceinline__ bool l3_tile_loan_apply_out(
    QUEUE_TYPE mlmq, const l3_channel_view &channel,
    int block_id, int warp_id, int lane_id, unsigned *debug_time,
    int v_begin, int v_end, VALUE_TYPE *node_data,
    VALUE_TYPE *last_processed, NODE_TYPE *node_in, int &node_in_num,
    int &on_the_fly_num, int &total_work)
{
    if (channel.loan_out.state == NULL ||
        channel.loan_out.generation == NULL ||
        channel.loan_out.seed_count == NULL ||
        channel.loan_out.result_count == NULL ||
        channel.loan_out.requeue_count == NULL ||
        channel.loan_out.results == NULL ||
        channel.loan_out.result_dists == NULL ||
        l3_tile_loan_load_state(channel.loan_out.state) !=
            L3_TILE_LOAN_RETURNED ||
        node_in_num != 0 || on_the_fly_num != 0 ||
        mlmq.get_local_queue_size(warp_id) != 0)
        return false;

    int generation = 0;
    int seed_count = 0;
    int result_count = 0;
    int fallback = 0;
    if (!lane_id)
    {
        generation = atomicAdd(channel.loan_out.generation, 0);
        seed_count = atomicAdd(channel.loan_out.seed_count, 0);
        result_count = atomicAdd(channel.loan_out.result_count, 0);
        fallback = atomicAdd(channel.loan_out.requeue_count, 0);
    }
    generation = __shfl_sync(FULL_MASK, generation, 0);
    seed_count = __shfl_sync(FULL_MASK, seed_count, 0);
    result_count = __shfl_sync(FULL_MASK, result_count, 0);
    fallback = __shfl_sync(FULL_MASK, fallback, 0);

    bool valid = channel.loan_out.results != NULL &&
                 channel.loan_out.result_dists != NULL &&
                 channel.loan_out.tasks != NULL &&
                 channel.loan_out.task_dists != NULL &&
                 channel.loan_out.generation != NULL &&
                 channel.loan_out.seed_count != NULL &&
                 channel.loan_out.result_count != NULL &&
                 seed_count > 0 &&
                 seed_count <= L3_TILE_LOAN_MAX_SEEDS &&
                 result_count >= 0 &&
                 result_count <= L3_TILE_LOAN_RESULT_CAP;
    if (!valid)
        fallback = -1;

    // Apply all returned candidates before examining seed distances.  This
    // ordering is what makes a late improvement visible to the requeue test.
#if (L3_TILE_LOAN_DIAG == true)
    int local_wins = 0;
    int local_enqueued = 0;
    int local_enqueue_fail = 0;
#endif
    if (fallback >= 0)
    {
        __threadfence_system();
        // Process one full warp-sized result group at a time so every lane
        // participates in the ballot even when result_count is not a
        // multiple of WARP_SIZE.  Local winners are ordinary new work and
        // must re-enter q2 before the borrowed seed credit is released.
        for (int base = 0; base < result_count; base += WARP_SIZE)
        {
            const int i = base + lane_id;
            NODE_TYPE result = node_struct(0, 0);
            bool coop_update = false;
            if (i < result_count)
                result = channel.loan_out.results[i];
            if (i < result_count)
            {
#if (WORK_COUNT == true)
                relax_dst(result.id, channel.loan_out.result_dists[i], node_data, v_begin, v_end,
                          channel.candidate_values, channel.candidate_mark,
                          channel.candidate_hint, channel.candidate_hint2,
                          channel.peer_cache, channel.peer_v_begin,
                          coop_update, total_work);
#else
                relax_dst(result.id, channel.loan_out.result_dists[i], node_data, v_begin, v_end,
                          channel.candidate_values, channel.candidate_mark,
                          channel.candidate_hint, channel.candidate_hint2,
                          channel.peer_cache, channel.peer_v_begin,
                          coop_update);
#endif
            }
            const unsigned local_mask = __ballot_sync(FULL_MASK, coop_update);
            const int local_count = count_bit(local_mask);
            const int local_pos = count_bit(
                set_bits(local_mask, 0, lane_id, WARP_SIZE));
            if (coop_update)
            {
                node_in[local_pos] = node_struct(
                    result.id, channel.loan_out.result_dists[i]);
#if (L3_TILE_LOAN_DIAG == true)
                ++local_wins;
#endif
            }
            __syncwarp();
            if (local_count > 0)
            {
                int write_num = local_count;
                const write_status write_status =
                    mlmq.write_through(node_in, write_num, block_id, warp_id,
                                       lane_id, debug_time);
#if (L3_TILE_LOAN_DIAG == true)
                if (!lane_id)
                {
                    if (write_status == WRITE_SUCCESS)
                        local_enqueued += local_count;
                    else
                        local_enqueue_fail += local_count;
                }
#endif
            }
            __syncwarp();
        }
    }
#if (L3_TILE_LOAN_DIAG == true)
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_wins += __shfl_down_sync(FULL_MASK, local_wins, offset);
        local_enqueued += __shfl_down_sync(FULL_MASK, local_enqueued, offset);
        local_enqueue_fail += __shfl_down_sync(FULL_MASK, local_enqueue_fail, offset);
    }
    local_wins = __shfl_sync(FULL_MASK, local_wins, 0);
    local_enqueued = __shfl_sync(FULL_MASK, local_enqueued, 0);
    local_enqueue_fail = __shfl_sync(FULL_MASK, local_enqueue_fail, 0);
#endif
    __syncwarp();

    int requeue_count = 0;
    if (!lane_id)
    {
        for (int i = 0; i < seed_count; ++i)
        {
            NODE_TYPE task = channel.loan_out.tasks[i];
            const int local = task.id - v_begin;
            if (fallback < 0 || local < 1 || local > (v_end - v_begin))
            {
                node_in[requeue_count++] = task;
                continue;
            }

            const VALUE_TYPE snapshot = channel.loan_out.task_dists[i];
            const VALUE_TYPE current =
                *((volatile VALUE_TYPE *)&node_data[local]);
            // The home may race with an ordinary worker that has just
            // recorded a different processed distance.  A loan snapshot is
            // safe only when the authoritative distance has not decreased;
            // requeue every strict decrease and let the normal queue filter
            // discard an already-consumed duplicate.
            if (current < snapshot)
            {
                node_in[requeue_count++] = node_struct(task.id, current);
            }
            else if (current >= snapshot)
            {
#ifdef TYPE_INT
            atomicMin(&last_processed[local], snapshot);
#else
                atomicMin_float(&last_processed[local], snapshot);
#endif
            }
        }
        atomicExch(channel.loan_out.requeue_count, requeue_count);
    }
    requeue_count = __shfl_sync(FULL_MASK, requeue_count, 0);
    if (requeue_count > 0)
    {
        int write_num = requeue_count;
        __syncwarp();
        mlmq.write_through(node_in, write_num, block_id, warp_id,
                           lane_id, debug_time);
        __syncwarp();
    }

    if (!lane_id)
        mlmq.update_done(seed_count);
    __threadfence_system();
    __syncwarp();
    if (!lane_id)
        atomicExch(channel.loan_out.state, L3_TILE_LOAN_APPLIED);
#if (L3_TILE_LOAN_DIAG == true)
    if (!lane_id)
        printf("L3_LOAN_APPLY gen=%d seeds=%d results=%d requeue=%d local_wins=%d local_enqueued=%d local_enqueue_fail=%d fallback=%d\n",
               generation, seed_count, result_count, requeue_count,
               local_wins, local_enqueued, local_enqueue_fail,
               fallback < 0 ? 1 : 0);
#endif
    return true;
}

__device__ __forceinline__ bool l3_tile_loan_reap_acked(
    const l3_channel_view &channel, int lane_id)
{
    if (channel.loan_out.state == NULL)
        return false;
    int reaped = 0;
    if (!lane_id)
        reaped = (l3_loan_state_cas(channel.loan_out.state,
                            L3_TILE_LOAN_ACKED,
                            L3_TILE_LOAN_FREE) == L3_TILE_LOAN_ACKED);
    reaped = __shfl_sync(FULL_MASK, reaped, 0);
#if (L3_TILE_LOAN_DIAG == true)
    if (reaped && !lane_id)
        printf("L3_LOAN_FREE gen=%d\n",
               channel.loan_out.generation == NULL
                   ? 0 : atomicAdd(channel.loan_out.generation, 0));
#endif
    return reaped != 0;
}

// Home-side q2 producer.  node_in doubles as the q2 read scratch buffer, so
// a runtime batch-size mismatch can still fall back without overflowing the
// fixed loan task array.
template <typename QUEUE_TYPE>
__device__ __forceinline__ bool l3_tile_loan_try_publish_impl(
    QUEUE_TYPE mlmq, const l3_channel_view &channel,
    int block_id, int warp_id, int lane_id, unsigned *debug_time,
    int *row_ptr, int v_begin, int v_local, NODE_TYPE *node_in,
    int &node_in_num, int &on_the_fly_num)
{
    if (channel.loan_out.state == NULL ||
        channel.loan_out.generation == NULL ||
        channel.loan_out.seed_count == NULL ||
        channel.loan_out.result_count == NULL ||
        channel.loan_out.requeue_count == NULL ||
        channel.loan_out.tasks == NULL ||
        channel.loan_out.task_dists == NULL ||
        channel.loan_out.results == NULL ||
        channel.loan_out.result_dists == NULL ||
        channel.loan_out.ready == NULL || node_in_num != 0 ||
        on_the_fly_num != 0 || mlmq.get_local_queue_size(warp_id) != 0 ||
#if (L3_CONTINUATION == true)
        // Reader reservations are not completed reads. A loose responsibility
        // gate permits this reader to try its own reservation; q2.read alone
        // decides whether any source credit is actually transferred.
        mlmq.get_global_queue_size() < L3_TILE_LOAN_MIN_QUEUE)
#else
        mlmq.get_available_queue_size() < L3_TILE_LOAN_MIN_QUEUE)
#endif
        return false;

    int claimed = 0;
    if (!lane_id)
    {
        if (l3_tile_loan_load_state(channel.loan_out.state) ==
                L3_TILE_LOAN_FREE &&
            atomicAdd(channel.loan_out.ready, 0) == 1)
        {
            claimed = (l3_loan_state_cas(channel.loan_out.state,
                                 L3_TILE_LOAN_FREE,
                                 L3_TILE_LOAN_CLAIMED) ==
                       L3_TILE_LOAN_FREE);
        }
    }
    claimed = __shfl_sync(FULL_MASK, claimed, 0);
    if (!claimed)
        return false;
#if (L3_TILE_LOAN_DIAG == true)
    if (!lane_id)
        printf("L3_LOAN_CLAIM v_begin=%d q=%d available=%d ready=%d\n",
               v_begin, mlmq.get_global_queue_size(),
               mlmq.get_available_queue_size(),
               atomicAdd(channel.loan_out.ready, 0));
#endif

    // One q2 read is normally only l2_batch_size (8) items.  Aggregate
    // consecutive q2 batches while this slot is claimed so the cross-GPU
    // state machine amortizes its P2P fences and ACK over the existing
    // node_in capacity rather than serializing one round trip per 8 seeds.
    int count = 0;
    int empty_retries = 0;
    while (count < L3_TILE_LOAN_MAX_SEEDS)
    {
        // q2.read has no destination-capacity argument and may write one
        // complete l2_batch_size chunk. Leave a full chunk of scratch space;
        // a partial tail is handled by the normal local fallback.
        if (L3_TILE_LOAN_MAX_SEEDS - count < l2_batch_size)
            break;
        int chunk = 0;
        mlmq.q2.read(node_in + count, chunk, block_id, warp_id,
                     lane_id, debug_time);
        __syncwarp();
        chunk = __shfl_sync(FULL_MASK, chunk, 0);
        if (chunk <= 0)
        {
            if (empty_retries++ >= L3_TILE_LOAN_READ_RETRIES)
                break;
            // A direct q2 read can race the manager's read_pos publication.
            // Keep the same reservation and give the manager a bounded
            // scheduling window; no queue record is consumed on an empty read.
            __threadfence();
            continue;
        }
        empty_retries = 0;
        count += chunk;
    }
    if (count <= 0)
    {
#if (L3_TILE_LOAN_DIAG == true)
        if (!lane_id)
            printf("L3_LOAN_READ_EMPTY v_begin=%d\n", v_begin);
#endif
        if (!lane_id)
            l3_loan_state_store(channel.loan_out.state, L3_TILE_LOAN_FREE);
        // No source was reserved.  Let the caller perform its ordinary q2
        // read in this same iteration; returning true here needlessly
        // suppresses normal queue progress when the current delta bucket is
        // published between manager polls.
        return false;
    }

    int edge_sum = 0;
    bool lane_valid = count <= node_size;
    for (int i = lane_id; i < count; i += WARP_SIZE)
    {
        NODE_TYPE task = node_in[i];
        const int local = task.id - 1 - v_begin;
        if (local < 0 || local >= v_local)
        {
            lane_valid = false;
            continue;
        }
        const int degree = row_ptr[local + 1] - row_ptr[local];
        edge_sum = min(L3_TILE_LOAN_RESULT_CAP + 1,
                       edge_sum + max(0, degree));
    }
    const unsigned valid_mask = __ballot_sync(FULL_MASK, lane_valid);
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        const int other = __shfl_down_sync(FULL_MASK, edge_sum, offset);
        edge_sum = min(L3_TILE_LOAN_RESULT_CAP + 1, edge_sum + other);
    }
    edge_sum = __shfl_sync(FULL_MASK, edge_sum, 0);

    if (count > L3_TILE_LOAN_MAX_SEEDS ||
        valid_mask != FULL_MASK || edge_sum > L3_TILE_LOAN_RESULT_CAP)
    {
        // The q2 records are still in node_in and keep their original
        // on_the_fly responsibility; ordinary simple_process will replay
        // them and update_done will account for exactly this read.
        node_in_num = count;
        on_the_fly_num = count;
        if (!lane_id)
            l3_loan_state_store(channel.loan_out.state, L3_TILE_LOAN_FREE);
#if (L3_TILE_LOAN_DIAG == true)
        if (!lane_id)
            printf("L3_LOAN_FALLBACK v_begin=%d seeds=%d edges=%d\n",
                   v_begin, count, edge_sum);
#endif
        return true;
    }

    for (int i = lane_id; i < count; i += WARP_SIZE)
    {
        NODE_TYPE task = node_in[i];
        channel.loan_out.tasks[i] = task;
        channel.loan_out.task_dists[i] = task.get_data();
    }
    // Every lane wrote part of the peer-visible payload. A block-local
    // barrier alone does not publish those writes to the other GPU.
    __threadfence_system();
    __syncwarp();
    if (!lane_id)
    {
        const int generation =
            atomicAdd(channel.loan_out.generation, 1) + 1;
        atomicExch(channel.loan_out.seed_count, count);
        atomicExch(channel.loan_out.result_count, edge_sum);
        atomicExch(channel.loan_out.requeue_count, 0);
        __threadfence_system();
        l3_loan_state_store(channel.loan_out.state, L3_TILE_LOAN_READY);
#if (L3_TILE_LOAN_DIAG == true)
        printf("L3_LOAN_PUBLISH v_begin=%d gen=%d seeds=%d edges=%d\n",
               v_begin, generation, count, edge_sum);
#endif
    }
    return true;
}

template<typename QUEUE_TYPE>
__device__ __forceinline__ bool l3_tile_loan_try_publish(
    QUEUE_TYPE mlmq,const l3_channel_view &channel,int bid,int wid,int lane,
    unsigned *debug,int *rows,int begin,int local,NODE_TYPE *input,int &count,int &fly)
{
#if (L3_MULTI_PRODUCER == true)
    if(mlmq.get_local_queue_size(wid)!=0) return false;
#if (L3_MULTI_PRODUCER_PREFILTER == true)
    // Advisory only: do not register when the original impl would reject.
    // A successful precheck grants no ownership; enter and slot CAS below
    // remain mandatory and impl rechecks its original admission conditions.
    int eligible=0;
    if(!lane) eligible=mlmq.get_global_queue_size()>=L3_TILE_LOAN_MIN_QUEUE &&
        l3_tile_loan_load_state(channel.loan_out.state)==L3_TILE_LOAN_FREE;
    eligible=__shfl_sync(FULL_MASK,eligible,0);
    if(!eligible) return false;
#endif
    int entered=0;
    if(!lane) entered=l3_gate_enter(channel.loan_gate,channel.loan_epoch);
    entered=__shfl_sync(FULL_MASK,entered,0);
    if(!entered) return false;
    if(!lane) ++channel.loan_producers[bid];
#endif
    bool result=l3_tile_loan_try_publish_impl(mlmq,channel,bid,wid,lane,debug,rows,
                                            begin,local,input,count,fly);
#if (L3_MULTI_PRODUCER == true)
    __syncwarp();
    if(!lane) {
        if(result && count==0) ++channel.loan_producers[4096+bid];
        if(!l3_gate_leave(channel.loan_gate,channel.loan_epoch)) asm("trap;");
    }
    __syncwarp();
#endif
    return result;
}

__device__ __forceinline__ void l3_tile_loan_update_ready(
    const l3_channel_view &channel, int global_wid, int lane_id,
    int node_in_num, int on_the_fly_num, int local_queue_empty,
    int global_queue_empty)
{
    if (global_wid != 0 || channel.loan_in.ready == NULL)
        return;
    int ready = 0;
    if (!lane_id)
    {
        ready = (node_in_num == 0 && on_the_fly_num == 0 &&
                 local_queue_empty && global_queue_empty &&
                 l3_tile_loan_load_state(channel.loan_in.state) ==
                     L3_TILE_LOAN_FREE);
        atomicExch(channel.loan_in.ready, ready);
    }
    __syncwarp();
}

#endif  // L3_TILE_LOAN
