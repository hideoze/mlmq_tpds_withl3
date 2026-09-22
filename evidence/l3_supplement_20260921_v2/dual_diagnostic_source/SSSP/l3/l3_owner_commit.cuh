#pragma once

// G3 direct owner-commit experiment.
//
// The existing tile-loan slot owns the source credit and the generation
// handshake.  This header changes only the borrower-side execution and the
// lender-side reclaim:
//
//   donor  -- lease/source CSR --> helper
//   helper -- local q2 write or P2P atomicMin+dirty --> destination owner
//   helper -- generation token --> donor
//
// No per-edge result array is consumed by this path.  The experiment is
// compile-time isolated and L3_OWNER_COMMIT is default-off.
#if (L3_OWNER_COMMIT == true)

// Commit one destination directly to the owner GPU.  dst_v is a global
// 1-based vertex id; the peer arrays use a 1-based local slot with slot 0
// deliberately unused, matching relax_dst and the existing direct-P2P path.
__device__ __forceinline__ bool l3_owner_commit_remote_relax(
    const l3_channel_view &channel, int dst_v, VALUE_TYPE new_dist)
{
    if (channel.peer_node_data == NULL ||
        channel.peer_dirty_bitmap == NULL ||
        channel.peer_dirty_hint == NULL)
        return false;

    const int peer_lidx = dst_v - channel.peer_v_begin;
    if (peer_lidx < 1 || peer_lidx > channel.peer_v_local)
        return false;

    const VALUE_TYPE old_dist =
        atomicMin(&channel.peer_node_data[peer_lidx], new_dist);

    // The cache is only an advisory lower-bound filter for the ordinary
    // candidate path.  Recording a value that was actually committed cannot
    // make a later candidate unsafe, and prevents a duplicate candidate when
    // the same helper subsequently expands another source.
    if (channel.peer_cache != NULL)
        atomicMin(&channel.peer_cache[peer_lidx], new_dist);

    if (new_dist >= old_dist)
        return false;

    const int peer_zero = peer_lidx - 1;
    atomicOr(&channel.peer_dirty_bitmap[peer_zero >> 5],
             1u << (peer_zero & 31));
    atomicOr(&channel.peer_dirty_hint[peer_zero >> 10],
             1u << ((peer_zero >> 5) & 31));
    return true;
}

// Borrower-side direct owner commit.  One lane owns one leased source, and
// all lanes advance through the same edge offset so local winners can be
// compacted into node_in and written to the helper's own q2 cooperatively.
template <typename QUEUE_TYPE>
__device__ __forceinline__ bool l3_owner_commit_service_incoming(
    QUEUE_TYPE mlmq, const l3_channel_view &channel,
    int block_id, int warp_id, int lane_id, unsigned *debug_time,
    int v_begin, int v_end, VALUE_TYPE *node_data,
    NODE_TYPE *node_in, int &node_in_num, int &on_the_fly_num,
    int &total_work, int &active_generation)
{
    (void)on_the_fly_num;

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
        if (!claimed)
            return false;

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
                     channel.loan_in.generation != NULL &&
                     channel.peer_node_data != NULL &&
                     channel.peer_dirty_bitmap != NULL &&
                     channel.peer_dirty_hint != NULL &&
                     channel.peer_row_ptr != NULL &&
                     channel.peer_col_idx != NULL &&
                     channel.peer_edge_data != NULL &&
                     seed_count > 0 &&
                     seed_count <= L3_TILE_LOAN_MAX_SEEDS &&
                     result_count >= 0 &&
                     result_count <= L3_TILE_LOAN_RESULT_CAP;

        bool lane_valid = valid;
        if (lane_valid)
        {
            for (int i = lane_id; i < seed_count; i += WARP_SIZE)
            {
                const NODE_TYPE task = channel.loan_in.tasks[i];
                const int local0 = task.id - 1 - channel.peer_v_begin;
                if (local0 < 0 || local0 >= channel.peer_v_local)
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
            __syncwarp();
            if (!lane_id)
                atomicExch(channel.loan_in.state, L3_TILE_LOAN_RETURNED);
#if (L3_TILE_LOAN_DIAG == true)
            if (!lane_id)
                printf("L3_OWNER_COMMIT_FALLBACK peer_begin=%d gen=%d seeds=%d edges=%d reason=contract\n",
                       channel.peer_v_begin, generation, seed_count,
                       result_count);
#endif
            return true;
        }

        int local0 = -1;
        int first = 0;
        int degree = 0;
        VALUE_TYPE source_dist = DIST_MAX;
        if (lane_id < seed_count)
        {
            const NODE_TYPE task = channel.loan_in.tasks[lane_id];
            local0 = task.id - 1 - channel.peer_v_begin;
            first = channel.peer_row_ptr[local0];
            degree = channel.peer_row_ptr[local0 + 1] - first;

            // The ordinary owner path expands from the authoritative distance,
            // not necessarily the queue record's snapshot.  Reading the peer
            // distance here preserves that property for a leased source.
            source_dist = *((volatile VALUE_TYPE *)&
                                channel.peer_node_data[local0 + 1]);
        }

        int max_degree = degree;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_degree = max(max_degree,
                             __shfl_down_sync(FULL_MASK, max_degree, offset));
        max_degree = __shfl_sync(FULL_MASK, max_degree, 0);

        bool protocol_ok = true;
#if (L3_TILE_LOAN_DIAG == true)
        unsigned long long edge_count = 0;
        unsigned long long remote_attempts = 0;
        unsigned long long remote_wins = 0;
        unsigned long long local_wins = 0;
#endif

        // Every lane owns one source.  The lockstep edge offset bounds the
        // temporary local-winner list to at most one entry per lane, so the
        // existing node_in[32] warp scratch is never overrun.
        for (int edge_offset = 0; edge_offset < max_degree; ++edge_offset)
        {
            bool local_update = false;
            NODE_TYPE local_record = node_struct(0, 0);
            int dst_v = 0;
            VALUE_TYPE new_dist = 0;

            if (lane_id < seed_count && edge_offset < degree &&
                source_dist != DIST_MAX)
            {
                const int edge = first + edge_offset;
                dst_v = channel.peer_col_idx[edge] + 1;
                new_dist = source_dist + channel.peer_edge_data[edge];
#if (L3_TILE_LOAN_DIAG == true)
                ++edge_count;
#endif

                const bool dst_is_local =
                    dst_v - 1 >= v_begin && dst_v - 1 < v_end;
                if (dst_is_local)
                {
                    const int local = dst_v - v_begin;
                    const VALUE_TYPE old_dist =
                        *((volatile VALUE_TYPE *)&node_data[local]);
                    if (new_dist < old_dist)
                    {
                        const VALUE_TYPE update_res =
                            atomicMin(&node_data[local], new_dist);
                        if (new_dist < update_res)
                        {
                            local_update = true;
                            local_record = node_struct(dst_v, new_dist);
#if (WORK_COUNT == true)
                            ++total_work;
                            atomicAdd(&g_local_work, (work_count_type)1);
                            atomicAdd(&g_hist[local], 1u);
#endif
#if (L3_TILE_LOAN_DIAG == true)
                            ++local_wins;
#endif
                        }
                    }
                }
                else if (dst_v - 1 >= channel.peer_v_begin &&
                         dst_v - 1 < channel.peer_v_begin +
                                          channel.peer_v_local)
                {
#if (L3_TILE_LOAN_DIAG == true)
                    ++remote_attempts;
#endif
                    const bool won = l3_owner_commit_remote_relax(
                        channel, dst_v, new_dist);
#if (L3_TILE_LOAN_DIAG == true)
                    if (won)
                        ++remote_wins;
#endif
                }
                else
                {
                    // A one-peer channel cannot safely commit a destination
                    // belonging to a third partition.  Reclaiming the source
                    // through the ordinary path is the conservative fallback.
                    protocol_ok = false;
                }
            }

            const unsigned update_mask = __ballot_sync(FULL_MASK, local_update);
            const int update_num = count_bit(update_mask);
            const int update_pos = count_bit(
                set_bits(update_mask, 0, lane_id, WARP_SIZE));
            if (local_update)
                node_in[update_pos] = local_record;
            __syncwarp();

            write_status write_result = WRITE_SUCCESS;
            if (update_num > 0)
            {
                int write_num = update_num;
                write_result = mlmq.write_through(
                    node_in, write_num, block_id, warp_id, lane_id,
                    debug_time);
            }
            const unsigned write_mask = __ballot_sync(
                FULL_MASK, write_result == WRITE_SUCCESS);
            if (write_mask != FULL_MASK)
                protocol_ok = false;
            __syncwarp();
        }

        const unsigned protocol_mask =
            __ballot_sync(FULL_MASK, protocol_ok);
        const bool commit_ok = protocol_mask == FULL_MASK;
        if (!lane_id)
            atomicExch(channel.loan_in.requeue_count,
                       commit_ok ? 0 : -1);
        __threadfence_system();
        __syncwarp();
        if (!lane_id)
            atomicExch(channel.loan_in.state, L3_TILE_LOAN_RETURNED);
#if (L3_TILE_LOAN_DIAG == true)
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            edge_count += __shfl_down_sync(FULL_MASK, edge_count, offset);
            remote_attempts +=
                __shfl_down_sync(FULL_MASK, remote_attempts, offset);
            remote_wins += __shfl_down_sync(FULL_MASK, remote_wins, offset);
            local_wins += __shfl_down_sync(FULL_MASK, local_wins, offset);
        }
        if (!lane_id)
            printf("L3_OWNER_COMMIT_EXECUTE peer_begin=%d gen=%d seeds=%d edges=%d remote_attempts=%llu remote_wins=%llu local_wins=%llu fallback=%d\n",
                   channel.peer_v_begin, generation, seed_count,
                   (int)edge_count, remote_attempts, remote_wins,
                   local_wins, commit_ok ? 0 : 1);
#endif
        return true;
    }

    // The helper ACKs only the generation it actually executed.  A stale or
    // unsolicited APPLIED state remains visible to the owner instead of being
    // recycled by a different query generation.
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
                printf("L3_OWNER_COMMIT_ACK peer_begin=%d gen=%d\n",
                       channel.peer_v_begin, generation);
#endif
            return true;
        }
    }

    return false;
}

// Lender-side reclaim for direct owner commit.  There are no returned edge
// records to apply: the helper has already committed each destination.  The
// donor only requeues a source whose authoritative distance decreased while
// the source lease was outstanding, then releases the original q2 credit.
template <typename QUEUE_TYPE>
__device__ __forceinline__ bool l3_owner_commit_reap_out(
    QUEUE_TYPE mlmq, const l3_channel_view &channel,
    int block_id, int warp_id, int lane_id, unsigned *debug_time,
    int v_begin, int v_end, VALUE_TYPE *node_data,
    VALUE_TYPE *last_processed, NODE_TYPE *node_in, int &node_in_num,
    int &on_the_fly_num, int &total_work)
{
    (void)on_the_fly_num;
    (void)total_work;

    if (channel.loan_out.state == NULL ||
        channel.loan_out.generation == NULL ||
        channel.loan_out.seed_count == NULL ||
        channel.loan_out.requeue_count == NULL ||
        channel.loan_out.tasks == NULL ||
        channel.loan_out.task_dists == NULL ||
        l3_tile_loan_load_state(channel.loan_out.state) !=
            L3_TILE_LOAN_RETURNED ||
        node_in_num != 0 || on_the_fly_num != 0 ||
        mlmq.get_local_queue_size(warp_id) != 0)
        return false;

    int generation = 0;
    int seed_count = 0;
    int fallback = 0;
    if (!lane_id)
    {
        generation = atomicAdd(channel.loan_out.generation, 0);
        seed_count = atomicAdd(channel.loan_out.seed_count, 0);
        fallback = atomicAdd(channel.loan_out.requeue_count, 0);
    }
    generation = __shfl_sync(FULL_MASK, generation, 0);
    seed_count = __shfl_sync(FULL_MASK, seed_count, 0);
    fallback = __shfl_sync(FULL_MASK, fallback, 0);

    const bool valid = channel.loan_out.tasks != NULL &&
                       channel.loan_out.task_dists != NULL &&
                       channel.loan_out.generation != NULL &&
                       channel.loan_out.seed_count != NULL &&
                       seed_count > 0 &&
                       seed_count <= L3_TILE_LOAN_MAX_SEEDS;
    if (!valid)
        fallback = -1;

    int requeue_count = 0;
    if (!lane_id)
    {
        for (int i = 0; i < seed_count; ++i)
        {
            const NODE_TYPE task = channel.loan_out.tasks[i];
            const int local = task.id - v_begin;
            const bool task_valid =
                local >= 1 && local <= (v_end - v_begin) &&
                node_data != NULL;

            if (!task_valid)
            {
                // This is a producer/metadata violation.  Keep the original
                // record for the normal path; no unsafe node_data index.
                node_in[requeue_count++] = task;
                continue;
            }

            const VALUE_TYPE snapshot = channel.loan_out.task_dists[i];
            const VALUE_TYPE current =
                *((volatile VALUE_TYPE *)&node_data[local]);
            if (fallback < 0 || current < snapshot)
            {
                // A direct commit may have improved the source itself, or the
                // ordinary donor worker may have raced the helper.  Requeue
                // the new authoritative value and let simple_process filter
                // any duplicate against last_processed.
                node_in[requeue_count++] = node_struct(task.id, current);
            }
            else if (last_processed != NULL)
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

    write_status write_result = WRITE_SUCCESS;
    if (requeue_count > 0)
    {
        int write_num = requeue_count;
        __syncwarp();
        write_result = mlmq.write_through(
            node_in, write_num, block_id, warp_id, lane_id, debug_time);
    }
    const unsigned write_mask = __ballot_sync(
        FULL_MASK, write_result == WRITE_SUCCESS);
    if (write_mask != FULL_MASK)
    {
#if (L3_TILE_LOAN_DIAG == true)
        if (!lane_id)
            printf("L3_OWNER_COMMIT_APPLY_RETRY gen=%d seeds=%d requeue=%d\n",
                   generation, seed_count, requeue_count);
#endif
        // Keep RETURNED and the q2 credit outstanding.  The next safe poll
        // retries the same lease; this is preferable to releasing work that
        // was not accepted by the queue.
        return true;
    }

    if (!lane_id)
        mlmq.update_done(seed_count);
    __threadfence_system();
    __syncwarp();
    if (!lane_id)
        atomicExch(channel.loan_out.state, L3_TILE_LOAN_APPLIED);
#if (L3_TILE_LOAN_DIAG == true)
    if (!lane_id)
        printf("L3_OWNER_COMMIT_APPLY gen=%d seeds=%d requeue=%d fallback=%d\n",
               generation, seed_count, requeue_count, fallback < 0 ? 1 : 0);
#endif
    return true;
}

#endif  // L3_OWNER_COMMIT
