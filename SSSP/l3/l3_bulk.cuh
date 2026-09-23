#pragma once

#include "l3_metrics.cuh"
#include "l3_transport.cuh"
#include "l3_sync.cuh"

// L3-06: 通用 BULK pack/apply/batch orchestration。槽位 generation/ACK 原语
// 由 l3_transport.cuh 提供；这里集中候选批次的消费、打包、发布和回挂。
#if (BULK_ROUND == true)

#if (BULK_DIAG == true)
extern __device__ unsigned long long g_bulk_frontier_diag_prints;
#endif
#if (BULK_TRACE == true)
extern __device__ unsigned long long g_bulk_trace_tx;
extern __device__ unsigned long long g_bulk_trace_rx;
#endif

// BULK_ROUND 接收侧：对端只发布已经完成的 epoch 列表，本卡在本地做 atomicMin，
// 只把真正改进的顶点注入自己的 L2。inbox 在 ack 前不会被发送卡复用，因此读取期间
// 不会被覆盖；epoch/count 的发布顺序由发送侧的 system fence 保证。
__device__ __forceinline__ int bulk_consume_inbox(
    NODE_TYPE *inbox, int *inbox_count, int *inbox_epoch, int *inbox_ack,
    int *inbox_state, int *inbox_generation,
    int &rx_epoch, VALUE_TYPE *node_data, int v_begin, int v_local,
    unsigned *dirty_bitmap, unsigned *dirty_hint, int lane_id
#if (BULK_FRONTIER_ENABLED == true)
    , NODE_TYPE *bulk_frontier, int *bulk_frontier_head, int *bulk_frontier_tail,
    int *frontier_append_base, int *frontier_append_count
#endif
    )
{
    if (inbox == NULL || inbox_count == NULL || inbox_epoch == NULL || inbox_ack == NULL
        || inbox_state == NULL || inbox_generation == NULL)
        return 0;

    int ready = bulk_inbox_ready_epoch(
        inbox_epoch, inbox_state, inbox_generation, rx_epoch);
    if (ready <= rx_epoch)
        return 0;

#if (BULK_FRONTIER_ENABLED == true)
    if (bulk_frontier == NULL || bulk_frontier_tail == NULL
        || frontier_append_base == NULL || frontier_append_count == NULL)
        return 0;
#endif

    int slot = bulk_inbox_slot(ready);
    bool claimed = false;
    if (!lane_id)
        claimed = bulk_inbox_claim_read(inbox_state, inbox_generation, inbox_epoch,
                                        slot, ready);
    claimed = __shfl_sync(FULL_MASK, claimed, 0);
    if (!claimed)
        return 0;
    const bool payload_visible = bulk_inbox_confirm_read_lane(inbox_state + slot);
    if (!__all_sync(FULL_MASK, payload_visible))
        return 0;

    NODE_TYPE *slot_inbox = inbox + slot * (v_local + 1);
    int *slot_count = inbox_count + slot;
    int cnt = l3_atomic_load_relaxed<cuda::thread_scope_system>(slot_count);
    if (cnt < 0) cnt = 0;
    if (cnt > v_local) cnt = v_local;
#if (L3_EVENT_RING == true)
    if (!lane_id)
        l3_event_push(L3_EVENT_RX_CLAIM, ready, slot, cnt, rx_epoch);
#endif
#if (BULK_FRONTIER_ENABLED == true)
    if (!lane_id)
    {
        *frontier_append_base = atomicAdd(bulk_frontier_tail, 0);
        *frontier_append_count = 0;
    }
    __syncwarp();
#endif
#if (BULK_DIAG == true)
    if (!lane_id)
        printf("BULK_RX vbeg=%d ready=%d old=%d cnt=%d\\n", v_begin, ready, rx_epoch, cnt);
#endif

    int improved_total = 0;
    for (int s0 = 0; s0 < cnt; s0 += BULK_INBOX_BATCH)
    {
        int n = mlq_min(BULK_INBOX_BATCH, cnt - s0);
        int lane_improved = 0;
        // 一个 warp 只有 32 个 lane，而 BULK_INBOX_BATCH 默认是 128。
        // 必须用 stride 消费整个批次；此前只读取 s0+lane_id，随后却
        // ack 整个 epoch，导致每批 128 条消息中只有前 32 条生效。
        for (int t = lane_id; t < n; t += WARP_SIZE)
        {
            NODE_TYPE item = slot_inbox[s0 + t];
            int local_id = item.id - v_begin;
            if (local_id >= 1 && local_id <= v_local)
            {
                VALUE_TYPE nd = item.get_data();
                VALUE_TYPE old_dist;
#ifdef TYPE_INT
                old_dist = atomicMin(&node_data[local_id], nd);
#else
                old_dist = atomicMin_float(&node_data[local_id], nd);
#endif
                bool improved_item = (nd < old_dist);
                if (improved_item)
                {
                    int j = local_id - 1;
#if (BULK_FRONTIER_ENABLED == true)
                    int slot = atomicAdd(frontier_append_count, 1);
                    int base = *frontier_append_base;
                    if (base >= 0 && base + slot <= v_local)
                    {
                        bulk_frontier[base + slot] = node_struct(item.id, nd);
                        __threadfence();
                        lane_improved++;
                    }
#else
                    l3_system_mark_publish(&dirty_bitmap[j >> 5], 1u << (j & 31));
                    l3_system_mark_publish(&dirty_hint[j >> 10],
                                           1u << ((j >> 5) & 31));
                    lane_improved++;
#endif
                }
#if (BULK_TRACE == true)
                if (item.id == BULK_TRACE_NODE)
                {
                    unsigned long long t = atomicAdd(&g_bulk_trace_rx, 1ull);
                    if (t < 32)
                        printf("BULK_TRACE_RX g=%d epoch=%d id=%d nd=%.0f old=%.0f imp=%d\\n",
                               v_begin, ready, item.id, (double)nd, (double)old_dist,
                               (int)improved_item);
                }
#endif
            }
        }

        int out_num = lane_improved;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
            out_num += __shfl_down_sync(FULL_MASK, out_num, offset);
        out_num = __shfl_sync(FULL_MASK, out_num, 0);
#if (BULK_DIAG == true)
        if (!lane_id)
            printf("BULK_RX_BATCH vbeg=%d epoch=%d s=%d n=%d improved=%d\\n",
                   v_begin, ready, s0, n, out_num);
#endif
        __syncwarp();
        improved_total += out_num;
    }

#if (BULK_FRONTIER_ENABLED == true)
    __threadfence();
    __syncwarp();
    if (!lane_id)
    {
        int base = *frontier_append_base;
        int appended = *frontier_append_count;
        int capacity = (base >= 0 && base <= v_local) ? (v_local - base + 1) : 0;
        if (appended > capacity) appended = capacity;
        atomicExch(bulk_frontier_tail, base + appended);
#if (BULK_DIAG == true)
        unsigned long long p = atomicAdd(&g_bulk_frontier_diag_prints, 1ull);
        if (p < 128)
            printf("BULK_FRONTIER_APPEND g%d epoch=%d base=%d count=%d head=%d tail=%d improved=%d\n",
                   v_begin, ready, base, appended,
                   atomicAdd(bulk_frontier_head, 0),
                   atomicAdd(bulk_frontier_tail, 0), improved_total);
#endif
    }
    __threadfence();
    __syncwarp();
#endif
    __syncwarp();
    if (!lane_id)
        bulk_inbox_finish_read(inbox_state, inbox_ack, slot, ready);
    __syncwarp();
#if (L3_EVENT_RING == true)
    if (!lane_id)
        l3_event_push(L3_EVENT_RX_APPLY, ready, slot, cnt, improved_total);
#endif
    rx_epoch = ready;
    return improved_total;
}


// BULK_ROUND 发送侧：work warp 全部确认静止后，remote_mark/remote_cand 不再被生产者
// 修改。本函数一次性消费整个 mark 位图，构造每个远程顶点至多一条的紧凑列表，再按
// data -> count -> epoch 顺序发布到对端 inbox。与旧 L3 路径不同，这里没有逐候选 P2P
// node_data/dirty 写，也没有中间值的跨卡可见窗口。
__device__ __forceinline__ bool bulk_pack_publish_warp(
    int epoch, int lane_id, int peer_v_begin, int peer_v_local,
    VALUE_TYPE *remote_cand, unsigned *remote_mark, unsigned *mark_hint, unsigned *mark_hint2,
    VALUE_TYPE *peer_cache, VALUE_TYPE *peer_node_data,
    unsigned *peer_dirty_bitmap, unsigned *peer_dirty_hint,
    NODE_TYPE *send_list, NODE_TYPE *peer_inbox, int *peer_inbox_count,
    int *peer_inbox_epoch, int *peer_inbox_ack,
    int *peer_inbox_state, int *peer_inbox_generation, int *send_count,
    volatile int *publish_done)
{
    if (send_list == NULL || peer_inbox == NULL
        || peer_inbox_count == NULL
        || peer_inbox_epoch == NULL || peer_inbox_ack == NULL
        || peer_inbox_state == NULL || peer_inbox_generation == NULL
        || send_count == NULL || publish_done == NULL)
    {
#if (BULK_DIAG == true)
        if (!lane_id)
            printf("BULK_PUB_SKIP epoch=%d send=%p inbox=%p cnt=%p ep=%p ack=%p\\n",
                   epoch, (void *)send_list, (void *)peer_inbox,
                   (void *)peer_inbox_count, (void *)peer_inbox_epoch,
                   (void *)peer_inbox_ack);
#endif
        return false;
    }
#if (GLOBAL_ROUND_DIRECT_P2P == true)
    if (peer_node_data == NULL || peer_dirty_bitmap == NULL || peer_dirty_hint == NULL)
        return false;
#endif

    // 先取得接收卡对应槽位的所有权，再消费本卡 remote_mark。这样即使接收卡
    // 仍在读取旧 generation，本轮候选也不会被提前清空。
    int slot = bulk_inbox_slot(epoch);
    bool claimed = false;
    if (!lane_id)
        claimed = bulk_inbox_try_acquire_write(
            peer_inbox_state, peer_inbox_generation, peer_inbox_ack,
            slot, epoch);
    claimed = __shfl_sync(FULL_MASK, claimed, 0);
    if (!claimed)
        return false;

    if (!lane_id)
        *send_count = 0;
    __syncwarp();

    int mark_words = (peer_v_local + 31) / 32;
    for (int w = lane_id; w < mark_words; w += WARP_SIZE)
    {
        unsigned mv = l3_atomic_exchange_acq_rel<cuda::thread_scope_device>(
            &remote_mark[w], 0u);
        if (!mv) continue;
        for (int b = 0; b < WARP_SIZE; b++)
        {
            if (!(mv & (1u << b))) continue;
            int r0 = w * WARP_SIZE + b;
            if (r0 >= peer_v_local) continue;
            int lidx = r0 + 1;
            VALUE_TYPE nd = atomicExch(&remote_cand[lidx], DIST_MAX);
            if (nd == DIST_MAX) continue;
            int slot = atomicAdd(send_count, 1);
            if (slot < peer_v_local)
                send_list[slot] = node_struct(peer_v_begin + lidx, nd);
        }
    }

    // BULK_ROUND 不依赖 hint 扫描；清理旧 summary，避免切换回退路径时遗留假阳。
    int mark_hint_words = (mark_words + 31) / 32;
    for (int h = lane_id; h < mark_hint_words; h += WARP_SIZE)
        atomicExch(&mark_hint[h], 0u);
    int mark_hint2_words = (mark_hint_words + 31) / 32;
    for (int h = lane_id; h < mark_hint2_words; h += WARP_SIZE)
        atomicExch(&mark_hint2[h], 0u);
    __syncwarp();

    int cnt = atomicAdd(send_count, 0);
    if (cnt < 0) cnt = 0;
    if (cnt > peer_v_local) cnt = peer_v_local;

    // 选择本 epoch 的槽位；同一槽位只在相隔 BULK_INBOX_SLOTS 个 epoch
    // 后复用。发布前等待该槽位的旧消息被接收卡确认。
    NODE_TYPE *slot_inbox = peer_inbox + slot * (peer_v_local + 1);
    int *slot_count = peer_inbox_count + slot;
    int *slot_epoch = peer_inbox_epoch + slot;
    bool candidate_filtered = false;

#if (GLOBAL_ROUND_CANDIDATE_PREFILTER == true)
    // 只对本轮已经聚合的 send_list 做 peer-node 快筛，不扫描完整对端分区。
    // node_data 单调不增：若此刻 peer_dist <= candidate，则该 candidate 以后
    // 不可能重新成为改进，安全丢弃；若读到并发更新前的旧值，只会保留冗余。
    // 发送卡 work 在自己的 quiesce 窗口内，但对端可能仍在本地计算，因此必须
    // 使用 volatile 读取并接受保守的旧值。
    // 首轮通常是从 DIST_MAX 初始状态传播的有效批次；避免为首轮所有候选
    // 付出一次额外的随机 P2P 读取。后续 round 才进入 stale feedback 高发区。
    if (peer_node_data != NULL && cnt > 0 && epoch > 1)
    {
        candidate_filtered = true;
        if (!lane_id)
            atomicExch(send_count, 0);
        __syncwarp();
        for (int i = lane_id; i < cnt; i += WARP_SIZE)
        {
            NODE_TYPE item = send_list[i];
            int lidx = item.id - peer_v_begin;
            if (lidx < 1 || lidx > peer_v_local)
                continue;
            VALUE_TYPE candidate = item.get_data();
            VALUE_TYPE peer_dist =
                l3_atomic_load_relaxed<cuda::thread_scope_system>(
                    &peer_node_data[lidx]);
            if (candidate >= peer_dist)
                continue;
            int out = atomicAdd(send_count, 1);
            if (out < peer_v_local)
                slot_inbox[out] = item;
        }
        __syncwarp();
        cnt = atomicAdd(send_count, 0);
        if (cnt < 0) cnt = 0;
        if (cnt > peer_v_local) cnt = peer_v_local;
    }
#endif

#if (GLOBAL_ROUND_DIRECT_P2P == true)
    // 此时对端 work/injection 已经 quiesce，候选源只有本卡 L3 warp 一个写者。
    // atomicMin 完成最终归约；只有真正胜出的候选才置对端 dirty/hint，接收卡
    // 随后沿既有 dirty -> L2 路径继续处理。默认 slot 只发布 count/完成标记；
    // 精确列表实验还把 atomicMin 真正胜出的条目复制到 slot，供接收卡恢复
    // dirty 信号，避免整分区 backstop。
#if (GLOBAL_ROUND_DIRECT_EXACT_LIST == true)
    if (!lane_id)
        atomicExch((int *)send_count, 0);
    __syncwarp();
#endif
    for (int i = lane_id; i < cnt; i += WARP_SIZE)
    {
        NODE_TYPE item = send_list[i];
        int lidx = item.id - peer_v_begin;
        if (lidx < 1 || lidx > peer_v_local)
            continue;
        VALUE_TYPE nd = item.get_data();
        VALUE_TYPE old = l3_atomic_min_system(&peer_node_data[lidx], nd);
#if (GLOBAL_ROUND_DIRECT_TRACE == true)
        if (item.id == 193 && nd < old)
        {
            atomicAdd(&g_direct_tx_193, 1ull);
            atomicMin(&g_direct_tx_193_min, (int)nd);
        }
        else if (item.id == 362 && nd < old)
        {
            atomicAdd(&g_direct_tx_362, 1ull);
            atomicMin(&g_direct_tx_362_min, (int)nd);
        }
#endif
        if (nd < old)
        {
            int r0 = lidx - 1;
#if (GLOBAL_ROUND_DIRECT_EXACT_LIST == true)
            int out = atomicAdd(send_count, 1);
            if (out < peer_v_local)
                slot_inbox[out] = item;
#endif
            l3_system_mark_publish(&peer_dirty_bitmap[r0 >> 5], 1u << (r0 & 31));
            l3_system_mark_publish(&peer_dirty_hint[r0 >> 10],
                                   1u << ((r0 >> 5) & 31));
        }
    }
    __syncwarp();
    int publish_count = cnt;
#if (GLOBAL_ROUND_DIRECT_EXACT_LIST == true)
    if (!lane_id)
        publish_count = atomicAdd(send_count, 0);
    publish_count = __shfl_sync(FULL_MASK, publish_count, 0);
#endif
    __threadfence_system();
    __syncwarp();
    if (!lane_id)
        l3_atomic_store_relaxed<cuda::thread_scope_system>(slot_count, publish_count);
    if (!lane_id)
        l3_atomic_store_relaxed<cuda::thread_scope_system>(slot_epoch, epoch);
    if (!lane_id)
        l3_atomic_store_release<cuda::thread_scope_system>(
            peer_inbox_state + slot, BULK_SLOT_READY);
    if (!lane_id)
    {
        atomicExch((int *)send_count, publish_count);
        l3_atomic_store_release<cuda::thread_scope_block>(
            (int *)publish_done, epoch);
    }
    __syncwarp();
    return true;
#else
    if (!candidate_filtered)
    {
        for (int i = lane_id; i < cnt; i += WARP_SIZE)
            slot_inbox[i] = send_list[i];
    }
    __syncwarp();
    __threadfence_system();
    __syncwarp();
    if (!lane_id)
        l3_atomic_store_relaxed<cuda::thread_scope_system>(slot_count, cnt);
    if (!lane_id)
        l3_atomic_store_relaxed<cuda::thread_scope_system>(slot_epoch, epoch);
    if (!lane_id)
        l3_atomic_store_release<cuda::thread_scope_system>(
            peer_inbox_state + slot, BULK_SLOT_READY);
    if (!lane_id)
        l3_atomic_store_release<cuda::thread_scope_block>(
            (int *)publish_done, epoch);
#if (BULK_DIAG == true)
    if (!lane_id)
        printf("BULK_PUB_DONE epoch=%d cnt=%d\\n", epoch, cnt);
#endif
    __syncwarp();
    return true;
#endif
}


// BULK_ROUND 通用路径：发布 L3 warp 已经聚合好的当前批次。
// 与 bulk_pack_publish_warp 不同，这里不再次消费 remote_mark/remote_cand，
// 而是直接复用 l3_lidx/l3_nd。这样发送侧仍可在 work warp 运行期间生产下一批
// remote_mark；当前批次只在本函数完成发布前保持不变。
//
// 多槽 inbox 的不变量：发送 epoch=t 前，只需等待同槽位的 t-BULK_INBOX_SLOTS
// 已被接收卡 ack；数据写完后才发布 count，最后发布该槽位 epoch。接收卡严格按
// rx_epoch+1 读取，完成 node_data/dirty 写入和 system fence 后才回写对应槽 ack，
// 因此不会覆盖仍在消费的槽位。
__device__ __forceinline__ void bulk_requeue_l3_batch(
    int count, const int *l3_lidx, const VALUE_TYPE *l3_nd,
    VALUE_TYPE *remote_cand, unsigned *remote_mark,
    unsigned *mark_hint, unsigned *mark_hint2)
{
    if (count <= 0 || l3_lidx == NULL || l3_nd == NULL
        || remote_cand == NULL || remote_mark == NULL)
        return;

    for (int i = threadIdx.x % WARP_SIZE; i < count; i += WARP_SIZE)
    {
        int lidx = l3_lidx[i];
        if (lidx < 1)
            continue;
        atomicMin(&remote_cand[lidx], l3_nd[i]);
        int r0 = lidx - 1;
        l3_device_mark_publish(&remote_mark[r0 >> 5], 1u << (r0 & 31));
        if (mark_hint != NULL)
            atomicOr(&mark_hint[r0 >> 10], 1u << ((r0 >> 5) & 31));
        if (mark_hint2 != NULL)
            atomicOr(&mark_hint2[r0 >> 15], 1u << ((r0 >> 10) & 31));
    }
    __syncwarp();
}

__device__ __forceinline__ bool bulk_publish_l3_batch(
    int tx_epoch, int lane_id, int peer_v_begin, int peer_v_local,
    int count, const int *l3_lidx, const VALUE_TYPE *l3_nd,
    NODE_TYPE *peer_inbox, int *peer_inbox_count,
    int *peer_inbox_epoch, int *peer_inbox_ack,
    int *peer_inbox_state, int *peer_inbox_generation
#if (L3_RX_FEEDBACK_MODE > 0)
    , l3_feedback_record *feedback = nullptr, l3_feedback_record *observed = nullptr
#endif
    )
{
    if (peer_inbox == NULL || peer_inbox_count == NULL
        || peer_inbox_epoch == NULL || peer_inbox_ack == NULL
        || peer_inbox_state == NULL || peer_inbox_generation == NULL)
        return false;

    if (count < 0) count = 0;
    if (count > peer_v_local) count = peer_v_local;

    int slot = bulk_inbox_slot(tx_epoch);
    bool claimed = false;
    if (!lane_id)
        claimed = bulk_inbox_try_acquire_write(
            peer_inbox_state, peer_inbox_generation, peer_inbox_ack,
            slot, tx_epoch);
    claimed = __shfl_sync(FULL_MASK, claimed, 0);
    if (!claimed)
        return false;

#if (L3_RX_FEEDBACK_MODE > 0)
    // ACK was checked by acquire. WRITING excludes the receiver until this
    // function publishes READY; take a stable copy before reusing the slot.
    if (!lane_id && feedback && observed && tx_epoch > BULK_INBOX_SLOTS) {
        auto *record = feedback + slot;
        *observed = {atomicAdd(&record->epoch, 0), atomicAdd(&record->count, 0),
                     atomicAdd(&record->improved, 0)};
        assert(observed->valid(tx_epoch - BULK_INBOX_SLOTS));
    }
#endif
    NODE_TYPE *slot_inbox = peer_inbox + slot * (peer_v_local + 1);
    int *slot_count = peer_inbox_count + slot;
    int *slot_epoch = peer_inbox_epoch + slot;

    for (int i = lane_id; i < count; i += WARP_SIZE)
    {
        int lidx = l3_lidx[i];
        if (lidx >= 1 && lidx <= peer_v_local)
        {
#if (BULK_TRACE == true)
            if (peer_v_begin + lidx == BULK_TRACE_NODE)
            {
                unsigned long long t = atomicAdd(&g_bulk_trace_tx, 1ull);
                if (t < 32)
                    printf("BULK_TRACE_TX epoch=%d id=%d nd=%.0f count=%d\\n",
                           tx_epoch, peer_v_begin + lidx, (double)l3_nd[i], count);
            }
#endif
            slot_inbox[i] = node_struct(peer_v_begin + lidx, l3_nd[i]);
        }
    }
    __syncwarp();
    __threadfence_system();
    __syncwarp();

    if (!lane_id)
        l3_atomic_store_relaxed<cuda::thread_scope_system>(slot_count, count);
    if (!lane_id)
        l3_atomic_store_relaxed<cuda::thread_scope_system>(slot_epoch, tx_epoch);
    if (!lane_id)
        l3_atomic_store_release<cuda::thread_scope_system>(
            peer_inbox_state + slot, BULK_SLOT_READY);
    __syncwarp();
    return true;
}

#endif
