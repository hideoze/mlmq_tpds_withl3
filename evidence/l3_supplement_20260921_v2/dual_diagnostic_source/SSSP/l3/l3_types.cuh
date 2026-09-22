#pragma once
#include "l3_feedback.h"

#if (L3_TILE_LOAN == true)
// One directional home-owned control/data slot.  The results pointer is
// rebound to the peer's local return buffer after both contexts allocate.
enum l3_tile_loan_state
{
    L3_TILE_LOAN_FREE = 0,
    L3_TILE_LOAN_CLAIMED = 1,
    L3_TILE_LOAN_READY = 2,
    L3_TILE_LOAN_EXECUTING = 3,
    L3_TILE_LOAN_RETURNED = 4,
    L3_TILE_LOAN_APPLIED = 5,
    L3_TILE_LOAN_ACKED = 6
};

struct l3_tile_loan_channel
{
    int *state;
    int *generation;
    int *seed_count;
    int *result_count;
    int *requeue_count;
    NODE_TYPE *tasks;
    VALUE_TYPE *task_dists;
    NODE_TYPE *results;
    VALUE_TYPE *result_dists;
    int *ready;
#if (L3_CONTINUATION == true)
    unsigned long long *continuation_metrics; // exec,seeds,rows,continued,records,released
#endif
};
#endif

#if (GLOBAL_ROUND_ASYNC == true)
// 旧 async candidate 的 host/device POD。CP2b 将其生命周期并入
// l3_host_context；协议 helper 仍按值接收，不改变 device 行为。
struct async_candidate_bank
{
    VALUE_TYPE *cand;
    unsigned *mark;
    unsigned *hint;
    unsigned *hint2;
    int *pending;
};

struct async_candidate_control
{
    int *active_bank;
    int *state;
    int *users;
};
#endif

// L3 只管理跨 GPU 候选，不拥有本地 MLMQ 的队列元素。
enum l3_dynamic_mode
{
    L3_DYNAMIC_EAGER = 0,
    L3_DYNAMIC_COALESCE = 1
};

struct l3_dynamic_state
{
    l3_dynamic_mode mode;
    int rounds_since_scan;
    int candidate_age;
    int last_batch_count;
    int forced_flushes;
};

struct l3_batch_meta
{
    int epoch;
    int count;
    int slot;
    int generation;
};

struct l3_stats
{
    unsigned long long candidates;
    unsigned long long unique_candidates;
    unsigned long long effective_updates;
    unsigned long long published_items;
    unsigned long long applied_items;
    unsigned long long forced_flushes;
};

// CP2a: 当前唯一 peer 的 L3 数据面视图。首版按值传给 manage kernel，
// 只替代分散参数，不改变候选、transport 或 termination 的所有权。
// 字段按 receiver-owned rx 与 peer-owned tx 明确方向；n=1 时 peer_id=-1，
// 所有 peer/transport 指针均允许为 NULL。
struct l3_channel_view
{
    int peer_id;
    int peer_v_begin;
    int peer_v_local;

    VALUE_TYPE *peer_node_data;
    unsigned *peer_dirty_bitmap;
    unsigned *peer_dirty_hint;
    int *peer_local_idle;

    VALUE_TYPE *candidate_values;
    unsigned *candidate_mark;
    unsigned *candidate_hint;
    unsigned *candidate_hint2;
    VALUE_TYPE *peer_cache;
    VALUE_TYPE *peer_cache_feedback;
    unsigned long long *remote_eff;
    unsigned long long *peer_remote_eff;

#if (L3_TILE_LOAN == true)
    // loan_out is home-owned by this GPU; loan_in is the peer's home-owned
    // slot with results/ready rebound to this GPU's local return resources.
    l3_tile_loan_channel loan_out;
    l3_tile_loan_channel loan_in;
#if (L3_MULTI_PRODUCER == true)
    unsigned long long *loan_gate, *peer_loan_gate;
    unsigned loan_epoch;
    unsigned long long *loan_producers;
#endif
    int *peer_row_ptr;
    int *peer_col_idx;
    VALUE_TYPE *peer_edge_data;
#endif

    int *quiesce_req;
    int *quiesce_ack;
    int *term_req;
    int *term_state;
    int *term_ack_slots;
    int *peer_term_state;
#if (L3_ADMISSION_BUDGET == true)
    int *admission;
    int *peer_admission;
#endif

    NODE_TYPE *send_list;
    NODE_TYPE *rx_payload;
    int *rx_count;
    int *rx_epoch;
    int *rx_ack;
    int *rx_state;
    int *rx_generation;
#if (L3_RX_FEEDBACK_MODE > 0)
    l3_feedback_record *rx_feedback;
    l3_feedback_record *tx_feedback;
#endif

    NODE_TYPE *tx_payload;
    int *tx_count;
    int *tx_epoch;
    int *tx_ack;
    int *tx_state;
    int *tx_generation;

    int *rx_read_head;
    int *rx_inflight;
    int *rx_active_slot;
    VALUE_TYPE *dense_rx_payload;
    VALUE_TYPE *dense_tx_payload;
    NODE_TYPE *rx_frontier;
    int *rx_frontier_head;
    int *rx_frontier_tail;
#if (L3_RX_EXPRESS == true)
    l3_rx_express_ring rx_express;
#endif
#if (L3_RX_L2_PULL == true)
    // Device-local RX manager -> worker event sequence.  It is not peer
    // transport state and is reset at every query.
    unsigned *rx_commit_seq;
#if (L3_RX_L2_PULL_DIAG == true)
    unsigned long long *rx_l2_pull_stats;
#endif
#endif
};

// CP2b: host 侧唯一 L3 资源 owner。gpu_ctx 暂时通过公开继承保留旧字段访问，
// 使 CP2b 只迁移 allocation/reset/release；CP2c/CP2d 再迁移 hot-path 调用点。
// peer 指针均为非 owning，只有本卡 cudaMalloc 得到的字段由 release() 释放。
struct l3_host_context
{
    bool enabled;
    int owner_gpu_id;
    int local_v;
    int peer_id;
    unsigned *dirty_bitmap;
    unsigned *dirty_hint;
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
    unsigned *async_rx_ready_bitmap;
    unsigned *async_rx_ready_hint;
#endif
    VALUE_TYPE *last_processed;
#if (L3_COMPLETED_ROWS == true)
    VALUE_TYPE *continuation_completed = nullptr;
#endif
#if (L3_WORK_DIAG == true)
    unsigned *source_expand_counts = nullptr;
#endif
    int *local_idle;

    VALUE_TYPE *peer_node_data;
    unsigned *peer_dirty_bitmap;
    unsigned *peer_dirty_hint;
    int peer_v_begin;
    int peer_v_local;
    int *peer_local_idle;

    VALUE_TYPE *remote_cand;
    unsigned *remote_mark;
    unsigned *mark_hint;
    unsigned *mark_hint2;
    VALUE_TYPE *peer_cache;
    VALUE_TYPE *peer_cache_feedback;
#if (GLOBAL_ROUND_ASYNC == true)
    async_candidate_bank cand_bank[2];
    async_candidate_control cand_ctl;
#endif
    unsigned long long *remote_eff;
    unsigned long long *peer_remote_eff;

#if (L3_TILE_LOAN == true)
    l3_tile_loan_channel loan_out;
    l3_tile_loan_channel loan_in;
    NODE_TYPE *loan_return_results;
    VALUE_TYPE *loan_return_dists;
    int *loan_ready;
#if (L3_MULTI_PRODUCER == true)
    unsigned long long *loan_gate=nullptr, *peer_loan_gate=nullptr;
    unsigned loan_epoch=0;
    unsigned long long *loan_producers=nullptr;
#endif
    int *peer_row_ptr;
    int *peer_col_idx;
    VALUE_TYPE *peer_edge_data;
#endif

#if (BULK_ROUND == true)
    NODE_TYPE *bulk_send_list;
    NODE_TYPE *bulk_inbox;
    int *bulk_inbox_count;
    int *bulk_inbox_epoch;
    int *bulk_inbox_ack;
    int *bulk_inbox_state;
    int *bulk_inbox_generation;
#if (L3_RX_FEEDBACK_MODE > 0)
    l3_feedback_record *bulk_feedback;
    l3_feedback_record *peer_bulk_feedback;
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    int *bulk_inbox_read_head;
    int *bulk_inbox_inflight;
    int *bulk_inbox_active_slot;
#endif
    int *bulk_quiesce_req;
    int *bulk_quiesce_ack;
    int *l3_term_req;
    int *l3_term_state;
    int *l3_term_ack_slots;
    int l3_term_ack_capacity;
    int *peer_l3_term_state;
#if (L3_ADMISSION_BUDGET == true)
    int *admission_state;
    int *peer_admission_symbol;
#endif
    NODE_TYPE *peer_bulk_inbox;
    int *peer_bulk_inbox_count;
    int *peer_bulk_inbox_epoch;
    int *peer_bulk_inbox_ack;
    int *peer_bulk_inbox_state;
    int *peer_bulk_inbox_generation;
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
    VALUE_TYPE *bulk_dense_inbox;
    VALUE_TYPE *peer_bulk_dense_inbox;
#endif
#if (BULK_FRONTIER_ENABLED == true)
    NODE_TYPE *bulk_frontier;
    int *bulk_frontier_head;
    int *bulk_frontier_tail;
#endif
#endif
#if (L3_RX_EXPRESS == true)
    // 本卡 RX manager -> 本卡 global_wid==0 worker 的 device-local SPSC ring。
    // 它不属于 peer transport，也不跨 GPU 共享。
    l3_rx_express_ring rx_express;
#endif
#if (L3_RX_L2_PULL == true)
    // 本卡 RX manager -> 本卡每个 work block 的 wid==0 warp 的 device-local hint。
    // 该指针不跨 GPU 共享，也不参与候选值/ACK 的所有权。
    unsigned *rx_commit_seq;
#if (L3_RX_L2_PULL_SINGLE_CLAIM == true)
    // At most one selected warp claims an RX sequence for the extra q2 read.
    unsigned *rx_pull_claim_seq;
#endif
#if (L3_RX_L2_PULL_DIAG == true)
    unsigned long long *rx_l2_pull_stats;
#endif
#endif

    void initialize(int gpu_id);
    int allocate_local(int v_local);
    int allocate_peer(int peer_id, int peer_v_begin, int peer_v_local);
    void bind_peer_data(VALUE_TYPE *node_data, unsigned *dirty_bitmap,
                        unsigned *dirty_hint, int *local_idle);
    void bind_peer_transport(const l3_host_context &peer);
#if (L3_TILE_LOAN == true)
    void bind_peer_graph(int *row_ptr, int *col_idx, VALUE_TYPE *edge_data);
#endif
    void disable();
    int reset_query(VALUE_TYPE init_max);
    void release();
    l3_channel_view make_channel_view() const;
};
