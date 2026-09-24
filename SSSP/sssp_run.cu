#include "sssp.cuh"
#ifndef L3_ACK_SCAN
#define L3_ACK_SCAN false
#endif
#ifndef L3_ACK_SCAN_FAULT
#define L3_ACK_SCAN_FAULT false
#endif
#if (L3_ACK_SCAN_FAULT == true && (L3_ACK_SCAN == false || L3_PROGRESS_DIAG == false))
#error "ACK scan faults require explicit ACK scan diagnostic mode"
#endif
#if (L3_ACK_SCAN == true && (L3_WORKER_RECOVERY == false || L3_TERM_WAIT_ACK == false || L3_STICKY_READY == true))
#error "ACK scan requires worker recovery, ACK wait, and ordinary READY cancellation"
#endif
#if (L3_IDLE_TOKEN_PROBE == true && (L3_ACK_SCAN == false || BULK_ROUND == false || L3_TERM_HANDSHAKE == false || BULK_EPOCH == true || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || SEED_BARRIER == true || L3_TILE_LOAN == true || L3_RX_EXPRESS == true))
#error "idle token probe requires ordinary BULK ACK scan without loan or express workers"
#endif
#if (L3_TERM_ONLY_WORKER == true && (L3_IDLE_TOKEN_PROBE == false || L3_ACK_SCAN == false || BULK_ROUND == false || L3_TERM_HANDSHAKE == false || BULK_EPOCH == true || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || SEED_BARRIER == true || BULK_FRONTIER_ENABLED == true || L3_TILE_LOAN == true || L3_RX_EXPRESS == true))
#error "term-only worker requires ordinary BULK with idle-gated ACK scan"
#endif
#include "ml_queue.cuh"
#include "graph_partition.h"
#include "l3/l3_boundary_index.cuh"
#include "l3/l3_recovery_domain.cuh"
#include "l3/l3_rx_express.cuh"
#include "l3/l3_rx_l2_pull.cuh"
#include "l3/l3_types.cuh"
#include "l3/l3_completed_rows.cuh"
#if (L3_MULTI_PRODUCER == true)
#include "l3/l3_admission_gate.cuh"
#if (BULK_ROUND == false || L3_TERM_HANDSHAKE == false || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || BULK_EPOCH == true || SEED_BARRIER == true)
#error "multi producer initially supports default BULK term handshake only"
#endif
#endif
#if (L3_COMPLETED_ROWS == true)
__device__ VALUE_TYPE *g_l3_completed_rows=nullptr;
#endif
#include "l3/l3_reset.cuh"
#include "l3/l3_candidate.cuh"
#include "l3/l3_queue.cuh"
#include "l3/l3_transport.cuh"
#include "l3/l3_bulk.cuh"
#include "l3/l3_diagnostics.cuh"
#include "l3/l3_progress.cuh"
#include "l3/l3_work_diag.cuh"
#include "l3/l3_wait_diag.cuh"
#include "l3/l3_recovery.h"
#include "l3/l3_worker_recovery.cuh"
#if (L3_ACK_SCAN == true)
__device__ __forceinline__ bool l3_ack_scan_token_live(int *req, int token,
                                                    volatile int *stop, int lane) {
    int live=0;
    if(!lane) live=l3_atomic_load_acquire<cuda::thread_scope_device>(req)==token &&
                   l3_atomic_load_acquire<cuda::thread_scope_device>(
                       const_cast<int *>(stop))==0;
    return __shfl_sync(FULL_MASK,live,0)!=0;
}
#endif
#include "l3/l3_admission_budget.cuh"
#include "l3/l3_receive.cuh"
#include "l3/l3_window.h"
#include "l3/l3_event_gate.h"
#include "l3/l3_feedback_policy.cuh"
#include "l3/l3_collect.cuh"
#include "l3/l3_mark_check.cuh"
#include "l3/l3_probe_policy.h"
#include "l3/l3_confirm_policy.h"
#if (L3_STICKY_READY == true && (BULK_ROUND == false || L3_TERM_HANDSHAKE == false || L3_DIRECT_RX == false || L3_RETAIN_TX == false || L3_CONFIRM_BACKOFF == true || L3_BOUNDED_RECOVERY == true || BULK_EPOCH == true || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || SEED_BARRIER == true))
#error "sticky READY requires default BULK transport and exact confirmation"
#endif
#if (L3_TERM_WAIT_ACK == true && (BULK_ROUND == false || L3_TERM_HANDSHAKE == false || L3_DIRECT_RX == false || L3_RETAIN_TX == false || L3_CONFIRM_BACKOFF == true || L3_BOUNDED_RECOVERY == true || L3_RECOVERY_MODE != 0 || BULK_EPOCH == true || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || SEED_BARRIER == true))
#error "ACK wait requires default BULK transport and exact confirmation"
#endif
#if (L3_BOUNDED_RECOVERY == true)
#include "l3/l3_recovery_cursor.h"
#if (BULK_ROUND == false || L3_TERM_HANDSHAKE == false || L3_DIRECT_RX == false || L3_RETAIN_TX == false || L3_CONFIRM_BACKOFF == true || BULK_EPOCH == true || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || SEED_BARRIER == true)
#error "bounded recovery requires default BULK transport and exact confirmation"
#endif
#endif
#ifndef L3_FEEDBACK_DRAIN
// ACK-scan/window mode must discharge a failed frozen mark check before
// freezing producers again. Otherwise sparse summaries can repeatedly admit
// a probe while the TX window has not consumed the authoritative candidates.
// Keep unrelated legacy protocols unchanged; explicit false is an ablation.
#define L3_FEEDBACK_DRAIN (L3_ACK_SCAN == true && L3_WINDOW_MODE != 0 && \
    BULK_ROUND == true && L3_TERM_HANDSHAKE == true && BULK_EPOCH == false && \
    GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && SEED_BARRIER == false)
#endif
#ifndef L3_EFFECTIVE_FEEDBACK
#define L3_EFFECTIVE_FEEDBACK false
#endif
#if (L3_FEEDBACK_DRAIN == true && !(BULK_ROUND == true && L3_TERM_HANDSHAKE == true && BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && SEED_BARRIER == false && L3_WINDOW_MODE != 0))
#error "feedback drain requires the default BULK handshake and window scheduler"
#endif
#if (L3_EFFECTIVE_FEEDBACK == true)
// This prototype reads TX-local l3_eff. BULK/direct-RX never increments that
// counter; an epoch-associated RX winner feedback channel is still missing.
#error "L3_EFFECTIVE_FEEDBACK unavailable: BULK RX winner feedback is not connected"
#endif
#ifndef L3_FAULT_HIDE_HINTS
#define L3_FAULT_HIDE_HINTS false
#endif
#ifndef L3_TEST_SKIP_FROZEN_MARK
#define L3_TEST_SKIP_FROZEN_MARK false
#endif
#if (L3_TEST_SKIP_FROZEN_MARK == true && L3_FAULT_HIDE_HINTS == false)
#error "test-only frozen mark bypass requires explicit hint fault injection"
#endif
#if (L3_FAULT_HIDE_HINTS == true && !(BULK_ROUND == true && L3_TERM_HANDSHAKE == true && BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && SEED_BARRIER == false))
#error "hint fault injection is restricted to the default BULK handshake path"
#endif
#if (L3_DEFER_BUSY_PROBE == true && !(BULK_ROUND == true && L3_TERM_HANDSHAKE == true && BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && SEED_BARRIER == false))
#error "busy probe deferral requires the default BULK handshake path"
#endif
#if (L3_CONFIRM_BACKOFF == true && !(BULK_ROUND == true && L3_TERM_HANDSHAKE == true && BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && SEED_BARRIER == false))
#error "confirm backoff requires the default BULK handshake path"
#endif
#include "l3/l3_order.cuh"
#include "l3/l3_termination.cuh"
#include "l3/l3_metrics.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "benchmark.h"

mlmq_benchmark g_benchmark;
#include "l3/l3_chain_partition.cuh"
#include "l3/l3_region_index.cuh"

#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
#define ASYNC_RX_COLLECT_ATTR __noinline__
#else
#define ASYNC_RX_COLLECT_ATTR __forceinline__
#endif


#if (TIMELINE64 == true)
typedef unsigned long long timeline_clock_type;
#define TIMELINE_CLOCK() clock64()
#else
typedef unsigned int timeline_clock_type;
#define TIMELINE_CLOCK() clock()
#endif

// per-GPU context (isolate global device pointers, v1 lesson: concurrent
// global pointer overwrite caused GPU0/GPU1 pointer swapping)
// CP2b 通过 host owner 基类集中 L3 生命周期；公开继承只作为 CP2c/CP2d 前的
// 行为等价兼容层，图、seed 与 ghost 资源仍由 gpu_ctx 保持。
struct gpu_ctx : public l3_host_context
{
    int m, nnz;
    int v_begin, v_end, v_local;   // 本卡归属区间 (0-based)
    int gpu_id;
    int *RowPtr;
    int *ColIdx;
    VALUE_TYPE *edge_data;
    VALUE_TYPE *node_data;         // 本卡权威距离, 局部 1-based: node_data[gid - v_begin]
    int *global_exit;
    // SEED_BARRIER（解法0）: phase 状态机 + seed_ready 跨卡屏障
    //   phase: 1=本地计算(禁 flush) 2=灌值(只写 node_data) 3=seed_ready 已置
    //   seed_ready: 源卡灌完最终值后 P2P 置 1；接收卡见 1 才启动 backstop/注入
    int *phase;
    int *seed_ready;
    int *peer_seed_ready;            // = gctx[j].seed_ready（源卡 P2P 写）
    // SEED_BARRIER: 接收卡一次性全量注入完成标志（work 门控读 L2，first_pos 保持 0）
    int *seed_inject_done;
    // SEED_BARRIER: 一次性全量注入的全局收集缓冲（node_struct 数组 + 计数）
    //   接收卡: gpu0 灌值时 P2P 写入（peer_seed_list = gctx[j].seed_list），
    //   gpu1 免 419万 node_data 扫描直接消费（注入块读列表 + write_through）
    NODE_TYPE *seed_list;
    int *seed_list_cnt;
    int *peer_seed_list_cnt;   // = gctx[j].seed_list_cnt（gpu0 灌值时 P2P atomicAdd）
    NODE_TYPE *peer_seed_list; // = gctx[j].seed_list（gpu0 灌值时 P2P 写）
#if (GHOST_DEPTH > 0)
    // 方案 C（ghost 顶点）: 源卡持有的对端边界顶点 ghost 子图（D 跳闭包）
    int ghost_num;                 // ghost 顶点数
    int *ghost_row_start;          // [ghost_num+1]
    int *ghost_col;                // 出边目标（全局 1-based）
    VALUE_TYPE *ghost_edge_data;   // 出边权重
    VALUE_TYPE *ghost_node_data;   // [ghost_num] ghost 距离（本卡探索值，初始 DIST_MAX）
    int *ghost_id_to_idx;          // [peer_v_local] peer 局部 0-based -> ghost idx（-1=非 ghost）
    unsigned *ghost_mark;          // [(ghost_num+31)/32] ghost 改进事件信号
#endif
    graph_partition *part;         // 本卡子图
};
gpu_ctx gctx[MAX_GPU];
#include "l3/l3_final_audit.cuh"

void l3_host_context::initialize(int gpu_id)
{
    l3_host_context empty = {};
    *this = empty;
    owner_gpu_id = gpu_id;
    peer_id = -1;
}

int l3_host_context::allocate_local(int v_local)
{
    cudaSetDevice(owner_gpu_id);
    local_v = v_local;

    int dwords = (local_v + 31) / 32;
    int hwords = (dwords + 31) / 32;
    cudaMalloc(&dirty_bitmap, sizeof(unsigned) * dwords);
    cudaMemset(dirty_bitmap, 0, sizeof(unsigned) * dwords);
    cudaMalloc(&dirty_hint, sizeof(unsigned) * hwords);
    cudaMemset(dirty_hint, 0, sizeof(unsigned) * hwords);
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
    cudaMalloc(&async_rx_ready_bitmap, sizeof(unsigned) * dwords);
    cudaMemset(async_rx_ready_bitmap, 0, sizeof(unsigned) * dwords);
    cudaMalloc(&async_rx_ready_hint, sizeof(unsigned) * hwords);
    cudaMemset(async_rx_ready_hint, 0, sizeof(unsigned) * hwords);
#endif

#if (L3_SOURCE_EXPAND_DIAG == true)
    cudaMalloc(&source_expand_counts, sizeof(unsigned) * (local_v + 1));
    cudaMemset(source_expand_counts, 0, sizeof(unsigned) * (local_v + 1));
#endif

    VALUE_TYPE *init_h = new VALUE_TYPE[local_v + 1];
    for (int i = 0; i <= local_v; i++) init_h[i] = DIST_MAX;
    cudaMalloc(&last_processed, sizeof(VALUE_TYPE) * (local_v + 1));
    cudaMemcpy(last_processed, init_h, sizeof(VALUE_TYPE) * (local_v + 1),
               cudaMemcpyHostToDevice);
#if (L3_COMPLETED_ROWS == true)
    g_benchmark.require(cudaMalloc(&continuation_completed,sizeof(VALUE_TYPE)*(local_v+1))==cudaSuccess,"completion allocation");
    g_benchmark.require(cudaMemcpy(continuation_completed,init_h,sizeof(VALUE_TYPE)*(local_v+1),cudaMemcpyHostToDevice)==cudaSuccess,"completion initialization");
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_completed_rows,&continuation_completed,sizeof(continuation_completed))==cudaSuccess,"completion binding");
#endif
    delete[] init_h;

    cudaMalloc(&local_idle, sizeof(int));
    cudaMemset(local_idle, 0, sizeof(int));
    cudaMalloc(&remote_eff, sizeof(unsigned long long));
    cudaMemset(remote_eff, 0, sizeof(unsigned long long));

#if (L3_TILE_LOAN == true)
    // Results for a loan arriving from the peer live on this GPU so the
    // borrower helper can write them locally; the home reads this buffer via
    // the P2P pointer rebound in bind_peer_transport().
    cudaMalloc(&loan_return_results,
               sizeof(NODE_TYPE) * L3_TILE_LOAN_RESULT_CAP);
    cudaMalloc(&loan_return_dists,
               sizeof(VALUE_TYPE) * L3_TILE_LOAN_RESULT_CAP);
    cudaMalloc(&loan_ready, sizeof(int));
    cudaMemset(loan_ready, 0, sizeof(int));
#if (L3_MULTI_PRODUCER == true)
    g_benchmark.require(cudaMalloc(&loan_gate,sizeof(unsigned long long))==cudaSuccess,"loan gate allocation");
    g_benchmark.require(cudaMalloc(&loan_producers,L3_PRODUCER_COUNTER_WORDS*sizeof(unsigned long long))==cudaSuccess,"loan producer counters allocation");
#endif
#endif

#if (L3_RX_EXPRESS == true)
    // The express channel is local to this CUDA device.  It is allocated in
    // allocate_local (not allocate_peer), because the manager and worker on
    // this same GPU are its only producer/consumer.
    cudaMalloc(&rx_express.payload,
               sizeof(NODE_TYPE) * L3_RX_EXPRESS_SLOTS * L3_RX_EXPRESS_BATCH);
    cudaMalloc(&rx_express.count, sizeof(int) * L3_RX_EXPRESS_SLOTS);
    cudaMalloc(&rx_express.sequence,
               sizeof(unsigned long long) * L3_RX_EXPRESS_SLOTS);
    cudaMalloc(&rx_express.head, sizeof(unsigned long long));
    cudaMalloc(&rx_express.tail, sizeof(unsigned long long));
    unsigned long long express_sequence_init[L3_RX_EXPRESS_SLOTS];
    for (int i = 0; i < L3_RX_EXPRESS_SLOTS; i++)
        express_sequence_init[i] = (unsigned long long)i;
    cudaMemcpy(rx_express.sequence, express_sequence_init,
               sizeof(express_sequence_init), cudaMemcpyHostToDevice);
    cudaMemset(rx_express.count, 0, sizeof(int) * L3_RX_EXPRESS_SLOTS);
    cudaMemset(rx_express.head, 0, sizeof(unsigned long long));
    cudaMemset(rx_express.tail, 0, sizeof(unsigned long long));
#if (L3_RX_EXPRESS_DIAG == true)
    cudaMalloc(&rx_express.stats,
               sizeof(unsigned long long) * L3_RX_EXPRESS_STAT_COUNT);
    cudaMemset(rx_express.stats, 0,
               sizeof(unsigned long long) * L3_RX_EXPRESS_STAT_COUNT);
#endif
#endif

#if (L3_RX_L2_PULL == true)
    cudaMalloc(&rx_commit_seq, sizeof(unsigned));
    cudaMemset(rx_commit_seq, 0, sizeof(unsigned));
#if (L3_RX_L2_PULL_SINGLE_CLAIM == true)
    cudaMalloc(&rx_pull_claim_seq, sizeof(unsigned));
    cudaMemset(rx_pull_claim_seq, 0, sizeof(unsigned));
#endif
#if (L3_RX_L2_PULL_DIAG == true)
    cudaMalloc(&rx_l2_pull_stats,
               sizeof(unsigned long long) * L3_RX_L2_PULL_STATS_COUNT);
    cudaMemset(rx_l2_pull_stats, 0,
               sizeof(unsigned long long) * L3_RX_L2_PULL_STATS_COUNT);
#endif
#endif

#if (BULK_ROUND == true)
    cudaMalloc(&bulk_quiesce_req, sizeof(int));
    cudaMalloc(&bulk_quiesce_ack, sizeof(int));
    cudaMemset(bulk_quiesce_req, 0, sizeof(int));
    cudaMemset(bulk_quiesce_ack, 0, sizeof(int));

    cudaDeviceProp term_prop;
    cudaGetDeviceProperties(&term_prop, owner_gpu_id);
    int term_work_blocks = term_prop.multiProcessorCount - 1;
    if (term_work_blocks < 1) term_work_blocks = 1;
    l3_term_ack_capacity = term_work_blocks * WARP_NUM_PER_BLOCK;
    cudaMalloc(&l3_term_req, sizeof(int));
    cudaMalloc(&l3_term_state, sizeof(int));
    cudaMalloc(&l3_term_ack_slots, sizeof(int) * l3_term_ack_capacity);
    cudaMemset(l3_term_req, 0, sizeof(int));
    cudaMemset(l3_term_state, 0, sizeof(int));
#if (L3_ADMISSION_BUDGET == true)
    g_benchmark.require(cudaMalloc(&admission_state,2*sizeof(int))==cudaSuccess,"admission allocation");
    g_benchmark.require(cudaMemset(admission_state,0,2*sizeof(int))==cudaSuccess,"admission state init");
    const int advice_init[4]={0,0,DIST_MAX,0};
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_admission,advice_init,sizeof(advice_init))==cudaSuccess,"admission init");
    peer_admission_symbol=nullptr;
#endif
    cudaMemset(l3_term_ack_slots, 0, sizeof(int) * l3_term_ack_capacity);

#if (BULK_FRONTIER_ENABLED == true)
    cudaMalloc(&bulk_frontier, sizeof(NODE_TYPE) * (local_v + 1));
    cudaMalloc(&bulk_frontier_head, sizeof(int));
    cudaMalloc(&bulk_frontier_tail, sizeof(int));
    cudaMemset(bulk_frontier_head, 0, sizeof(int));
    cudaMemset(bulk_frontier_tail, 0, sizeof(int));
#endif
#endif
    return 0;
}

int l3_host_context::allocate_peer(int peer_id_in, int peer_v_begin_in,
                                   int peer_v_local_in)
{
    cudaSetDevice(owner_gpu_id);
    enabled = true;
    peer_id = peer_id_in;
    peer_v_begin = peer_v_begin_in;
    peer_v_local = peer_v_local_in;

    int mark_words = (peer_v_local + 31) / 32;
    int mh_words = (mark_words + 31) / 32;
    int mh2_words = (mh_words + 31) / 32;
#if (GLOBAL_ROUND_ASYNC == true)
    for (int b = 0; b < 2; b++)
    {
        cudaMalloc(&cand_bank[b].cand, sizeof(VALUE_TYPE) * (peer_v_local + 1));
        VALUE_TYPE *cand_h = new VALUE_TYPE[peer_v_local + 1];
        for (int i = 0; i <= peer_v_local; i++) cand_h[i] = DIST_MAX;
        cudaMemcpy(cand_bank[b].cand, cand_h,
                   sizeof(VALUE_TYPE) * (peer_v_local + 1), cudaMemcpyHostToDevice);
        delete[] cand_h;
        cudaMalloc(&cand_bank[b].mark, sizeof(unsigned) * mark_words);
        cudaMalloc(&cand_bank[b].hint, sizeof(unsigned) * mh_words);
        cudaMalloc(&cand_bank[b].hint2, sizeof(unsigned) * mh2_words);
        cudaMalloc(&cand_bank[b].pending, sizeof(int));
        cudaMemset(cand_bank[b].mark, 0, sizeof(unsigned) * mark_words);
        cudaMemset(cand_bank[b].hint, 0, sizeof(unsigned) * mh_words);
        cudaMemset(cand_bank[b].hint2, 0, sizeof(unsigned) * mh2_words);
        cudaMemset(cand_bank[b].pending, 0, sizeof(int));
    }
    remote_cand = cand_bank[0].cand;
    remote_mark = cand_bank[0].mark;
    mark_hint = cand_bank[0].hint;
    mark_hint2 = cand_bank[0].hint2;
    cudaMalloc(&cand_ctl.active_bank, sizeof(int));
    cudaMalloc(&cand_ctl.state, sizeof(int) * 2);
    cudaMalloc(&cand_ctl.users, sizeof(int) * 2);
    int active_init = 0;
    int state_init[2] = {ASYNC_CAND_BANK_ACTIVE, ASYNC_CAND_BANK_FREE};
    int users_init[2] = {0, 0};
    cudaMemcpy(cand_ctl.active_bank, &active_init, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cand_ctl.state, state_init, sizeof(state_init), cudaMemcpyHostToDevice);
    cudaMemcpy(cand_ctl.users, users_init, sizeof(users_init), cudaMemcpyHostToDevice);
#else
    cudaMalloc(&remote_cand, sizeof(VALUE_TYPE) * (peer_v_local + 1));
    VALUE_TYPE *cand_h = new VALUE_TYPE[peer_v_local + 1];
    for (int i = 0; i <= peer_v_local; i++) cand_h[i] = DIST_MAX;
    cudaMemcpy(remote_cand, cand_h, sizeof(VALUE_TYPE) * (peer_v_local + 1),
               cudaMemcpyHostToDevice);
    delete[] cand_h;
    cudaMalloc(&remote_mark, sizeof(unsigned) * mark_words);
    cudaMalloc(&mark_hint, sizeof(unsigned) * mh_words);
    cudaMalloc(&mark_hint2, sizeof(unsigned) * mh2_words);
    cudaMemset(remote_mark, 0, sizeof(unsigned) * mark_words);
    cudaMemset(mark_hint, 0, sizeof(unsigned) * mh_words);
    cudaMemset(mark_hint2, 0, sizeof(unsigned) * mh2_words);
#endif

    cudaMalloc(&peer_cache, sizeof(VALUE_TYPE) * (peer_v_local + 1));
    VALUE_TYPE *cache_h = new VALUE_TYPE[peer_v_local + 1];
    for (int i = 0; i <= peer_v_local; i++) cache_h[i] = DIST_MAX;
    cudaMemcpy(peer_cache, cache_h, sizeof(VALUE_TYPE) * (peer_v_local + 1),
               cudaMemcpyHostToDevice);
    delete[] cache_h;

#if (L3_TILE_LOAN == true)
    // This slot is home-owned by owner_gpu_id.  Its results pointer is filled
    // only after the peer's local return buffer has been allocated.
    cudaMalloc(&loan_out.state, sizeof(int));
    cudaMalloc(&loan_out.generation, sizeof(int));
    cudaMalloc(&loan_out.seed_count, sizeof(int));
    cudaMalloc(&loan_out.result_count, sizeof(int));
    cudaMalloc(&loan_out.requeue_count, sizeof(int));
#if (L3_CONTINUATION == true)
    cudaMalloc(&loan_out.continuation_metrics,6*sizeof(unsigned long long));
    cudaMemset(loan_out.continuation_metrics,0,6*sizeof(unsigned long long));
#endif
    cudaMalloc(&loan_out.tasks,
               sizeof(NODE_TYPE) * L3_TILE_LOAN_MAX_SEEDS);
    cudaMalloc(&loan_out.task_dists,
               sizeof(VALUE_TYPE) * L3_TILE_LOAN_MAX_SEEDS);
    cudaMemset(loan_out.state, 0, sizeof(int));
    cudaMemset(loan_out.generation, 0, sizeof(int));
    cudaMemset(loan_out.seed_count, 0, sizeof(int));
    cudaMemset(loan_out.result_count, 0, sizeof(int));
    cudaMemset(loan_out.requeue_count, 0, sizeof(int));
    loan_out.results = NULL;
    loan_out.result_dists = NULL;
    loan_out.ready = NULL;
#endif

#if (BULK_ROUND == true)
    cudaMalloc(&bulk_send_list, sizeof(NODE_TYPE) * (peer_v_local + 1));
    cudaMalloc(&bulk_inbox,
               sizeof(NODE_TYPE) * (local_v + 1) * BULK_INBOX_SLOTS);
    cudaMalloc(&bulk_inbox_count, sizeof(int) * BULK_INBOX_SLOTS);
#if (L3_RX_FEEDBACK_MODE > 0)
    g_benchmark.require(cudaMalloc(&bulk_feedback, sizeof(l3_feedback_record) * BULK_INBOX_SLOTS)
                        == cudaSuccess, "RX feedback allocation");
    cudaMemset(bulk_feedback, 0, sizeof(l3_feedback_record) * BULK_INBOX_SLOTS);
#endif
    cudaMalloc(&bulk_inbox_epoch, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMalloc(&bulk_inbox_ack, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMalloc(&bulk_inbox_state, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMalloc(&bulk_inbox_generation, sizeof(int) * BULK_INBOX_SLOTS);
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    cudaMalloc(&bulk_inbox_read_head, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMalloc(&bulk_inbox_inflight, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMalloc(&bulk_inbox_active_slot, sizeof(int));
#endif
    cudaMemset(bulk_inbox_count, 0, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMemset(bulk_inbox_epoch, 0, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMemset(bulk_inbox_ack, 0, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMemset(bulk_inbox_state, 0, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMemset(bulk_inbox_generation, 0, sizeof(int) * BULK_INBOX_SLOTS);
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    cudaMemset(bulk_inbox_read_head, 0, sizeof(int) * BULK_INBOX_SLOTS);
    cudaMemset(bulk_inbox_inflight, 0, sizeof(int) * BULK_INBOX_SLOTS);
    int no_slot = -1;
    cudaMemcpy(bulk_inbox_active_slot, &no_slot, sizeof(int), cudaMemcpyHostToDevice);
#endif
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
    cudaMalloc(&bulk_dense_inbox, sizeof(VALUE_TYPE) * (local_v + 1));
#endif
#endif
    return 0;
}

void l3_host_context::bind_peer_data(VALUE_TYPE *node_data,
                                     unsigned *dirty_bitmap_in,
                                     unsigned *dirty_hint_in,
                                     int *local_idle_in)
{
    peer_node_data = node_data;
    peer_dirty_bitmap = dirty_bitmap_in;
    peer_dirty_hint = dirty_hint_in;
    peer_local_idle = local_idle_in;
}

void l3_host_context::bind_peer_transport(const l3_host_context &peer)
{
#if (BULK_ROUND == true)
    peer_l3_term_state = peer.l3_term_state;
#if (L3_ADMISSION_BUDGET == true)
    peer_admission_symbol=peer.admission_state;
#endif
    peer_bulk_inbox = peer.bulk_inbox;
#if (L3_RX_FEEDBACK_MODE > 0)
    peer_bulk_feedback = peer.bulk_feedback;
#endif
    peer_bulk_inbox_count = peer.bulk_inbox_count;
    peer_bulk_inbox_epoch = peer.bulk_inbox_epoch;
    peer_bulk_inbox_ack = peer.bulk_inbox_ack;
    peer_bulk_inbox_state = peer.bulk_inbox_state;
    peer_bulk_inbox_generation = peer.bulk_inbox_generation;
    peer_cache_feedback = peer.peer_cache;
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
    peer_bulk_dense_inbox = peer.bulk_dense_inbox;
#endif
#endif
    peer_remote_eff = peer.remote_eff;
#if (L3_TILE_LOAN == true)
    loan_out.results = peer.loan_return_results;
    loan_out.result_dists = peer.loan_return_dists;
    loan_out.ready = peer.loan_ready;
    loan_in = peer.loan_out;
    loan_in.results = loan_return_results;
    loan_in.result_dists = loan_return_dists;
    loan_in.ready = loan_ready;
#if (L3_MULTI_PRODUCER == true)
    peer_loan_gate=peer.loan_gate;
#endif
#endif
}

#if (L3_TILE_LOAN == true)
void l3_host_context::bind_peer_graph(int *row_ptr, int *col_idx,
                                      VALUE_TYPE *edge_data_in)
{
    peer_row_ptr = row_ptr;
    peer_col_idx = col_idx;
    peer_edge_data = edge_data_in;
}
#endif

void l3_host_context::disable()
{
    enabled = false;
    peer_id = -1;
    peer_v_begin = 0;
    peer_v_local = 0;
    peer_node_data = NULL;
    peer_dirty_bitmap = NULL;
    peer_dirty_hint = NULL;
    peer_local_idle = NULL;
    peer_cache_feedback = NULL;
    peer_remote_eff = NULL;
#if (L3_TILE_LOAN == true)
    loan_out.results = NULL;
    loan_out.result_dists = NULL;
    loan_out.ready = NULL;
    loan_in = {};
    peer_row_ptr = NULL;
    peer_col_idx = NULL;
    peer_edge_data = NULL;
#endif
#if (BULK_ROUND == true)
    peer_l3_term_state = NULL;
#if (L3_ADMISSION_BUDGET == true)
    peer_admission_symbol=nullptr;
#endif
    peer_bulk_inbox = NULL;
#if (L3_RX_FEEDBACK_MODE > 0)
    peer_bulk_feedback = NULL;
#endif
    peer_bulk_inbox_count = NULL;
    peer_bulk_inbox_epoch = NULL;
    peer_bulk_inbox_ack = NULL;
    peer_bulk_inbox_state = NULL;
    peer_bulk_inbox_generation = NULL;
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
    peer_bulk_dense_inbox = NULL;
#endif
#endif
}

int l3_host_context::reset_query(VALUE_TYPE init_max)
{
    cudaSetDevice(owner_gpu_id);
    int dwords = (local_v + 31) / 32;
    int hwords = (dwords + 31) / 32;
    cudaMemset(dirty_bitmap, 0, sizeof(unsigned) * dwords);
    cudaMemset(dirty_hint, 0, sizeof(unsigned) * hwords);
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
    if (async_rx_ready_bitmap)
        cudaMemset(async_rx_ready_bitmap, 0, sizeof(unsigned) * dwords);
    if (async_rx_ready_hint)
        cudaMemset(async_rx_ready_hint, 0, sizeof(unsigned) * hwords);
#endif
    g_benchmark.require(l3_fill_values(last_processed, size_t(local_v) + 1, init_max)
                        == cudaSuccess, "L3 last_processed reset launch");
#if (L3_COMPLETED_ROWS == true)
    g_benchmark.require(l3_fill_values(continuation_completed,size_t(local_v)+1,init_max)==cudaSuccess,"completion reset launch");
#endif
    cudaMemset(local_idle, 0, sizeof(int));
    if (remote_eff)
        cudaMemset(remote_eff, 0, sizeof(unsigned long long));

#if (L3_TILE_LOAN == true)
    if (loan_ready)
        cudaMemset(loan_ready, 0, sizeof(int));
#if (L3_MULTI_PRODUCER == true)
    g_benchmark.require(++loan_epoch!=0,"loan query epoch overflow");
    auto initial_gate=l3_gate_initial(loan_epoch);
    g_benchmark.require(cudaMemcpy(loan_gate,&initial_gate,sizeof(initial_gate),cudaMemcpyHostToDevice)==cudaSuccess,"loan gate reset");
    g_benchmark.require(cudaMemset(loan_producers,0,L3_PRODUCER_COUNTER_WORDS*sizeof(unsigned long long))==cudaSuccess,"loan producer reset");
#endif
    if (loan_return_results)
        cudaMemset(loan_return_results, 0,
                   sizeof(NODE_TYPE) * L3_TILE_LOAN_RESULT_CAP);
    if (loan_return_dists)
        cudaMemset(loan_return_dists, 0,
                   sizeof(VALUE_TYPE) * L3_TILE_LOAN_RESULT_CAP);
    if (loan_out.state) cudaMemset(loan_out.state, 0, sizeof(int));
#if (L3_CONTINUATION == true)
    if(loan_out.continuation_metrics) cudaMemset(loan_out.continuation_metrics,0,6*sizeof(unsigned long long));
#endif
    if (loan_out.generation) cudaMemset(loan_out.generation, 0, sizeof(int));
    if (loan_out.seed_count) cudaMemset(loan_out.seed_count, 0, sizeof(int));
    if (loan_out.result_count) cudaMemset(loan_out.result_count, 0, sizeof(int));
    if (loan_out.requeue_count) cudaMemset(loan_out.requeue_count, 0, sizeof(int));
#endif

#if (L3_RX_EXPRESS == true)
    if (rx_express.sequence)
    {
        unsigned long long express_sequence_init[L3_RX_EXPRESS_SLOTS];
        for (int i = 0; i < L3_RX_EXPRESS_SLOTS; i++)
            express_sequence_init[i] = (unsigned long long)i;
        cudaMemcpy(rx_express.sequence, express_sequence_init,
                   sizeof(express_sequence_init), cudaMemcpyHostToDevice);
    }
    if (rx_express.count)
        cudaMemset(rx_express.count, 0, sizeof(int) * L3_RX_EXPRESS_SLOTS);
    if (rx_express.head)
        cudaMemset(rx_express.head, 0, sizeof(unsigned long long));
    if (rx_express.tail)
        cudaMemset(rx_express.tail, 0, sizeof(unsigned long long));
#if (L3_RX_EXPRESS_DIAG == true)
    if (rx_express.stats)
        cudaMemset(rx_express.stats, 0,
                   sizeof(unsigned long long) * L3_RX_EXPRESS_STAT_COUNT);
#endif
#endif

#if (L3_RX_L2_PULL == true)
    if (rx_commit_seq)
        cudaMemset(rx_commit_seq, 0, sizeof(unsigned));
#if (L3_RX_L2_PULL_SINGLE_CLAIM == true)
    if (rx_pull_claim_seq)
        cudaMemset(rx_pull_claim_seq, 0, sizeof(unsigned));
#endif
#if (L3_RX_L2_PULL_DIAG == true)
    if (rx_l2_pull_stats)
        cudaMemset(rx_l2_pull_stats, 0,
                   sizeof(unsigned long long) * L3_RX_L2_PULL_STATS_COUNT);
#endif
#endif

#if (BULK_ROUND == true)
    if (bulk_quiesce_req) cudaMemset(bulk_quiesce_req, 0, sizeof(int));
    if (bulk_quiesce_ack) cudaMemset(bulk_quiesce_ack, 0, sizeof(int));
    if (l3_term_req) cudaMemset(l3_term_req, 0, sizeof(int));
    if (l3_term_state) cudaMemset(l3_term_state, 0, sizeof(int));
#if (L3_ADMISSION_BUDGET == true)
    const int advice_reset[4]={0,0,DIST_MAX,0};
    g_benchmark.require(cudaMemset(admission_state,0,2*sizeof(int))==cudaSuccess,"admission state reset");
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_admission,advice_reset,sizeof(advice_reset))==cudaSuccess,"admission reset");
#endif
    if (l3_term_ack_slots)
        cudaMemset(l3_term_ack_slots, 0, sizeof(int) * l3_term_ack_capacity);
    if (bulk_inbox_count)
        cudaMemset(bulk_inbox_count, 0, sizeof(int) * BULK_INBOX_SLOTS);
#if (L3_RX_FEEDBACK_MODE > 0)
    if (bulk_feedback)
        cudaMemset(bulk_feedback, 0, sizeof(l3_feedback_record) * BULK_INBOX_SLOTS);
#endif
    if (bulk_inbox_epoch)
        cudaMemset(bulk_inbox_epoch, 0, sizeof(int) * BULK_INBOX_SLOTS);
    if (bulk_inbox_ack)
        cudaMemset(bulk_inbox_ack, 0, sizeof(int) * BULK_INBOX_SLOTS);
    if (bulk_inbox_state)
        cudaMemset(bulk_inbox_state, 0, sizeof(int) * BULK_INBOX_SLOTS);
    if (bulk_inbox_generation)
        cudaMemset(bulk_inbox_generation, 0, sizeof(int) * BULK_INBOX_SLOTS);
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    if (bulk_inbox_read_head)
        cudaMemset(bulk_inbox_read_head, 0, sizeof(int) * BULK_INBOX_SLOTS);
    if (bulk_inbox_inflight)
        cudaMemset(bulk_inbox_inflight, 0, sizeof(int) * BULK_INBOX_SLOTS);
    if (bulk_inbox_active_slot)
    {
        int no_slot = -1;
        cudaMemcpy(bulk_inbox_active_slot, &no_slot, sizeof(int),
                   cudaMemcpyHostToDevice);
    }
#endif
#if (BULK_FRONTIER_ENABLED == true)
    if (bulk_frontier_head) cudaMemset(bulk_frontier_head, 0, sizeof(int));
    if (bulk_frontier_tail) cudaMemset(bulk_frontier_tail, 0, sizeof(int));
#endif
#endif

    if (remote_cand)
    {
#if (GLOBAL_ROUND_ASYNC == true)
        int bank_mark_words = (peer_v_local + 31) / 32;
        int bank_mh_words = (bank_mark_words + 31) / 32;
        for (int b = 0; b < 2; b++)
        {
            if (cand_bank[b].cand)
                g_benchmark.require(l3_fill_values(cand_bank[b].cand,
                    size_t(peer_v_local) + 1, init_max) == cudaSuccess,
                    "L3 candidate bank reset launch");
            if (cand_bank[b].mark)
                cudaMemset(cand_bank[b].mark, 0,
                           sizeof(unsigned) * bank_mark_words);
            if (cand_bank[b].hint)
                cudaMemset(cand_bank[b].hint, 0,
                           sizeof(unsigned) * bank_mh_words);
            if (cand_bank[b].hint2)
                cudaMemset(cand_bank[b].hint2, 0,
                           sizeof(unsigned) * ((bank_mh_words + 31) / 32));
            if (cand_bank[b].pending)
                cudaMemset(cand_bank[b].pending, 0, sizeof(int));
        }
        int active_init = 0;
        int state_init[2] = {ASYNC_CAND_BANK_ACTIVE, ASYNC_CAND_BANK_FREE};
        int users_init[2] = {0, 0};
        cudaMemcpy(cand_ctl.active_bank, &active_init, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cand_ctl.state, state_init, sizeof(state_init), cudaMemcpyHostToDevice);
        cudaMemcpy(cand_ctl.users, users_init, sizeof(users_init), cudaMemcpyHostToDevice);
#endif
        g_benchmark.require(l3_fill_values(remote_cand, size_t(peer_v_local) + 1,
                            init_max, peer_cache) == cudaSuccess,
                            "L3 candidate/cache reset launch");
        int mark_words = (peer_v_local + 31) / 32;
        int mh_words = (mark_words + 31) / 32;
        cudaMemset(remote_mark, 0, sizeof(unsigned) * mark_words);
        cudaMemset(mark_hint, 0, sizeof(unsigned) * mh_words);
        cudaMemset(mark_hint2, 0,
                   sizeof(unsigned) * ((mh_words + 31) / 32));
    }
    return 0;
}

void l3_host_context::release()
{
    cudaSetDevice(owner_gpu_id);
#define L3_FREE_OWNED(ptr) do { if (ptr) { cudaFree(ptr); ptr = NULL; } } while (0)
    L3_FREE_OWNED(dirty_bitmap);
    L3_FREE_OWNED(dirty_hint);
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
    L3_FREE_OWNED(async_rx_ready_bitmap);
    L3_FREE_OWNED(async_rx_ready_hint);
#endif
#if (L3_WORK_DIAG == true)
    L3_FREE_OWNED(source_expand_counts);
#endif
    L3_FREE_OWNED(last_processed);
#if (L3_COMPLETED_ROWS == true)
    VALUE_TYPE *no_completed_rows=nullptr;
    cudaMemcpyToSymbol(g_l3_completed_rows,&no_completed_rows,sizeof(no_completed_rows));
    L3_FREE_OWNED(continuation_completed);
#endif
    L3_FREE_OWNED(local_idle);
    L3_FREE_OWNED(remote_eff);
#if (GLOBAL_ROUND_ASYNC == true)
    remote_cand = NULL;
    remote_mark = NULL;
    mark_hint = NULL;
    mark_hint2 = NULL;
    for (int b = 0; b < 2; b++)
    {
        L3_FREE_OWNED(cand_bank[b].cand);
        L3_FREE_OWNED(cand_bank[b].mark);
        L3_FREE_OWNED(cand_bank[b].hint);
        L3_FREE_OWNED(cand_bank[b].hint2);
        L3_FREE_OWNED(cand_bank[b].pending);
    }
    L3_FREE_OWNED(cand_ctl.active_bank);
    L3_FREE_OWNED(cand_ctl.state);
    L3_FREE_OWNED(cand_ctl.users);
#else
    L3_FREE_OWNED(remote_cand);
    L3_FREE_OWNED(remote_mark);
    L3_FREE_OWNED(mark_hint);
    L3_FREE_OWNED(mark_hint2);
#endif
    L3_FREE_OWNED(peer_cache);
#if (L3_TILE_LOAN == true)
    L3_FREE_OWNED(loan_out.state);
#if (L3_CONTINUATION == true)
    L3_FREE_OWNED(loan_out.continuation_metrics);
#endif
    L3_FREE_OWNED(loan_out.generation);
    L3_FREE_OWNED(loan_out.seed_count);
    L3_FREE_OWNED(loan_out.result_count);
    L3_FREE_OWNED(loan_out.requeue_count);
    L3_FREE_OWNED(loan_out.tasks);
    L3_FREE_OWNED(loan_out.task_dists);
    L3_FREE_OWNED(loan_return_results);
    L3_FREE_OWNED(loan_return_dists);
    L3_FREE_OWNED(loan_ready);
#if (L3_MULTI_PRODUCER == true)
    L3_FREE_OWNED(loan_gate);
    L3_FREE_OWNED(loan_producers);
    peer_loan_gate=nullptr;
#endif
    loan_out = {};
    loan_in = {};
    peer_row_ptr = NULL;
    peer_col_idx = NULL;
    peer_edge_data = NULL;
#endif
#if (BULK_ROUND == true)
    L3_FREE_OWNED(bulk_send_list);
    L3_FREE_OWNED(bulk_inbox);
    L3_FREE_OWNED(bulk_inbox_count);
#if (L3_RX_FEEDBACK_MODE > 0)
    L3_FREE_OWNED(bulk_feedback);
#endif
    L3_FREE_OWNED(bulk_inbox_epoch);
    L3_FREE_OWNED(bulk_inbox_ack);
    L3_FREE_OWNED(bulk_inbox_state);
    L3_FREE_OWNED(bulk_inbox_generation);
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    L3_FREE_OWNED(bulk_inbox_read_head);
    L3_FREE_OWNED(bulk_inbox_inflight);
    L3_FREE_OWNED(bulk_inbox_active_slot);
#endif
    L3_FREE_OWNED(bulk_quiesce_req);
    L3_FREE_OWNED(bulk_quiesce_ack);
    L3_FREE_OWNED(l3_term_req);
    L3_FREE_OWNED(l3_term_state);
#if (L3_ADMISSION_BUDGET == true)
    L3_FREE_OWNED(admission_state);
#endif
    L3_FREE_OWNED(l3_term_ack_slots);
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
    L3_FREE_OWNED(bulk_dense_inbox);
#endif
#if (BULK_FRONTIER_ENABLED == true)
    L3_FREE_OWNED(bulk_frontier);
    L3_FREE_OWNED(bulk_frontier_head);
    L3_FREE_OWNED(bulk_frontier_tail);
#endif
#endif
#if (L3_RX_EXPRESS == true)
    L3_FREE_OWNED(rx_express.payload);
    L3_FREE_OWNED(rx_express.count);
    L3_FREE_OWNED(rx_express.sequence);
    L3_FREE_OWNED(rx_express.head);
    L3_FREE_OWNED(rx_express.tail);
#if (L3_RX_EXPRESS_DIAG == true)
    L3_FREE_OWNED(rx_express.stats);
#endif
#endif
#if (L3_RX_L2_PULL == true)
    L3_FREE_OWNED(rx_commit_seq);
#if (L3_RX_L2_PULL_SINGLE_CLAIM == true)
    L3_FREE_OWNED(rx_pull_claim_seq);
#endif
#if (L3_RX_L2_PULL_DIAG == true)
    L3_FREE_OWNED(rx_l2_pull_stats);
#endif
#endif
#undef L3_FREE_OWNED
    disable();
    local_v = 0;
#if (BULK_ROUND == true)
    l3_term_ack_capacity = 0;
#endif
}

l3_channel_view l3_host_context::make_channel_view() const
{
    l3_channel_view view = {};
    view.peer_id = enabled ? peer_id : -1;
    view.peer_v_begin = peer_v_begin;
    view.peer_v_local = peer_v_local;
    view.peer_node_data = peer_node_data;
    view.peer_dirty_bitmap = peer_dirty_bitmap;
    view.peer_dirty_hint = peer_dirty_hint;
    view.peer_local_idle = peer_local_idle;
    view.candidate_values = remote_cand;
    view.candidate_mark = remote_mark;
    view.candidate_hint = mark_hint;
    view.candidate_hint2 = mark_hint2;
    view.peer_cache = peer_cache;
    view.peer_cache_feedback = peer_cache_feedback;
    view.remote_eff = remote_eff;
    view.peer_remote_eff = peer_remote_eff;
#if (L3_TILE_LOAN == true)
    if (enabled)
    {
        view.loan_out = loan_out;
        view.loan_in = loan_in;
#if (L3_MULTI_PRODUCER == true)
        view.loan_gate=loan_gate; view.peer_loan_gate=peer_loan_gate;
        view.loan_epoch=loan_epoch; view.loan_producers=loan_producers;
#endif
        view.peer_row_ptr = peer_row_ptr;
        view.peer_col_idx = peer_col_idx;
        view.peer_edge_data = peer_edge_data;
    }
#endif
#if (BULK_ROUND == true)
    view.quiesce_req = bulk_quiesce_req;
    view.quiesce_ack = bulk_quiesce_ack;
    view.term_req = l3_term_req;
    view.term_state = l3_term_state;
    view.term_ack_slots = l3_term_ack_slots;
    view.peer_term_state = peer_l3_term_state;
#if (L3_ADMISSION_BUDGET == true)
    view.admission=admission_state;
    view.peer_admission=peer_admission_symbol;
#endif
    view.send_list = bulk_send_list;
    view.rx_payload = bulk_inbox;
    view.rx_count = bulk_inbox_count;
#if (L3_RX_FEEDBACK_MODE > 0)
    view.rx_feedback = bulk_feedback;
    view.tx_feedback = peer_bulk_feedback;
#endif
    view.rx_epoch = bulk_inbox_epoch;
    view.rx_ack = bulk_inbox_ack;
    view.rx_state = bulk_inbox_state;
    view.rx_generation = bulk_inbox_generation;
    view.tx_payload = peer_bulk_inbox;
    view.tx_count = peer_bulk_inbox_count;
    view.tx_epoch = peer_bulk_inbox_epoch;
    view.tx_ack = peer_bulk_inbox_ack;
    view.tx_state = peer_bulk_inbox_state;
    view.tx_generation = peer_bulk_inbox_generation;
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    view.rx_read_head = bulk_inbox_read_head;
    view.rx_inflight = bulk_inbox_inflight;
    view.rx_active_slot = bulk_inbox_active_slot;
#endif
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
    view.dense_rx_payload = bulk_dense_inbox;
    view.dense_tx_payload = peer_bulk_dense_inbox;
#endif
#if (BULK_FRONTIER_ENABLED == true)
    view.rx_frontier = bulk_frontier;
    view.rx_frontier_head = bulk_frontier_head;
    view.rx_frontier_tail = bulk_frontier_tail;
#endif
#endif
#if (L3_RX_EXPRESS == true)
    view.rx_express = rx_express;
#endif
#if (L3_RX_L2_PULL == true)
    view.rx_commit_seq = rx_commit_seq;
#if (L3_RX_L2_PULL_DIAG == true)
    view.rx_l2_pull_stats = rx_l2_pull_stats;
#endif
#endif
    return view;
}

int g_n_gpu = 1;
#if (L3_TILE_LOAN == true)
int g_l3_tile_loan_enabled = 1;
#endif
// E2 实验: 命令行 -d 覆盖 delta（-1=不覆盖，>0=覆盖 s_l2_delta）
int g_delta_override = -1;
// 队列自适应（SESSION 31）: main.cu 按图特征选队列（-1=默认 init_setup；>=0 覆盖 type）
int g_queue_override = -1;
// 多源 batch 计时默认关闭每个 query 的诊断打印，避免多个 GPU worker 的
// stdout 锁竞争污染吞吐测量；kernel 错误仍由 launch error 路径打印。
int g_multi_source_quiet = 0;
// Set once before query threads start; never changed by a running query.
int g_rx_express_enabled = 0;
#if (L3_RX_L2_PULL == true)
int g_rx_l2_pull_enabled = 0;
#endif
#if (L3_FAULT_INJECT_CLAIM_RETRY == true)
// 每个 CUDA device/module 各有一份；每次 query 在 sssp_re_init 中清零，
// 由接收端 claim helper 原子消费一次注入令牌。
__device__ unsigned int g_l3_fault_claim_retry = 0;
#endif
#if (L3_FAULT_INJECT_READY_DELAY == true)
// 每个 CUDA device/module 各有一份；每次 query 在 sssp_re_init 中清零，
// 由接收端 ready helper 原子消费一次注入令牌。
__device__ unsigned int g_l3_fault_ready_delay = 0;
#endif
#if (L3_FAULT_INJECT_ACK_DELAY == true)
// 每个 CUDA device/module 各有一份；每次 query 在 sssp_re_init 中清零。
// ack_delay 是注入计数，ack_pending 保存尚未重发的 epoch。
__device__ unsigned int g_l3_fault_ack_delay = 0;
__device__ int g_l3_fault_ack_pending = 0;
#endif
#if (L3_EVENT_RING == true)
// CP1c：每个 CUDA device/module 各有一份固定容量事件环；host 在 query
// 完成后读取，生产构建不分配也不触发这些符号。
__device__ l3_event_entry g_l3_event_ring[L3_EVENT_RING_CAP];
__device__ unsigned int g_l3_event_head = 0;
__device__ unsigned int g_l3_event_overflow = 0;
#endif

#if (L3_LIVE_SNAPSHOT == true)
// CP1d：把 persistent kernel 内只存在于寄存器/shared memory 的控制流状态
// 镜像到一个小型 device 全局结构，供 host watchdog 只读采样。该结构只存在于
// 显式诊断构建，默认构建没有写热路径或存储开销。
struct l3_live_device_state
{
    unsigned long long manager_iter;
    unsigned long long l3_iter;
    unsigned long long inject_iter[INJECT_WARP_NUM];
    int manager_stage;
    int manager_rx_epoch;
    int manager_node_in;
    int manager_l2_size;
    int bs_req;
    int bs_done;
    int bs_found;
    int l3_stage;
    int l3_tx_epoch;
    int l3_count;
    int inject_stage[INJECT_WARP_NUM];
    int manager_lane_stage[WARP_SIZE];
    int manager_lane_l2_empty[WARP_SIZE];
    unsigned long long manager_lane_iter[WARP_SIZE];
};
__device__ l3_live_device_state g_l3_live_state = {};

__device__ __forceinline__ void l3_live_manager_state(
    int stage, int rx_epoch, int node_in, int l2_size,
    volatile int *bs_req, int *bs_done, int *bs_found)
{
    atomicExch(&g_l3_live_state.manager_stage, stage);
    atomicExch(&g_l3_live_state.manager_rx_epoch, rx_epoch);
    atomicExch(&g_l3_live_state.manager_node_in, node_in);
    atomicExch(&g_l3_live_state.manager_l2_size, l2_size);
    atomicExch(&g_l3_live_state.bs_req, *(volatile int *)bs_req);
    atomicExch(&g_l3_live_state.bs_done, atomicAdd(bs_done, 0));
    atomicExch(&g_l3_live_state.bs_found, atomicAdd(bs_found, 0));
}

__device__ __forceinline__ void l3_live_backstop_state(
    int stage, volatile int *bs_req, int *bs_done, int *bs_found)
{
    atomicExch(&g_l3_live_state.manager_stage, stage);
    atomicExch(&g_l3_live_state.bs_req, *(volatile int *)bs_req);
    atomicExch(&g_l3_live_state.bs_done, atomicAdd(bs_done, 0));
    atomicExch(&g_l3_live_state.bs_found, atomicAdd(bs_found, 0));
}

__device__ __forceinline__ void l3_live_l3_state(
    int stage, int tx_epoch, int count)
{
    atomicExch(&g_l3_live_state.l3_stage, stage);
    atomicExch(&g_l3_live_state.l3_tx_epoch, tx_epoch);
    atomicExch(&g_l3_live_state.l3_count, count);
}

__device__ __forceinline__ void l3_live_inject_state(
    int inject_id, int stage)
{
    if (inject_id < 0 || inject_id >= INJECT_WARP_NUM)
        return;
    atomicExch(&g_l3_live_state.inject_stage[inject_id], stage);
}

__device__ __forceinline__ void l3_live_manager_lane_state(
    int lane_id, int stage, int l2_empty)
{
    atomicExch(&g_l3_live_state.manager_lane_stage[lane_id], stage);
    atomicExch(&g_l3_live_state.manager_lane_l2_empty[lane_id], l2_empty);
}
#endif

#if (L3_EVENT_RING == true)
static const char *l3_event_code_name(int code)
{
    switch (code)
    {
    case L3_EVENT_SCAN_COALESCED: return "scan_coalesced";
    case L3_EVENT_PACK_BEGIN: return "pack_begin";
    case L3_EVENT_TX_PUBLISHED: return "tx_published";
    case L3_EVENT_TX_RETRY: return "tx_retry";
    case L3_EVENT_RX_CLAIM: return "rx_claim";
    case L3_EVENT_RX_APPLY: return "rx_apply";
    case L3_EVENT_TERM_REQUEST: return "term_request";
    case L3_EVENT_TERM_CANCEL: return "term_cancel";
    case L3_EVENT_TERM_READY: return "term_ready";
    case L3_EVENT_TERM_EXIT: return "term_exit";
    default: return "unknown";
    }
}

static void l3_event_dump_host(int gpu_id)
{
    unsigned int head = 0;
    unsigned int overflow = 0;
    cudaError_t err = cudaMemcpyFromSymbol(
        &head, g_l3_event_head, sizeof(head), 0, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("gpu%d L3_EVENT_RING read head failed: %s\n",
               gpu_id, cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpyFromSymbol(
        &overflow, g_l3_event_overflow, sizeof(overflow), 0,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("gpu%d L3_EVENT_RING read overflow failed: %s\n",
               gpu_id, cudaGetErrorString(err));
        return;
    }

    std::vector<l3_event_entry> entries(L3_EVENT_RING_CAP);
    err = cudaMemcpyFromSymbol(
        entries.data(), g_l3_event_ring,
        sizeof(l3_event_entry) * L3_EVENT_RING_CAP, 0,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("gpu%d L3_EVENT_RING read entries failed: %s\n",
               gpu_id, cudaGetErrorString(err));
        return;
    }

    unsigned int first = head > L3_EVENT_RING_CAP
                        ? head - L3_EVENT_RING_CAP : 0;
    printf("gpu%d L3_EVENT_RING head=%u overflow=%u kept=%u\n",
           gpu_id, head, overflow, head - first);
    for (unsigned int seq = first; seq < head; ++seq)
    {
        const l3_event_entry &entry = entries[seq % L3_EVENT_RING_CAP];
        printf("gpu%d L3_EVENT seq=%u tick=%llu code=%d(%s) v=%d,%d,%d,%d\n",
               gpu_id, seq, (unsigned long long)entry.tick, entry.code,
               l3_event_code_name(entry.code), entry.value0, entry.value1,
               entry.value2, entry.value3);
    }
}
#endif

#define work_count_type unsigned long long

// 每种编译期 queue 类型、每张物理 GPU 各保留一个 query workspace。
// 多源 batch 中同一张卡上的 source worker 是串行的，因此 workspace 不需要
// 在 GPU 内并发复用锁；source 之间只执行现有 re_init/reset。这样 queue 的
// 大块 device allocation 和 query 临时 buffer 不再随 source 数累积。
template <typename QUEUE_TYPE>
struct query_workspace
{
    bool initialized;
    int v_local;
    QUEUE_TYPE mlmq;

    work_count_type *global_work_count;
    int *global_comp_count;
    int *profile;
    unsigned int *hist;

#if (MANAGE_PROFILE == true)
    unsigned long long *mgmt_profile;
#endif

#if (L3_TIMING_DIAG == true)
    cudaEvent_t timing_manage_start, timing_manage_end;
    cudaEvent_t timing_work_start, timing_work_end;
#endif
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaStream_t work_stream;
    cudaStream_t manage_stream;

#if (QUERY_WORKSPACE_DIAG == true)
    size_t initial_free_bytes;
    size_t min_free_bytes;
    size_t max_free_bytes;
    unsigned long long reset_count;
#endif

    bool ensure(int gpu_id, int v_local_in, NODE_TYPE init_limits,
                mlmq_setup setup)
    {
        cudaSetDevice(gpu_id);

        if (initialized)
        {
            if (v_local != v_local_in)
            {
                printf("gpu%d query workspace graph size changed: %d -> %d\n",
                       gpu_id, v_local, v_local_in);
                return false;
            }
            return true;
        }

        if (mlmq.init_host(GPU_MEMORY, init_limits, setup) != INIT_SUCCESS)
        {
            printf("gpu%d query workspace MLMQ initialization failed\n", gpu_id);
            return false;
        }

        v_local = v_local_in;

#if (WORK_COUNT == true)
        cudaMalloc(&global_work_count, sizeof(work_count_type));
        cudaMalloc(&global_comp_count, sizeof(int));
        cudaMalloc(&profile, 2 * sizeof(int));
        // v_local+1 保留原有越界保护：dst_v-v_begin 可能等于 v_local。
        cudaMalloc(&hist, (v_local + 1) * sizeof(unsigned int));
#endif

#if (MANAGE_PROFILE == true)
        cudaMalloc(&mgmt_profile, sizeof(unsigned long long) * 13);
#endif

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
#if (L3_TIMING_DIAG == true)
        g_benchmark.require(cudaEventCreate(&timing_manage_start) == cudaSuccess, "timing manager start create");
        g_benchmark.require(cudaEventCreate(&timing_manage_end) == cudaSuccess, "timing manager end create");
        g_benchmark.require(cudaEventCreate(&timing_work_start) == cudaSuccess, "timing work start create");
        g_benchmark.require(cudaEventCreate(&timing_work_end) == cudaSuccess, "timing work end create");
#endif
        cudaStreamCreateWithFlags(&work_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&manage_stream, cudaStreamNonBlocking);

#if (QUERY_WORKSPACE_DIAG == true)
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        initial_free_bytes = free_bytes;
        min_free_bytes = free_bytes;
        max_free_bytes = free_bytes;
        reset_count = 0;
#endif

        initialized = true;
        return true;
    }
};

template <typename QUEUE_TYPE>
query_workspace<QUEUE_TYPE> g_query_workspace[MAX_GPU];

#if (L3_TIMING_DIAG == true)
// Host records have one writer per GPU and are read only after all query threads
// join. No diagnostic printing/event readback occurs inside solve/query timing.
struct l3_host_timing {
    double barrier_return, manage_before, manage_after, work_before, work_after;
    double sync_before, sync_after, finish_after;
    cudaEvent_t manage_start, manage_end, work_start, work_end;
};
static l3_host_timing g_l3_host_timing[MAX_GPU];
void sssp_report_timing(int n_gpu, int sample, bool warmup) {
    for (int gpu = 0; gpu < n_gpu; ++gpu) {
        const auto &t = g_l3_host_timing[gpu];
        float manage_ms = 0, work_ms = 0;
        g_benchmark.require(cudaSetDevice(gpu) == cudaSuccess, "timing report device");
        g_benchmark.require(cudaEventElapsedTime(&manage_ms, t.manage_start, t.manage_end) == cudaSuccess, "timing manager elapsed");
        g_benchmark.require(cudaEventElapsedTime(&work_ms, t.work_start, t.work_end) == cudaSuccess, "timing work elapsed");
        const double base = g_benchmark.start;
        printf("L3_TIMING gpu=%d repeat=%d warmup=%d barrier_return_ms=%.6f "
               "manage_before_ms=%.6f manage_after_ms=%.6f work_before_ms=%.6f "
               "work_after_ms=%.6f sync_before_ms=%.6f sync_after_ms=%.6f "
               "finish_after_ms=%.6f manage_stream_ms=%.6f work_stream_ms=%.6f\n",
               gpu, sample, int(warmup), t.barrier_return-base,
               t.manage_before-base, t.manage_after-base, t.work_before-base,
               t.work_after-base, t.sync_before-base, t.sync_after-base,
               t.finish_after-base, double(manage_ms), double(work_ms));
    }
}
#endif

// device node data ptr on device (单卡模式 / 本地访问)
VALUE_TYPE *node_data;
__device__ VALUE_TYPE *node_data_dev;
// indicate exiting signals;
int *global_exit;

// B1-1 诊断: 本地/远程改进计数拆分（device 全局，每卡独立，atomicAdd 累加）
__device__ work_count_type g_local_work = 0;
__device__ work_count_type g_remote_work = 0;
// B1-2a 诊断: per-vertex 本地改进计数直方图（区分"均匀乱序 vs 边界集中"）
__device__ unsigned int *g_hist = nullptr;
// E4 诊断: w17 三级扫下钻计数（区分真候选 vs mark_hint 假阳下钻）
__device__ unsigned long long g_h2_hit = 0;   // mark_hint2 外层命中（非零 word）
__device__ unsigned long long g_w_scan = 0;   // 内层 remote_mark 实际扫描 word 数
__device__ unsigned long long g_w_hit = 0;    // 内层 remote_mark 真命中（有候选）word 数
// E1 前置验证: 跨卡 flush 真有效改进计数（atomicMin 返回 < old 即 peer 真正更新）
__device__ unsigned long long g_remote_effective = 0;
// E-3 诊断: delta 桶倒退距离 clamp 计数（extern 声明于 cu_delta_queue.cuh）
#if (DQ_CLAMP_DIAG == true)
__device__ unsigned long long g_dq_clamp = 0;
#endif
#if (DQ_QUEUE_DIAG == true)
__device__ dq_queue_diag_metrics g_dq_queue_diag = {};
#endif
#if (DQ_SPARSE_DIAG == true)
__device__ dq_sparse_metrics g_dq_sparse[DQ_SPARSE_SLOTS];
#if (DQ_SPARSE_PHASE == true)
__device__ dq_phase_metrics g_dq_phase[DQ_SPARSE_SLOTS];
#endif
#endif
// SEED_EXP: 边界最终值种子（device 符号，sssp_set_seeds 由 host 填充，work kernel 初始化注入）
#if (SEED_EXP == true)
__device__ int g_seed_num = 0;
__device__ int *g_seed_ids = nullptr;            // 全局 1-based 顶点 id
__device__ VALUE_TYPE *g_seed_dists = nullptr;   // 种子最终距离
#endif
// SEED_BARRIER 诊断: seed_ready 置位次数（源卡原子加，打印确认屏障触发）
__device__ unsigned long long g_seed_ready_fired = 0;
// SEED_BARRIER 诊断: 接收卡 gate 跳过次数（注入 warp） + 接收卡 backstop skip 次数
__device__ unsigned long long g_inj_gate_skip = 0;
__device__ unsigned long long g_bs_gate_skip = 0;
// SEED_BARRIER 诊断: 关键事件时间戳（clock 周期，打印换算 ms）
__device__ timeline_clock_type g_t_phase2 = 0;   // 源卡 phase 1→2
__device__ timeline_clock_type g_t_seed_ready = 0;   // 源卡置 seed_ready
__device__ timeline_clock_type g_t_sr_seen = 0;   // 接收卡首见 seed_ready
__device__ timeline_clock_type g_t_inject_done = 0;  // 接收卡注入完成
__device__ timeline_clock_type g_t_idle = 0;         // 本卡置 local_idle（首见 peer idle 前）
__device__ timeline_clock_type g_t_term = 0;         // 本卡终止（manager_end）
__device__ timeline_clock_type g_t_peer_idle_seen = 0;  // 首见 peer_idle==1
__device__ timeline_clock_type g_t_sr_w0 = 0;           // warp0 首见 seed_ready
#if (TIMELINE64 == true)
__device__ timeline_clock_type g_t_idle_last_set = 0;
__device__ timeline_clock_type g_t_idle_last_reset = 0;
__device__ unsigned long long g_idle_set_count = 0;
__device__ unsigned long long g_idle_reset_count = 0;
__device__ unsigned long long g_idle_reset_reason[16] = {};
__device__ unsigned long long g_term_mark_actual = 0;
__device__ unsigned long long g_term_mark_stale = 0;
__device__ unsigned long long g_term_mark_sampled = 0;
#endif
__device__ unsigned long long g_confirm_exec = 0;  // term_ok 二次确认执行次数
#if (WORK_SEG_PROFILE == true)
__device__ unsigned long long g_wk_active_clks = 0;  // work kernel 活跃处理时钟（warp0 lane0 累积）
#endif
// SEED_BARRIER 诊断: 注入块执行次数 + 时钟耗时
__device__ unsigned long long g_inject_exec = 0;
__device__ unsigned long long g_inject_clks = 0;
__device__ unsigned int g_inject_scan_clks = 0;
__device__ unsigned int g_inject_wt_clks = 0;
#if (BULK_DIAG == true)
__device__ unsigned long long g_bulk_work_tick = 0;
__device__ unsigned long long g_bulk_dq_prints = 0;
__device__ unsigned long long g_bulk_inj_prints = 0;
__device__ unsigned long long g_bulk_quiesce_diag_prints = 0;
__device__ unsigned long long g_bulk_frontier_diag_prints = 0;
#endif
#if (BULK_TRACE == true)
__device__ unsigned long long g_bulk_trace_relax = 0;
__device__ unsigned long long g_bulk_trace_cand = 0;
__device__ unsigned long long g_bulk_trace_tx = 0;
__device__ unsigned long long g_bulk_trace_rx = 0;
#endif
#if (BULK_ROUND == true && SEED_BARRIER == false)
// 通用 BULK：remote_mark 的生产事件和本卡 idle 唤醒信号。
// signal 供 L3 quiet 状态低成本唤醒；local_idle 供对端终止握手立即失效。
__device__ unsigned long long g_bulk_mark_signal = 0;
__device__ int *g_bulk_local_idle = nullptr;
#endif
#if (GLOBAL_ROUND_STATS == true)
// GLOBAL_ROUND 低开销诊断：每卡只在完成一轮时更新一次，禁止 device printf。
__device__ unsigned long long g_global_round_count = 0;
__device__ unsigned long long g_global_round_sent = 0;
__device__ unsigned long long g_global_round_recv = 0;
__device__ unsigned long long g_global_round_improved = 0;
__device__ unsigned long long g_global_round_frontier_append = 0;
#endif
#if (GLOBAL_ROUND_PROFILE == true)
// GLOBAL_ROUND 阶段计时：仅由每卡 manager warp0 lane0 累加，使用 clock64()。
__device__ unsigned long long g_gr_local_empty = 0;
__device__ unsigned long long g_gr_quiesce = 0;
__device__ unsigned long long g_gr_pack = 0;
__device__ unsigned long long g_gr_inbox_wait = 0;
__device__ unsigned long long g_gr_apply = 0;
__device__ unsigned long long g_gr_release = 0;
__device__ unsigned long long g_gr_local_empty_max = 0;
__device__ unsigned long long g_gr_quiesce_max = 0;
__device__ unsigned long long g_gr_pack_max = 0;
__device__ unsigned long long g_gr_inbox_wait_max = 0;
__device__ unsigned long long g_gr_apply_max = 0;
__device__ unsigned long long g_gr_release_max = 0;
__device__ unsigned long long g_gr_profile_rounds = 0;
#endif
#if (GLOBAL_ROUND_DIRECT_TRACE == true)
// Direct P2P 诊断：只记录两个小图目标顶点的候选/发送/注入事件，避免
// 在 work 热路径 printf 导致 kernel 资源超限。
__device__ unsigned long long g_direct_relax_193 = 0;
__device__ unsigned long long g_direct_relax_362 = 0;
__device__ unsigned long long g_direct_cand_193 = 0;
__device__ unsigned long long g_direct_cand_362 = 0;
__device__ unsigned long long g_direct_tx_193 = 0;
__device__ unsigned long long g_direct_tx_362 = 0;
__device__ unsigned long long g_direct_inj_193 = 0;
__device__ unsigned long long g_direct_inj_362 = 0;
__device__ int g_direct_relax_193_min = 0x7fffffff;
__device__ int g_direct_relax_362_min = 0x7fffffff;
__device__ int g_direct_tx_193_min = 0x7fffffff;
__device__ int g_direct_tx_362_min = 0x7fffffff;
__device__ int g_direct_inj_193_min = 0x7fffffff;
__device__ int g_direct_inj_362_min = 0x7fffffff;
#endif

#if (L3_LIVE_SNAPSHOT == true)
// CP1d：Host 侧只读取既有协议字段，不写入任何 device 状态。数据目标使用
// cudaMallocHost 分配的 pinned memory，copy 通过独立 nonblocking stream 排队；
// watchdog 只做 stream query，不在 kernel 仍运行时调用 stream synchronize。
struct l3_live_snapshot_data
{
    unsigned long long sample;
    unsigned long long elapsed_us;
    int global_exit;
    int local_idle;
    int phase;
    int seed_ready;
    int seed_inject_done;
    int term_req;
    int term_state;
    int peer_term_state;
    int quiesce_req;
    int quiesce_ack;
    unsigned long long local_work;
    unsigned long long remote_work;
    unsigned long long inject_exec;
    unsigned long long confirm_exec;
    unsigned int event_head;
    unsigned int event_overflow;
#if (L3_EVENT_RING == true)
    l3_event_entry events[L3_EVENT_RING_CAP];
#endif
    l3_live_device_state device;
    int rx_count[BULK_INBOX_SLOTS];
    int rx_epoch[BULK_INBOX_SLOTS];
    int rx_ack[BULK_INBOX_SLOTS];
    int rx_state[BULK_INBOX_SLOTS];
    int rx_generation[BULK_INBOX_SLOTS];
    int tx_count[BULK_INBOX_SLOTS];
    int tx_epoch[BULK_INBOX_SLOTS];
    int tx_ack[BULK_INBOX_SLOTS];
    int tx_state[BULK_INBOX_SLOTS];
    int tx_generation[BULK_INBOX_SLOTS];
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    int rx_read_head[BULK_INBOX_SLOTS];
    int rx_inflight[BULK_INBOX_SLOTS];
    int rx_active_slot;
#endif
    int *term_ack_slots;
    int term_ack_capacity;
};

static void l3_live_snapshot_prepare(l3_live_snapshot_data &data)
{
    data.global_exit = -1;
    data.local_idle = -1;
    data.phase = -1;
    data.seed_ready = -1;
    data.seed_inject_done = -1;
    data.term_req = -1;
    data.term_state = -1;
    data.peer_term_state = -1;
    data.quiesce_req = -1;
    data.quiesce_ack = -1;
    data.local_work = 0;
    data.remote_work = 0;
    data.inject_exec = 0;
    data.confirm_exec = 0;
    memset(&data.device, 0, sizeof(data.device));
#if (L3_EVENT_RING == true)
    data.event_head = 0;
    data.event_overflow = 0;
#else
    data.event_head = static_cast<unsigned int>(-1);
    data.event_overflow = static_cast<unsigned int>(-1);
#endif
    for (int slot = 0; slot < BULK_INBOX_SLOTS; slot++)
    {
        data.rx_count[slot] = -1;
        data.rx_epoch[slot] = -1;
        data.rx_ack[slot] = -1;
        data.rx_state[slot] = -1;
        data.rx_generation[slot] = -1;
        data.tx_count[slot] = -1;
        data.tx_epoch[slot] = -1;
        data.tx_ack[slot] = -1;
        data.tx_state[slot] = -1;
        data.tx_generation[slot] = -1;
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
        data.rx_read_head[slot] = -1;
        data.rx_inflight[slot] = -1;
#endif
    }
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    data.rx_active_slot = -1;
#endif
    for (int i = 0; i < data.term_ack_capacity; i++)
        data.term_ack_slots[i] = -1;
}

static bool l3_live_check(cudaError_t err, int gpu_id, const char *field)
{
    if (err == cudaSuccess)
        return true;
    printf("gpu%d L3_LIVE enqueue %s failed: %s\n",
           gpu_id, field, cudaGetErrorString(err));
    fflush(stdout);
    return false;
}

template <typename T>
static bool l3_live_enqueue_copy(T *host_dst, const T *device_src,
                                 size_t count, cudaStream_t stream,
                                 int gpu_id, const char *field)
{
    if (device_src == NULL || count == 0)
        return true;
    return l3_live_check(
        cudaMemcpyAsync(host_dst, device_src, sizeof(T) * count,
                        cudaMemcpyDeviceToHost, stream),
        gpu_id, field);
}

static bool l3_live_snapshot_enqueue(int gpu_id, const gpu_ctx &ctx,
                                     l3_live_snapshot_data &data,
                                     cudaStream_t stream)
{
    l3_live_snapshot_prepare(data);
#define L3_LIVE_COPY_SCALAR(field, ptr) \
    if (!l3_live_enqueue_copy(&data.field, ptr, 1, stream, gpu_id, #field)) return false
#define L3_LIVE_COPY_ARRAY(field, ptr, count) \
    if (!l3_live_enqueue_copy(data.field, ptr, count, stream, gpu_id, #field)) return false

    L3_LIVE_COPY_SCALAR(global_exit, ctx.global_exit);
    L3_LIVE_COPY_SCALAR(local_idle, ctx.local_idle);
    L3_LIVE_COPY_SCALAR(phase, ctx.phase);
    L3_LIVE_COPY_SCALAR(seed_ready, ctx.seed_ready);
    L3_LIVE_COPY_SCALAR(seed_inject_done, ctx.seed_inject_done);
#if (BULK_ROUND == true)
    L3_LIVE_COPY_SCALAR(term_req, ctx.l3_term_req);
    L3_LIVE_COPY_SCALAR(term_state, ctx.l3_term_state);
    L3_LIVE_COPY_SCALAR(peer_term_state, ctx.peer_l3_term_state);
    L3_LIVE_COPY_SCALAR(quiesce_req, ctx.bulk_quiesce_req);
    L3_LIVE_COPY_SCALAR(quiesce_ack, ctx.bulk_quiesce_ack);
    L3_LIVE_COPY_ARRAY(rx_count, ctx.bulk_inbox_count, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_ARRAY(rx_epoch, ctx.bulk_inbox_epoch, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_ARRAY(rx_ack, ctx.bulk_inbox_ack, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_ARRAY(rx_state, ctx.bulk_inbox_state, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_ARRAY(rx_generation, ctx.bulk_inbox_generation, BULK_INBOX_SLOTS);
    // peer_* inbox 元数据是本卡发送侧正在观察的 outbox。启用 P2P/UVA 时，
    // 当前 device 的 D2H copy 可直接读取它；若平台不支持则显式报错。
    L3_LIVE_COPY_ARRAY(tx_count, ctx.peer_bulk_inbox_count, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_ARRAY(tx_epoch, ctx.peer_bulk_inbox_epoch, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_ARRAY(tx_ack, ctx.peer_bulk_inbox_ack, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_ARRAY(tx_state, ctx.peer_bulk_inbox_state, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_ARRAY(tx_generation, ctx.peer_bulk_inbox_generation, BULK_INBOX_SLOTS);
    if (ctx.l3_term_ack_slots != NULL && data.term_ack_capacity > 0)
    {
        if (!l3_live_check(
                cudaMemcpyAsync(data.term_ack_slots, ctx.l3_term_ack_slots,
                                sizeof(int) * data.term_ack_capacity,
                                cudaMemcpyDeviceToHost, stream),
                gpu_id, "term_ack_slots"))
            return false;
    }
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    L3_LIVE_COPY_ARRAY(rx_read_head, ctx.bulk_inbox_read_head, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_ARRAY(rx_inflight, ctx.bulk_inbox_inflight, BULK_INBOX_SLOTS);
    L3_LIVE_COPY_SCALAR(rx_active_slot, ctx.bulk_inbox_active_slot);
#endif
#endif

    if (!l3_live_check(
            cudaMemcpyFromSymbolAsync(&data.local_work, g_local_work,
                                      sizeof(data.local_work), 0,
                                      cudaMemcpyDeviceToHost, stream),
            gpu_id, "local_work")) return false;
    if (!l3_live_check(
            cudaMemcpyFromSymbolAsync(&data.remote_work, g_remote_work,
                                      sizeof(data.remote_work), 0,
                                      cudaMemcpyDeviceToHost, stream),
            gpu_id, "remote_work")) return false;
    if (!l3_live_check(
            cudaMemcpyFromSymbolAsync(&data.inject_exec, g_inject_exec,
                                      sizeof(data.inject_exec), 0,
                                      cudaMemcpyDeviceToHost, stream),
            gpu_id, "inject_exec")) return false;
    if (!l3_live_check(
            cudaMemcpyFromSymbolAsync(&data.confirm_exec, g_confirm_exec,
                                      sizeof(data.confirm_exec), 0,
                                      cudaMemcpyDeviceToHost, stream),
            gpu_id, "confirm_exec")) return false;
#if (L3_EVENT_RING == true)
    if (!l3_live_check(
            cudaMemcpyFromSymbolAsync(&data.event_head, g_l3_event_head,
                                      sizeof(data.event_head), 0,
                                      cudaMemcpyDeviceToHost, stream),
            gpu_id, "event_head")) return false;
    if (!l3_live_check(
            cudaMemcpyFromSymbolAsync(&data.event_overflow, g_l3_event_overflow,
                                      sizeof(data.event_overflow), 0,
                                      cudaMemcpyDeviceToHost, stream),
            gpu_id, "event_overflow")) return false;
    if (!l3_live_check(
            cudaMemcpyFromSymbolAsync(data.events, g_l3_event_ring,
                                      sizeof(data.events), 0,
                                      cudaMemcpyDeviceToHost, stream),
            gpu_id, "events")) return false;
#endif
    if (!l3_live_check(
            cudaMemcpyFromSymbolAsync(&data.device, g_l3_live_state,
                                      sizeof(data.device), 0,
                                      cudaMemcpyDeviceToHost, stream),
            gpu_id, "device_state")) return false;
#undef L3_LIVE_COPY_ARRAY
#undef L3_LIVE_COPY_SCALAR
    return true;
}

static void l3_live_snapshot_emit(int gpu_id,
                                  const l3_live_snapshot_data &data)
{
    int ack_match = 0;
    int ack_nonzero = 0;
    int ack_min = 0x7fffffff;
    int ack_max = -1;
    for (int i = 0; i < data.term_ack_capacity; i++)
    {
        int ack = data.term_ack_slots[i];
        if (ack >= 0)
        {
            if (ack != 0) ack_nonzero++;
            if (ack < ack_min) ack_min = ack;
            if (ack > ack_max) ack_max = ack;
            if (data.term_req > 0 && ack == data.term_req) ack_match++;
        }
    }
    if (ack_min == 0x7fffffff) ack_min = -1;
    printf("L3_LIVE gpu%d sample=%llu elapsed_ms=%.1f exit=%d idle=%d "
           "phase=%d seed_ready=%d seed_inject=%d term_req=%d term_state=%d "
           "peer_term=%d quiesce=%d/%d work=%llu remote=%llu inject=%llu "
           "confirm=%llu event=%u/%u ack=%d/%d/%d/%d/%d\n",
           gpu_id, data.sample, data.elapsed_us / 1000.0,
           data.global_exit, data.local_idle, data.phase, data.seed_ready,
           data.seed_inject_done, data.term_req, data.term_state,
           data.peer_term_state, data.quiesce_req, data.quiesce_ack,
           data.local_work, data.remote_work, data.inject_exec,
           data.confirm_exec, data.event_head, data.event_overflow,
           ack_match, data.term_ack_capacity, ack_nonzero, ack_min, ack_max);
    printf("L3_LIVE_CTRL gpu%d sample=%llu mgr=%d/%llu rx=%d node=%d q=%d "
           "bs=%d/%d/%d l3=%d/%llu tx=%d cnt=%d",
           gpu_id, data.sample,
           data.device.manager_stage, data.device.manager_iter,
           data.device.manager_rx_epoch, data.device.manager_node_in,
           data.device.manager_l2_size, data.device.bs_req,
           data.device.bs_done, data.device.bs_found,
           data.device.l3_stage, data.device.l3_iter,
           data.device.l3_tx_epoch, data.device.l3_count);
    for (int i = 0; i < INJECT_WARP_NUM; i++)
        printf(" inj%d=%d/%llu", i, data.device.inject_stage[i],
               data.device.inject_iter[i]);
    printf("\n");
    unsigned int lane_l2_empty_mask = 0;
    printf("L3_LIVE_LANES gpu%d sample=%llu", gpu_id, data.sample);
    for (int lane = 0; lane < WARP_SIZE; lane++)
    {
        if (data.device.manager_lane_l2_empty[lane] != 0)
            lane_l2_empty_mask |= 1u << lane;
        printf(" %d/%llu", data.device.manager_lane_stage[lane],
               data.device.manager_lane_iter[lane]);
    }
    printf(" l2mask=0x%08x\n", lane_l2_empty_mask);
#if (L3_EVENT_RING == true)
    unsigned int event_count = std::min(
        data.event_head, static_cast<unsigned int>(L3_EVENT_RING_CAP));
    unsigned int event_first = data.event_head - event_count;
    unsigned int tail_first = data.event_head > 4 ? data.event_head - 4 : 0;
    if (tail_first < event_first) tail_first = event_first;
    for (unsigned int seq = tail_first; seq < data.event_head; seq++)
    {
        const l3_event_entry &entry = data.events[seq % L3_EVENT_RING_CAP];
        printf("L3_LIVE_EVT gpu%d sample=%llu seq=%u tick=%llu code=%s "
               "v=%d/%d/%d/%d\n",
               gpu_id, data.sample, seq, entry.tick,
               l3_event_code_name(entry.code), entry.value0, entry.value1,
               entry.value2, entry.value3);
    }
#endif
    for (int slot = 0; slot < BULK_INBOX_SLOTS; slot++)
    {
        printf("L3_LIVE_SLOT gpu%d sample=%llu slot=%d "
               "rx_count=%d rx_epoch=%d rx_ack=%d rx_state=%d rx_gen=%d "
               "tx_count=%d tx_epoch=%d tx_ack=%d tx_state=%d tx_gen=%d\n",
               gpu_id, data.sample, slot,
               data.rx_count[slot], data.rx_epoch[slot], data.rx_ack[slot],
               data.rx_state[slot], data.rx_generation[slot],
               data.tx_count[slot], data.tx_epoch[slot], data.tx_ack[slot],
               data.tx_state[slot], data.tx_generation[slot]);
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
        printf("L3_LIVE_RX gpu%d sample=%llu slot=%d head=%d inflight=%d active=%d\n",
               gpu_id, data.sample, slot, data.rx_read_head[slot],
               data.rx_inflight[slot], data.rx_active_slot);
#endif
    }
    fflush(stdout);
}

static void l3_live_snapshot_watchdog(int gpu_id, const gpu_ctx &ctx,
                                      std::atomic<bool> &stop)
{
    cudaSetDevice(gpu_id);
    cudaStream_t stream = NULL;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess)
    {
        printf("gpu%d L3_LIVE stream create failed: %s\n",
               gpu_id, cudaGetErrorString(err));
        fflush(stdout);
        return;
    }

    l3_live_snapshot_data *data = NULL;
    err = cudaMallocHost((void **)&data, sizeof(*data));
    if (err != cudaSuccess)
    {
        printf("gpu%d L3_LIVE data alloc failed: %s\n",
               gpu_id, cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        fflush(stdout);
        return;
    }
    data->term_ack_capacity = ctx.l3_term_ack_capacity > 0
                            ? ctx.l3_term_ack_capacity : 0;
    data->term_ack_slots = NULL;
    if (data->term_ack_capacity > 0)
    {
        err = cudaMallocHost((void **)&data->term_ack_slots,
                             sizeof(int) * data->term_ack_capacity);
        if (err != cudaSuccess)
        {
            printf("gpu%d L3_LIVE ack alloc failed: %s\n",
                   gpu_id, cudaGetErrorString(err));
            cudaFreeHost(data);
            cudaStreamDestroy(stream);
            fflush(stdout);
            return;
        }
    }

    const auto begin = std::chrono::steady_clock::now();
    bool pending = false;
    bool stream_failed = false;
    unsigned long long next_sample = 1;
    while (!stop.load(std::memory_order_acquire))
    {
        if (pending)
        {
            cudaError_t query_err = cudaStreamQuery(stream);
            if (query_err == cudaSuccess)
            {
                l3_live_snapshot_emit(gpu_id, *data);
                pending = false;
            }
            else if (query_err != cudaErrorNotReady)
            {
                printf("gpu%d L3_LIVE stream query failed: %s\n",
                       gpu_id, cudaGetErrorString(query_err));
                fflush(stdout);
                stream_failed = true;
                break;
            }
        }
        if (!pending && !stop.load(std::memory_order_acquire))
        {
            data->sample = next_sample++;
            data->elapsed_us = static_cast<unsigned long long>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - begin).count());
            if (!l3_live_snapshot_enqueue(gpu_id, ctx, *data, stream))
            {
                stream_failed = true;
                break;
            }
            pending = true;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(L3_LIVE_SNAPSHOT_PERIOD_MS));
    }

    // 主线程只在 cudaDeviceSynchronize 返回后置 stop；因此正常收尾时可以
    // 等待最后一批 D2H copy。若 watchdog 自身遇到 CUDA 错误，则不再阻塞。
    if (!stream_failed && stop.load(std::memory_order_acquire))
    {
        if (pending)
        {
            if (cudaStreamSynchronize(stream) == cudaSuccess)
                l3_live_snapshot_emit(gpu_id, *data);
            pending = false;
        }
        data->sample = next_sample++;
        data->elapsed_us = static_cast<unsigned long long>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - begin).count());
        if (l3_live_snapshot_enqueue(gpu_id, ctx, *data, stream))
        {
            if (cudaStreamSynchronize(stream) == cudaSuccess)
                l3_live_snapshot_emit(gpu_id, *data);
        }
    }
    if (data->term_ack_slots != NULL)
        cudaFreeHost(data->term_ack_slots);
    cudaFreeHost(data);
    cudaStreamDestroy(stream);
}
#endif

#if (TIMELINE64 == true)
__device__ __forceinline__ void timeline_idle_write(int *idle_ptr, int value)
{
    int old = l3_atomic_exchange_acq_rel<cuda::thread_scope_system>(
        idle_ptr, value);
    if (old == value)
        return;
    if (value != 0)
    {
        atomicAdd(&g_idle_set_count, 1ull);
        atomicExch(&g_t_idle_last_set, TIMELINE_CLOCK());
    }
    else
    {
        atomicAdd(&g_idle_reset_count, 1ull);
        atomicExch(&g_t_idle_last_reset, TIMELINE_CLOCK());
    }
}
#else
#define timeline_idle_write(idle_ptr, value) \
    l3_atomic_store_release<cuda::thread_scope_system>((idle_ptr), (value))
#endif

#if (TIMELINE64 == true)
__device__ __forceinline__ void timeline_idle_write_reason(int *idle_ptr, int value, unsigned reason)
{
    int old = l3_atomic_exchange_acq_rel<cuda::thread_scope_system>(
        idle_ptr, value);
    if (old == value)
        return;
    if (value != 0)
    {
        atomicAdd(&g_idle_set_count, 1ull);
        atomicExch(&g_t_idle_last_set, TIMELINE_CLOCK());
    }
    else
    {
        atomicAdd(&g_idle_reset_count, 1ull);
        if (reason < 16)
            atomicAdd(&g_idle_reset_reason[reason], 1ull);
        atomicExch(&g_t_idle_last_reset, TIMELINE_CLOCK());
    }
}
#else
#define timeline_idle_write_reason(idle_ptr, value, reason) \
    l3_atomic_store_release<cuda::thread_scope_system>((idle_ptr), (value))
#endif

#if (GLOBAL_ROUND_PROFILE == true)
__device__ __forceinline__ void global_round_profile_add(
    unsigned long long *sum, unsigned long long *max_cycles,
    timeline_clock_type begin, timeline_clock_type end)
{
    unsigned long long delta = (end >= begin) ? (end - begin) : 0ull;
    atomicAdd(sum, delta);
    atomicMax(max_cycles, delta);
}
#endif

#if (GLOBAL_ROUND_ASYNC == true)
// 取得一个 active candidate bank 的短 lease。先增 users 再复核 active/state，
// 从而覆盖“读到 ACTIVE 后 manager 恰好冻结”的窗口；失败的 lease 不会向旧
// bank 写入，manager 等 users==0 后才消费 frozen bank。
__device__ __forceinline__ int async_candidate_acquire(
    async_candidate_control ctl)
{
    if (ctl.active_bank == NULL || ctl.state == NULL || ctl.users == NULL)
        return -1;
    while (true)
    {
        int bank = atomicAdd(ctl.active_bank, 0) & 1;
        if (atomicAdd(ctl.state + bank, 0) != ASYNC_CAND_BANK_ACTIVE)
        {
            __threadfence();
            continue;
        }
        atomicAdd(ctl.users + bank, 1);
        __threadfence();
        int active_now = atomicAdd(ctl.active_bank, 0) & 1;
        int state_now = atomicAdd(ctl.state + bank, 0);
        if (active_now == bank && state_now == ASYNC_CAND_BANK_ACTIVE)
            return bank;
        atomicSub(ctl.users + bank, 1);
        __threadfence();
    }
}

__device__ __forceinline__ void async_candidate_release(
    async_candidate_control ctl, int bank)
{
    if (bank >= 0 && bank < 2 && ctl.users != NULL)
        atomicSub(ctl.users + bank, 1);
}

// 仅由 L3 warp 调用。冻结旧 active bank、切换新 bank、等待旧 bank 的所有
// lease 结束；成功返回后，旧 bank 对所有 producer 已不可见且可安全打包。
__device__ __forceinline__ bool async_candidate_freeze(
    async_candidate_control ctl, int &old_bank, int &new_bank)
{
    old_bank = -1;
    new_bank = -1;
    if (ctl.active_bank == NULL || ctl.state == NULL || ctl.users == NULL)
        return false;

    int old = atomicAdd(ctl.active_bank, 0) & 1;
    int next = old ^ 1;
    if (atomicAdd(ctl.state + next, 0) != ASYNC_CAND_BANK_FREE)
        return false;
    if (atomicCAS(ctl.state + old,
                  ASYNC_CAND_BANK_ACTIVE,
                  ASYNC_CAND_BANK_FROZEN) != ASYNC_CAND_BANK_ACTIVE)
        return false;

    __threadfence_system();
    if (atomicCAS(ctl.state + next,
                  ASYNC_CAND_BANK_FREE,
                  ASYNC_CAND_BANK_ACTIVE) != ASYNC_CAND_BANK_FREE)
    {
        atomicExch(ctl.state + old, ASYNC_CAND_BANK_ACTIVE);
        __threadfence_system();
        return false;
    }
    atomicExch(ctl.active_bank, next);
    __threadfence_system();

    while (atomicAdd(ctl.users + old, 0) != 0)
        __threadfence();
    old_bank = old;
    new_bank = next;
    return true;
}

__device__ __forceinline__ bool async_candidate_quiet(
    async_candidate_control ctl,
    async_candidate_bank bank0, async_candidate_bank bank1)
{
    if (ctl.active_bank == NULL || ctl.state == NULL || ctl.users == NULL
        || bank0.pending == NULL || bank1.pending == NULL)
        return true;
    int active = atomicAdd(ctl.active_bank, 0) & 1;
    int other = active ^ 1;
    if (atomicAdd(ctl.state + active, 0) != ASYNC_CAND_BANK_ACTIVE
        || atomicAdd(ctl.state + other, 0) != ASYNC_CAND_BANK_FREE
        || atomicAdd(ctl.users + active, 0) != 0
        || atomicAdd(ctl.users + other, 0) != 0)
        return false;
    return atomicAdd(bank0.pending, 0) == 0
        && atomicAdd(bank1.pending, 0) == 0;
}
#endif

__device__ __forceinline__ bool node_struct :: operator<(const node_struct& b)
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist < b.dist;
#else
    return node_data_dev[this->id] < node_data_dev[b.id];
#endif
}

__device__ __forceinline__ bool node_struct :: operator>(const node_struct& b)
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist > b.dist;
#else
    return node_data_dev[this->id] > node_data_dev[b.id];
#endif
}

__device__ __forceinline__ bool node_struct :: operator<=(const node_struct& b)
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist <= b.dist;
#else
    return node_data_dev[this->id] <= node_data_dev[b.id];
#endif
}

__device__ __forceinline__ bool node_struct :: operator>=(const node_struct& b)
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist >= b.dist;
#else
    return node_data_dev[this->id] >= node_data_dev[b.id];
#endif
}

__device__ __forceinline__ VALUE_TYPE node_struct :: get_data()
{
#if (USE_DIST_IN_STRUCT == true)
    return this->dist;
#else
    return node_data_dev[this->id];
#endif
}

__device__ __forceinline__ bool node_struct :: filter()
{
#if (USE_DIST_IN_STRUCT == true)
    #if (L3_CHAIN_PARTITION == true)
    return this->dist > node_data_dev[this->id & ~L3_CHAIN_PARTITION_TAG];
#else
    return this->dist > node_data_dev[this->id];
#endif
#else
    return false;
#endif
}

__host__ __device__ node_struct& node_struct :: operator=(const int id_in)
{
    id = id_in;
#if (USE_DIST_IN_STRUCT == true)
    dist = DIST_MAX;
#endif
    return *this;
}

__host__ __device__ node_struct :: node_struct(int id_in, VALUE_TYPE dist_in)
{
    id = id_in;
#if (USE_DIST_IN_STRUCT == true)
    dist = dist_in;
#endif
}

__host__ __device__ node_struct :: node_struct()
{}

// L0 vector queue
template <typename QUEUE_TYPE>
__device__ __forceinline__ void check_out_buffer(NODE_TYPE *node_out, int &node_out_num, 
QUEUE_TYPE mlmq, int block_id, int warp_id, int lane_id, unsigned *debug_time
#if (WORK_COUNT == true)
, int &total_work
#endif
)
{
    const int thresh = node_size;
    // write back for a whole node size
    while (node_out_num > thresh)
    {
        int fill_num = mlq_min(thresh / 2, node_out_num);
        // int fill_num = thresh / 2;

#if (WORK_CLOCK == true)
        if (!lane_id)
        {
            debug_time[1]++;
            debug_time[2]+=fill_num;
        }
#endif

        int write_s = mlmq.write(node_out + node_out_num - fill_num, fill_num, block_id, warp_id, lane_id, debug_time);

        if (write_s == 0)
        {
            node_out_num -= fill_num;
        }

        __syncwarp();
    }
}

// 松弛 dst_v（全局 1-based id），返回是否改进者（赢家）
// 本地: atomicMin(node_data[dst]) ; 远程: 本卡 atomicMin(remote_cand) 聚合（v3，无跨卡无等待）
// 远程改进者（聚合胜出）置位本卡 remote_mark 事件信号（L3 flush warp 消费）
// 远程过滤用 peer_cache（peer node_data 本地镜像，永不重置）：仅 new_dist < cache 才发候选，
//   避免 L3 取走 remote_cand 后 work 又写（remote_cand 被重置为 DIST_MAX → 伪改进 → 无限乒乓）
__device__ __forceinline__ void relax_dst(
    int dst_v, VALUE_TYPE new_dist,
    VALUE_TYPE *node_data, int v_begin, int v_end,
    VALUE_TYPE *remote_cand, unsigned *remote_mark, unsigned *mark_hint, unsigned *mark_hint2, VALUE_TYPE *peer_cache, int peer_v_begin,
#if (GLOBAL_ROUND_ASYNC == true)
    int *remote_pending,
#endif
#if (GHOST_DEPTH > 0)
    int *ghost_id_to_idx, VALUE_TYPE *ghost_node_data, unsigned *ghost_mark,
#endif
    bool &coop_update
#if (WORK_COUNT == true)
    , int &total_work
#endif
)
{
    coop_update = false;
    if (dst_v - 1 >= v_begin && dst_v - 1 < v_end)
    {
        // 本地
        VALUE_TYPE old_dist = *((volatile VALUE_TYPE *)&node_data[dst_v - v_begin]);
        if (new_dist < old_dist)
        {
            VALUE_TYPE update_res;
#ifdef TYPE_INT
            update_res = atomicMin(&node_data[dst_v - v_begin], new_dist);
#else
            update_res = atomicMin_float(&node_data[dst_v - v_begin], new_dist);
#endif
            if (new_dist < update_res)
            {
                coop_update = true;
#if (WORK_COUNT == true)
                total_work++;
                atomicAdd(&g_local_work, (work_count_type)1);
                atomicAdd(&g_hist[dst_v - v_begin], 1u);
#endif
            }
        }
    }
    else
    {
#if (L3_DIAGNOSTICS == true)
        atomicAdd(&g_l3_diagnostics.remote_attempts, 1ull);
#endif
        // 远程: v3 仅本卡原子聚合到 remote_cand（无跨卡无返回值等待）
        // 局部 1-based = dst_v - peer_v_begin（布局对齐 v1：索引 0 闲置）
        // SEED_EXP: peer_cache==NULL（单分区种子实验，无跨卡对端）→ 跨分区边直接跳过，
        //   其影响已由边界种子携带（gpu1 出边到 gpu0 不产生回传，P2 单调性保证）
        if (peer_cache == NULL) { coop_update = false; return; }
        int r_lidx = l3_candidate_index(dst_v - peer_v_begin);
        // 过滤：仅当 new_dist < peer_cache（镜像的 peer node_data 最小值）才发候选
        // peer_cache 用 atomicMin 更新（本卡），永不重置（与 remote_cand 不同）
        VALUE_TYPE cache_old = DIST_MAX;
#if (BULK_NO_CACHE == false)
#if (BULK_CACHE_PREFILTER == true)
        // peer_cache 单调不增且永不清空：普通读只负责过滤已知不可能改进的
        // 候选；读到旧值只会增加一次 atomicMin，不会改变正确性。
        VALUE_TYPE cache_seen = *((volatile VALUE_TYPE *)&peer_cache[r_lidx]);
        if (new_dist >= cache_seen)
        {
#if (L3_DIAGNOSTICS == true)
            atomicAdd(&g_l3_diagnostics.cache_filtered, 1ull);
#endif
            coop_update = false;
            return;
        }
#endif
        cache_old = atomicMin(&peer_cache[r_lidx], new_dist);
#endif
#if (GLOBAL_ROUND_DIRECT_TRACE == true)
        if (dst_v == 193)
        {
            atomicAdd(&g_direct_relax_193, 1ull);
            atomicMin(&g_direct_relax_193_min, (int)new_dist);
        }
        else if (dst_v == 362)
        {
            atomicAdd(&g_direct_relax_362, 1ull);
            atomicMin(&g_direct_relax_362_min, (int)new_dist);
        }
#endif
#if (BULK_TRACE == true)
        if (dst_v == BULK_TRACE_NODE || dst_v == 193 || dst_v == 362)
        {
            unsigned long long t = atomicAdd(&g_bulk_trace_relax, 1ull);
            if (t < 32)
                printf("BULK_TRACE_RELAX g=%d dst=%d nd=%.0f cache_old=%.0f\\n",
                       v_begin, dst_v, (double)new_dist, (double)cache_old);
        }
#endif
        if (new_dist < cache_old)
        {
#if (L3_DIAGNOSTICS == true)
            atomicAdd(&g_l3_diagnostics.candidate_updates, 1ull);
#endif
            // 聚合胜出 = 候选改进者: 本卡 atomicMin 归约 + 置位 remote_mark 事件信号
            VALUE_TYPE rc_old = DIST_MAX;
#if (GLOBAL_ROUND == false)
            unsigned *candidate_hint = mark_hint;
            unsigned *candidate_hint2 = mark_hint2;
#else
            unsigned *candidate_hint = NULL;
            unsigned *candidate_hint2 = NULL;
#endif
#if (BULK_ROUND == true && SEED_BARRIER == false && GLOBAL_ROUND == false)
            unsigned long long *candidate_signal = &g_bulk_mark_signal;
            int *candidate_idle = g_bulk_local_idle;
#else
            unsigned long long *candidate_signal = NULL;
            int *candidate_idle = NULL;
#endif
            bool candidate_won = l3_record_candidate(
                remote_cand, remote_mark,
#if (L3_FAULT_HIDE_HINTS == true)
                NULL, NULL, // simulate false-negative summaries, keep real work
#else
                candidate_hint, candidate_hint2,
#endif
                r_lidx - 1, new_dist, &rc_old,
#if (GLOBAL_ROUND_ASYNC == true)
                remote_pending,
#else
                NULL,
#endif
                candidate_signal, candidate_idle);
#if (GLOBAL_ROUND_DIRECT_TRACE == true)
            if (dst_v == 193 && new_dist < rc_old)
                atomicAdd(&g_direct_cand_193, 1ull);
            else if (dst_v == 362 && new_dist < rc_old)
                atomicAdd(&g_direct_cand_362, 1ull);
#endif
#if (BULK_TRACE == true)
            if (dst_v == BULK_TRACE_NODE || dst_v == 193 || dst_v == 362)
            {
                unsigned long long t = atomicAdd(&g_bulk_trace_cand, 1ull);
                if (t < 32)
                    printf("BULK_TRACE_CAND g=%d dst=%d nd=%.0f rc_old=%.0f mark=%d\\n",
                           v_begin, dst_v, (double)new_dist, (double)rc_old,
                           (int)(new_dist < rc_old));
            }
#endif
            if (candidate_won)
            {
                // mark bit 局部 0-based = dst_v - 1 - peer_v_begin
                int r0 = dst_v - 1 - peer_v_begin;
            }
#if (GHOST_DEPTH > 0)
            // 方案 C: ghost 命中——本卡更新 ghost 距离 + 置事件信号（注入 warp 松弛 ghost 出边，
            //   发现"穿到对端绕回本卡"的路径）。ghost 只是本卡探索锚点，不替代灌值候选。
            if (ghost_id_to_idx != NULL)
            {
                int g2 = ghost_id_to_idx[r_lidx - 1];   // 局部 0-based
                if (g2 >= 0)
                {
                    VALUE_TYPE g_old = atomicMin(&ghost_node_data[g2], new_dist);
                    if (new_dist < g_old)
                        l3_device_mark_publish(
                            &ghost_mark[g2 >> 5], 1u << (g2 & 31));
                }
            }
#endif
#if (WORK_COUNT == true)
            total_work++;
            atomicAdd(&g_remote_work, (work_count_type)1);
#endif
        }
        coop_update = false; // 远程改进不入本卡 node_out（由归属卡 L3 注入处理）
    }
}

#include "l3/l3_region_relax.cuh"

template <typename QUEUE_TYPE>
__device__ int simple_process(int m, int nnz, int *RowPtr, int *ColIdx, 
VALUE_TYPE *edge_data, VALUE_TYPE *node_data, NODE_TYPE *node_in, NODE_TYPE *node_out, 
int &node_in_num, QUEUE_TYPE mlmq, int qshm_size, int block_id, int warp_id, int lane_id, unsigned *debug_time
#if (WORK_COUNT == true)
, int &total_work, int &total_comp
#endif
, int v_begin, int v_end
, VALUE_TYPE *remote_cand, unsigned *remote_mark, unsigned *mark_hint, unsigned *mark_hint2, VALUE_TYPE *peer_cache, int peer_v_begin, int peer_v_local
#if (GLOBAL_ROUND_ASYNC == true)
, int *remote_pending
#endif
#if (GHOST_DEPTH > 0)
, int *ghost_id_to_idx, VALUE_TYPE *ghost_node_data, unsigned *ghost_mark
#endif
, VALUE_TYPE *last_processed
#if (DQ_SPARSE_PHASE == true)
, dq_phase_metrics *phase_metrics
#endif
#if (L3_WORK_DIAG == true)
, unsigned *source_expand_counts
, l3_work_metrics &worker_metrics
#endif
)
{
#if (L3_REGION_RELAX == true)
    if(g_l3_regions.map)
        return l3_region_process<QUEUE_TYPE>(RowPtr,ColIdx,edge_data,node_data,node_in,node_out,node_in_num,mlmq,
                   block_id,warp_id,lane_id,debug_time,v_begin,v_end,
                   remote_cand,remote_mark,mark_hint,mark_hint2,peer_cache,peer_v_begin,last_processed);
#endif
#if (WORK_CLOCK == true)
    unsigned start_time, end_time;
#endif

    int node_out_num = 0;

#if (DQ_SPARSE_PHASE == true)
    bool phase_accepted = false;
#endif

#if (WORK_CLOCK == true)
    start_time = clock();
#endif
    // __shared__ int node_prefix_buffer[node_size * WARP_NUM_PER_BLOCK];
    // __shared__ int node_id_buffer[node_size * WARP_NUM_PER_BLOCK];

    // each edge is processed by one thread
    for (int i = 0; i < node_in_num; i+=WARP_SIZE)
    {

#if (WORK_CLOCK == true)
        unsigned start_time2, end_time2;
#endif

        int src_v = 0;
#if (L3_CHAIN_PARTITION == true)
        int rx_origin=0, chain_source=-1, chain_snapshot=0;
#endif
        int i_idx = i + lane_id;
        if (i_idx < node_in_num)
        {
            // 只处理本卡归属顶点（edge-cut 分区）；远程顶点跳过（步骤 A，步骤 C 跨卡原子）
            // v3 去重（修复 L3 冗余 dirty 乒乓）：仅处理比上次处理距离更小的顶点
            //   （get_data() < last_processed[id]），冗余注入（dist>=last_processed）直接跳过
            // 注意：不用 get_data() <= node_data 条件——L3 可能注入读后改进 node_data，
            //   注入 dist 稍旧，但 work 松弛用权威 node_data 计算，触发处理仍安全。
            int input_id=node_in[i_idx].id;
#if (L3_CHAIN_PARTITION == true)
            rx_origin=input_id & L3_CHAIN_PARTITION_TAG;
            input_id &= ~L3_CHAIN_PARTITION_TAG;
#endif
            if (input_id-1>=v_begin && input_id-1<v_end
                && (last_processed==NULL || node_in[i_idx].get_data()<last_processed[input_id-v_begin]))
                src_v=input_id;
        }
        int node_len = 0;
        int first_edge = 0;
#if (L0_SOURCE_SNAPSHOT == true)
        VALUE_TYPE source_snapshot = 0;
#endif
        
        if (src_v)
        {
            first_edge = RowPtr[src_v - 1 - v_begin];
            node_len = RowPtr[src_v - v_begin] - first_edge;
#if (L3_CHAIN_PARTITION == true)
            // Membership implies original CSR degree exactly two. Reuse the
            // row lengths already loaded, before touching the large route map.
#if (L3_CHAIN_PARTITION_DIAG == true)
            if(rx_origin) atomicAdd(g_l3_chain_partition_counts+3,1ull);
#endif
            if(rx_origin && g_l3_chain_route
#if (L3_CHAIN_PARTITION_DEGREE_GATE == true)
               && node_len==2
#endif
            ) {
#if (L3_CHAIN_PARTITION_DIAG == true)
                atomicAdd(g_l3_chain_partition_counts+4,1ull);
#endif
                const uint64_t route=g_l3_chain_route[src_v-v_begin];
                if(route) {
                    first_edge=int(route>>32)-1;chain_source=int(uint32_t(route));
                    node_len=g_l3_chain_nodes[first_edge].length-1;
                    chain_snapshot=*((volatile VALUE_TYPE *)&node_data[src_v-v_begin]);
#if (L3_CHAIN_PARTITION_DIAG == true)
                    atomicAdd(g_l3_chain_partition_counts,1ull);
                    const unsigned visits=atomicAdd(g_l3_chain_segment_visits+first_edge,1u)+1u;
                    atomicAdd(g_l3_chain_partition_counts+(visits==1u?5:6),1ull);
                    atomicMax(g_l3_chain_partition_counts+7,(unsigned long long)visits);
#endif
                }
            }
#endif
#if (L3_SOURCE_EXPAND_DIAG == true)
            if (source_expand_counts != NULL)
            {
                const unsigned prior = atomicAdd(
                    &source_expand_counts[src_v - v_begin], 1u);
                if (prior == 0)
                    atomicAdd(&g_l3_source_expand_metrics.unique_sources, 1ull);
                else
                {
                    atomicAdd(&g_l3_source_expand_metrics.repeated_sources, 1ull);
                    atomicAdd(&g_l3_source_expand_metrics.repeated_edges,
                              (unsigned long long)node_len);
                }
            }
#endif
            // 记录"以该距离被处理"，兜底扫描据此检测未处理的改进
#if (L0_SOURCE_SNAPSHOT == true)
            source_snapshot = *((volatile VALUE_TYPE *)&node_data[src_v - v_begin]);
            if (last_processed)
                last_processed[src_v - v_begin] = source_snapshot;
#else
            if (last_processed)
                last_processed[src_v - v_begin] =
#if (L3_CHAIN_PARTITION == true)
                    chain_source>=0 ? chain_snapshot :
#endif
                    *((volatile VALUE_TYPE *)&node_data[src_v - v_begin]);
#endif
        }
#if (DQ_SPARSE_PHASE == true)
        if (warp_id == block_id % WARP_NUM_PER_BLOCK) {
            const unsigned accepted = __ballot_sync(FULL_MASK, src_v != 0);
            if (!lane_id && accepted) {
                phase_accepted = true;
                dq_phase_metrics &p = *phase_metrics;
                if (!p.first) {
                    p.first = clock64();
                    p.first_empty = g_dq_sparse[block_id].empty;
                    p.first_unfinished = g_dq_sparse[block_id].unfinished;
                }
            }
        }
#endif
#if (L3_WORK_DIAG == true)
        worker_metrics.popped += i_idx < node_in_num;
        worker_metrics.expanded += src_v != 0;
        worker_metrics.edges += node_len;
#endif
#if (WORK_COUNT == true)
        total_comp += node_len;
#endif
#if (L3_DIAGNOSTICS == true)
        unsigned long long edge_sum = node_len;
        unsigned long long vertex_sum = src_v != 0;
        for (int offset = 16; offset; offset >>= 1) {
            edge_sum += __shfl_down_sync(FULL_MASK, edge_sum, offset);
            vertex_sum += __shfl_down_sync(FULL_MASK, vertex_sum, offset);
        }
        if (!lane_id) {
            atomicAdd(&g_l3_diagnostics.edges, edge_sum);
            atomicAdd(&g_l3_diagnostics.vertices, vertex_sum);
        }
#endif
        __syncwarp();

        // cooperatively process large vertices
        unsigned large_mask = __ballot_sync(FULL_MASK, node_len >= LARGEV);
        while (large_mask != 0)
        {
            unsigned large_lane = find_ms_bit(large_mask);
            int large_st = __shfl_sync(FULL_MASK, first_edge, large_lane);
            int large_len = __shfl_sync(FULL_MASK, node_len, large_lane);
            int large_v = __shfl_sync(FULL_MASK, src_v, large_lane);
#if (L3_CHAIN_PARTITION == true)
            int large_origin=__shfl_sync(FULL_MASK,rx_origin,large_lane);
            int large_chain=__shfl_sync(FULL_MASK,chain_source,large_lane);
            int large_snapshot_chain=__shfl_sync(FULL_MASK,chain_snapshot,large_lane);
#endif
#if (L0_SOURCE_SNAPSHOT == true)
            VALUE_TYPE large_snapshot = __shfl_sync(FULL_MASK, source_snapshot, large_lane);
#endif

            for (int iter_idx = 0; iter_idx < large_len; iter_idx += WARP_SIZE)
            {
                bool coop_update = false;
                int coop_idx = iter_idx + lane_id;
                int dst_v = 0;
                VALUE_TYPE new_dist = 0;
                if (coop_idx < large_len)
                {
                    bool partition_interior=false;
#if (L3_CHAIN_PARTITION == true)
                    if(large_chain>=0)
                        l3_chain_partition_target(large_st,coop_idx,large_chain,large_snapshot_chain,
                                                  dst_v,new_dist,partition_interior);
                    else
#endif
                    {
                    dst_v = ColIdx[large_st + coop_idx] + 1;
                    //new_dist = cub::ThreadLoad<cub::LOAD_CG>(&node_data[large_v]) + edge_data[large_st + coop_idx];
#if (L0_SOURCE_SNAPSHOT == true)
                    new_dist = large_snapshot + edge_data[large_st + coop_idx];
#else
                    new_dist = *((volatile VALUE_TYPE *)&node_data[large_v - v_begin]) + edge_data[large_st + coop_idx];
#endif
                    }
                    // 本地原子 / 远程跨卡原子（relax_dst 统一处理）
#if (L3_WORK_DIAG == true && L3_WORK_COUNT_ONLY == false)
                    if (dst_v - 1 >= v_begin && dst_v - 1 < v_end)
                        ++worker_metrics.owner_destination_edges;
                    else
                        ++worker_metrics.cross_owner_destination_edges;
#endif
#if (WORK_COUNT == true)
                    relax_dst(dst_v, new_dist, node_data, v_begin, v_end, remote_cand, remote_mark, mark_hint, mark_hint2, peer_cache, peer_v_begin,
#if (GLOBAL_ROUND_ASYNC == true)
                              remote_pending,
#endif
#if (GHOST_DEPTH > 0)
                              ghost_id_to_idx, ghost_node_data, ghost_mark,
#endif
                              coop_update, total_work);
#else
                    relax_dst(dst_v, new_dist, node_data, v_begin, v_end, remote_cand, remote_mark, mark_hint, mark_hint2, peer_cache, peer_v_begin,
#if (GLOBAL_ROUND_ASYNC == true)
                              remote_pending,
#endif
#if (GHOST_DEPTH > 0)
                              ghost_id_to_idx, ghost_node_data, ghost_mark,
#endif
                              coop_update);
#endif
#if (L3_CHAIN_PARTITION == true)
                    if(coop_update && partition_interior) {
                        if(last_processed) atomicMin(last_processed+dst_v-v_begin,new_dist);
                        coop_update=false;
#if (L3_CHAIN_PARTITION_DIAG == true)
                        atomicAdd(g_l3_chain_partition_counts+1,1ull);
#endif
                    }
#if (L3_CHAIN_PARTITION_DIAG == true)
                    if(coop_update && large_chain>=0) atomicAdd(g_l3_chain_partition_counts+2,1ull);
#endif
#endif
                }
                __syncwarp();

                unsigned update_mask = __ballot_sync(FULL_MASK, coop_update);
                // push into node_in
                int update_num = count_bit(update_mask);
                int update_pos = node_out_num + count_bit(set_bits(update_mask, 0, lane_id, 32));
                node_out_num += update_num;
                if (coop_update)
                    node_out[update_pos] = node_struct(
#if (L3_CHAIN_PARTITION == true)
                    dst_v | large_origin,
#else
                    dst_v,
#endif
                    new_dist);

                __syncwarp();

#if (WORK_CLOCK == true)
                start_time2 = clock();
#endif

#if (WORK_COUNT == true)
                check_out_buffer<QUEUE_TYPE>(node_out, node_out_num, mlmq, block_id, warp_id, lane_id, debug_time, total_work);
#else
                check_out_buffer<QUEUE_TYPE>(node_out, node_out_num, mlmq, block_id, warp_id, lane_id, debug_time);
#endif
                __syncwarp();
#if (WORK_CLOCK == true)
                end_time2 = clock();
                debug_time[0] += end_time2 - start_time2;
#endif

            }
            large_mask = set_bits(large_mask, 0, large_lane, 1);
        }
        if (node_len >= LARGEV) node_len = 0;
        __syncwarp();

        // process small vertices with load balance
#if (L0_DIRECT_SMALL == true)
        // Low-degree alternative: one source per lane. No edge-to-source
        // scratch map or prefix scan. Large vertices still use the path above.
        int total_edges = __reduce_max_sync(FULL_MASK, unsigned(node_len));
#else
        extern __shared__ int s[];
        VALUE_TYPE *edge_lane_buffer = (VALUE_TYPE *)(s + qshm_size / sizeof(int));

        int nl_st = warp_id * LARGEV * WARP_SIZE;

        typedef cub::WarpScan<int> WarpScan;
        __shared__ typename WarpScan::TempStorage temp_storage[WARP_NUM_PER_BLOCK];
        int edge_offset, total_edges;
        WarpScan(temp_storage[warp_id]).ExclusiveSum(node_len, edge_offset, total_edges);
        // if (!lane_id && !global_wid) printf("size %d\n", total_edges);

#if (WORK_CLOCK == true)
        // if (!lane_id)
        // {
        //     debug_time[1]++;
        //     debug_time[2]+=total_edges;
        // }
#endif

		//write the target lane storage
		int edge_idx = edge_offset;
		while (edge_idx < edge_offset + node_len) {
			edge_lane_buffer[nl_st + edge_idx] = lane_id;
			edge_idx++;
		}

        __syncwarp();

// #if (WORK_COUNT == true)
//         if (!lane_id)
//             total_work += total_edges;
// #endif

#endif

        for (int iter_idx = 0; iter_idx < total_edges;
#if (L0_DIRECT_SMALL == true)
             ++iter_idx)
        {
            bool edge_active = iter_idx < node_len;
#else
             iter_idx += WARP_SIZE)
        {
            edge_idx = iter_idx + lane_id;
            int leader = 0;
            if (edge_idx < total_edges) leader = edge_lane_buffer[nl_st + edge_idx];
            int leader_edge_offset = __shfl_sync(FULL_MASK, edge_offset, leader);
            int leader_first_edge = __shfl_sync(FULL_MASK, first_edge, leader);
            int leader_vertex = __shfl_sync(FULL_MASK, src_v, leader);
#if (L3_CHAIN_PARTITION == true)
            int leader_origin=__shfl_sync(FULL_MASK,rx_origin,leader);
            int leader_chain=__shfl_sync(FULL_MASK,chain_source,leader);
            int leader_snapshot_chain=__shfl_sync(FULL_MASK,chain_snapshot,leader);
#endif
#if (L0_SOURCE_SNAPSHOT == true)
            VALUE_TYPE leader_snapshot = __shfl_sync(FULL_MASK, source_snapshot, leader);
#endif
            bool edge_active = edge_idx < total_edges;
#endif

#if (WORK_CLOCK == true)
            unsigned start_time2, end_time2;
#endif

            bool coop_update = false;
            int dst_v = 0;
            VALUE_TYPE new_dist = 0;
            if (edge_active)
            {
#if (L0_DIRECT_SMALL == true)
                int global_idx = first_edge + iter_idx;
                int leader_vertex = src_v;
#else
                int global_idx = edge_idx - leader_edge_offset + leader_first_edge;
#endif
                bool partition_interior=false;
#if (L3_CHAIN_PARTITION == true)
                if(leader_chain>=0)
                    l3_chain_partition_target(leader_first_edge,edge_idx - leader_edge_offset,leader_chain,leader_snapshot_chain,
                                              dst_v,new_dist,partition_interior);
                else
#endif
                {
                dst_v = ColIdx[global_idx] + 1;
                //new_dist = cub::ThreadLoad<cub::LOAD_CG>(&node_data[leader_vertex]) + edge_data[global_idx];
#if (L0_SOURCE_SNAPSHOT == true && L0_DIRECT_SMALL == true)
                new_dist = source_snapshot + edge_data[global_idx];
#elif (L0_SOURCE_SNAPSHOT == true)
                new_dist = leader_snapshot + edge_data[global_idx];
#else
                new_dist = *((volatile VALUE_TYPE *)&node_data[leader_vertex - v_begin]) + edge_data[global_idx];
#endif
                }
                // 本地原子 / 远程本卡聚合（relax_dst 统一处理）
#if (L3_WORK_DIAG == true && L3_WORK_COUNT_ONLY == false)
                if (dst_v - 1 >= v_begin && dst_v - 1 < v_end)
                    ++worker_metrics.owner_destination_edges;
                else
                    ++worker_metrics.cross_owner_destination_edges;
#endif
#if (WORK_COUNT == true)
                relax_dst(dst_v, new_dist, node_data, v_begin, v_end, remote_cand, remote_mark, mark_hint, mark_hint2, peer_cache, peer_v_begin,
#if (GLOBAL_ROUND_ASYNC == true)
                          remote_pending,
#endif
#if (GHOST_DEPTH > 0)
                          ghost_id_to_idx, ghost_node_data, ghost_mark,
#endif
                          coop_update, total_work);
#else
                relax_dst(dst_v, new_dist, node_data, v_begin, v_end, remote_cand, remote_mark, mark_hint, mark_hint2, peer_cache, peer_v_begin,
#if (GLOBAL_ROUND_ASYNC == true)
                          remote_pending,
#endif
#if (GHOST_DEPTH > 0)
                          ghost_id_to_idx, ghost_node_data, ghost_mark,
#endif
                          coop_update);
#endif
#if (L3_CHAIN_PARTITION == true)
                if(coop_update && partition_interior) {
                    if(last_processed) atomicMin(last_processed+dst_v-v_begin,new_dist);
                    coop_update=false;
#if (L3_CHAIN_PARTITION_DIAG == true)
                    atomicAdd(g_l3_chain_partition_counts+1,1ull);
#endif
                }
#if (L3_CHAIN_PARTITION_DIAG == true)
                if(coop_update && leader_chain>=0) atomicAdd(g_l3_chain_partition_counts+2,1ull);
#endif
#endif
            }

            __syncwarp();

            unsigned update_mask = __ballot_sync(FULL_MASK, coop_update);
            // push into node_in
            int update_num = count_bit(update_mask);
            int update_pos = node_out_num + count_bit(set_bits(update_mask, 0, lane_id, 32));
            node_out_num += update_num;
            if (coop_update)
            {
                node_out[update_pos] = node_struct(
#if (L3_CHAIN_PARTITION == true)
                    dst_v | leader_origin,
#else
                    dst_v,
#endif
                    new_dist);
            }

            __syncwarp();

#if (WORK_CLOCK == true)
            start_time2 = clock();
#endif

#if (WORK_COUNT == true)
            check_out_buffer<QUEUE_TYPE>(node_out, node_out_num, mlmq, block_id, warp_id, lane_id, debug_time, total_work);
#else
            check_out_buffer<QUEUE_TYPE>(node_out, node_out_num, mlmq, block_id, warp_id, lane_id, debug_time);
#endif

            __syncwarp();

#if (WORK_CLOCK == true)
            end_time2 = clock();
            debug_time[0] += end_time2 - start_time2;
#endif
        }
    }

#if (WORK_CLOCK == true)
    start_time = clock();
#endif

    while (node_out_num > 0)
    {
        int fill_num = node_out_num;

#if (WORK_CLOCK == true)
        if (!lane_id)
        {
            debug_time[1]++;
            debug_time[2]+=fill_num;
        }
#endif

        int write_s = mlmq.write(node_out, fill_num, block_id, warp_id, lane_id, debug_time);

        if (write_s == 0)
        {
            node_out_num -= fill_num;
        }
    }
    node_in_num = 0;

#if (DQ_SPARSE_PHASE == true)
    if (!lane_id && phase_accepted) {
        dq_phase_metrics &p = *phase_metrics;
        p.last = clock64();
        ++p.work_calls;
        p.last_empty = g_dq_sparse[block_id].empty;
        p.last_unfinished = g_dq_sparse[block_id].unfinished;
    }
#endif

#if (WORK_CLOCK == true)
    end_time = clock();
    debug_time[0] += end_time - start_time;
#endif

    return 0;
}

#include "l3/l3_tile_loan.cuh"
#include "l3/l3_continuation.cuh"
#if (L3_OWNER_COMMIT == true)
#include "l3/l3_owner_commit.cuh"
#endif

#if (BULK_ROUND == true && BULK_FRONTIER_ENABLED == true)
// BULK_EPOCH 接收 frontier 的发布/领取原语。
// tail 只表示已经写完、可被 work warp 读取的条目数；生产者先写条目并 fence，
// 再一次性发布 tail，避免 work 看到预留槽但读到未完成数据。
__device__ __forceinline__ bool bulk_frontier_empty(
    int *frontier_head, int *frontier_tail, int lane_id)
{
    if (frontier_head == NULL || frontier_tail == NULL)
        return true;
    int head = 0, tail = 0;
    if (!lane_id)
    {
        head = atomicAdd(frontier_head, 0);
        tail = atomicAdd(frontier_tail, 0);
    }
    head = __shfl_sync(FULL_MASK, head, 0);
    tail = __shfl_sync(FULL_MASK, tail, 0);
    return head >= tail;
}

__device__ __forceinline__ void bulk_frontier_reset(
    int *frontier_head, int *frontier_tail, int lane_id)
{
    if (frontier_head == NULL || frontier_tail == NULL)
        return;
    if (!lane_id)
    {
        atomicExch(frontier_head, 0);
        atomicExch(frontier_tail, 0);
    }
    __threadfence();
    __syncwarp();
}

// 安全领取一批 frontier 条目。CAS 防止多个 work warp 因为读取到相同 tail
// 而重复消费；条目复制完成后才返回给 simple_process。
__device__ __forceinline__ int bulk_claim_frontier(
    NODE_TYPE *frontier, int *frontier_head, int *frontier_tail,
    NODE_TYPE *node_in, int capacity, int lane_id)
{
    if (frontier == NULL || frontier_head == NULL || frontier_tail == NULL || capacity <= 0)
        return 0;

    while (true)
    {
        int head = 0, tail = 0;
        if (!lane_id)
        {
            head = atomicAdd(frontier_head, 0);
            tail = atomicAdd(frontier_tail, 0);
        }
        head = __shfl_sync(FULL_MASK, head, 0);
        tail = __shfl_sync(FULL_MASK, tail, 0);
        if (head >= tail)
            return 0;

        int count = mlq_min(capacity, tail - head);
        int old_head = head;
        if (!lane_id)
            old_head = atomicCAS(frontier_head, head, head + count);
        old_head = __shfl_sync(FULL_MASK, old_head, 0);
        if (old_head != head)
            continue;

        for (int i = lane_id; i < count; i += WARP_SIZE)
            node_in[i] = frontier[head + i];
        __syncwarp();
        return count;
    }
}
#endif

#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
// 前置声明：helper 的实现位于本文件后部，但 work kernel 在前面实例化。
__device__ __forceinline__ int async_rx_claim_frontier_batch(
    NODE_TYPE *inbox, int *inbox_count, int *inbox_epoch,
    int *inbox_read_head, int *inbox_inflight,
    int *inbox_state, int *inbox_generation,
    int *active_slot_ptr, int v_begin, int v_local,
    VALUE_TYPE *node_data, NODE_TYPE *node_in, int capacity,
    int lane_id, int &claim_slot);
#endif

#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
// manager-owned receive frontier 复用后部定义的 dirty bitmap 收集器；
// work kernel 位于该定义之前，因此在这里提供前置声明。
__device__ ASYNC_RX_COLLECT_ATTR void collect_dirty_slice(
    unsigned *dirty_bitmap, unsigned *dirty_hint, int c_beg, int c_end,
    int v_local, int v_begin, VALUE_TYPE *node_data,
    NODE_TYPE *node_in, int &out_num, int max_per_lane);
#endif

template <typename QUEUE_TYPE>
__global__
#if (L3_REGION_RELAX == true)
__launch_bounds__(ALIGN_THREAD_PER_BLOCK + WARP_SIZE)
#endif
void work_block_kernel(int m, int nnz, int *RowPtr, int *ColIdx, VALUE_TYPE *edge_data, VALUE_TYPE *node_data, int src,
QUEUE_TYPE mlmq, mlmq_setup setup, int qshm_size, int nshm_size, int *global_exit, work_count_type *global_work_count, int *global_comp_count, int *profile,
int v_begin, int v_end, int v_local,
VALUE_TYPE *remote_cand, unsigned *remote_mark, unsigned *mark_hint, unsigned *mark_hint2, VALUE_TYPE *peer_cache, int peer_v_begin, int peer_v_local,
#if (GLOBAL_ROUND_ASYNC == true)
async_candidate_bank cand_bank0, async_candidate_bank cand_bank1,
async_candidate_control cand_ctl,
#endif
#if (GHOST_DEPTH > 0)
int *ghost_id_to_idx, VALUE_TYPE *ghost_node_data, unsigned *ghost_mark,
#endif
VALUE_TYPE *last_processed
#if (L3_WORK_DIAG == true)
, unsigned *source_expand_counts
#endif
, int n_gpu
#if (L3_TILE_LOAN == true)
, l3_channel_view l3_channel, int l3_tile_loan_enabled
#endif
#if (L3_RX_EXPRESS == true)
, l3_rx_express_ring rx_express, int rx_express_enabled
#endif
#if (L3_RX_L2_PULL == true)
, unsigned *rx_commit_seq, int rx_l2_pull_enabled
#if (L3_RX_L2_PULL_SINGLE_CLAIM == true)
, unsigned *rx_pull_claim_seq
#endif
#if (L3_RX_L2_PULL_DIAG == true)
, unsigned long long *rx_l2_pull_stats
#endif
#endif
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
, unsigned *async_rx_ready_bitmap, unsigned *async_rx_ready_hint
#endif
#if (BULK_ROUND == true)
    , int *bulk_quiesce_req, int *bulk_quiesce_ack,
      int *l3_term_req, int *l3_term_ack_slots
#if (BULK_FRONTIER_ENABLED == true)
, NODE_TYPE *bulk_frontier, int *bulk_frontier_head, int *bulk_frontier_tail
#endif
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
, NODE_TYPE *async_rx_inbox, int *async_rx_inbox_count,
 int *async_rx_inbox_epoch, int *async_rx_inbox_state,
 int *async_rx_inbox_generation, int *async_rx_inbox_read_head,
 int *async_rx_inbox_inflight, int *async_rx_active_slot
#endif
#if (SEED_BARRIER == true)
, int *seed_inject_done
#endif
)
{

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int global_wid = bid * WARP_NUM_PER_BLOCK + wid;
    int lane_id = tid % WARP_SIZE;
#if (L3_IDLE_BACKOFF == true)
    // Warp-uniform consecutive idle-read count (register, reset on any work).
    // The read outcomes feeding it are warp-collective, so every lane tracks
    // the same value without shared memory.
    int idle_streak = 0;
#endif
#if (L3_WAIT_DIAG == true)
    l3_wait_clock wait_diag(!lane_id && wid==0 && bid%8==0 && bid<256);
#endif
#if (L3_WORK_DIAG == true)
    assert(global_wid < L3_WORK_SLOTS);
    l3_work_metrics worker_metrics = {};
#if (L3_WORK_COUNT_ONLY == false)
    const unsigned long long worker_start = clock64();
#endif
#endif

    if (tid >= THREAD_NUM_PER_BLOCK + WARP_SIZE) return;

    __shared__ bool local_exit;

    extern __shared__ int s[];
    NODE_TYPE *node_buf = (NODE_TYPE*)(s + qshm_size / sizeof(int));

    NODE_TYPE *node_in = node_buf + 2 * node_size * wid;
    NODE_TYPE *node_out = node_buf + 2 * node_size * WARP_NUM_PER_BLOCK + wid * node_size * 2;

    // if (!bid && !tid) printf("begin %d", bid);

    if (tid < THREAD_NUM_PER_BLOCK)
    { 
        mlmq.init_device(bid, wid, lane_id, setup);
    }
    else
    {
        if (!lane_id)
            local_exit = false;
    }

    __syncthreads();

    // Local manager
    if (tid >= THREAD_NUM_PER_BLOCK)
    {
        while (l3_atomic_load_acquire<cuda::thread_scope_device>(global_exit) == 0)
        {
            mlmq.update_local_info(lane_id);
            __threadfence();
        }
        local_exit = true;
        __threadfence();
        return;
    }

#if (DQ_SPARSE_PHASE == true)
    __shared__ dq_phase_metrics block_phase;
    if (!lane_id && wid == bid % WARP_NUM_PER_BLOCK) {
        assert(bid < DQ_SPARSE_SLOTS);
        block_phase = {};
        block_phase.start = clock64();
    }
#endif
unsigned debug_time[3];
#if (WORK_CLOCK == true)
    unsigned start_time, end_time;
    unsigned read_time = 0, process_time = 0;
    unsigned total_time = 0;
    for (int i = 0; i < 3; i++)
        debug_time[i] = 0;
#endif

    if (global_wid == 0)
    {
#if (SEED_EXP == true)
        // SEED_EXP（决定性对照）: 边界最终值种子注入——node_data 预置 + 分批 write_through 入 L2。
        // 种子 = gpu1 边界顶点最终距离（host 已算好），绕过 flush→dirty→inject 增量路径。
        if (g_seed_num > 0)
        {
            // 1) 预置权威距离（warp 协作，先全部置值再注入，防 work 读到未置值）
            for (int s0 = 0; s0 < g_seed_num; s0 += WARP_SIZE)
            {
                int si = s0 + lane_id;
                if (si < g_seed_num)
                    node_data[g_seed_ids[si] - v_begin] = g_seed_dists[si];
                __syncwarp();
            }
            // 2) 分批 write_through（node_in 容量 node_size，逐批注入）
            for (int s0 = 0; s0 < g_seed_num; s0 += node_size)
            {
                int cnt = mlq_min(node_size, g_seed_num - s0);
                if (lane_id < cnt)
                    node_in[lane_id] = node_struct(g_seed_ids[s0 + lane_id], g_seed_dists[s0 + lane_id]);
                __syncwarp();
                int write_num = cnt;
                mlmq.write_through(node_in, write_num, bid, wid, lane_id, debug_time);
                __syncwarp();
            }
        }
        else
        {
            // 无种子（种子计算错误或空）: 保持 src 逻辑兜底
            if (src >= v_begin && src < v_end)
            {
                if (!lane_id) node_data[src + 1 - v_begin] = 0;
                node_in[0] = node_struct(src + 1, 0);
                int write_num = 1;
                __syncwarp();
                mlmq.write_through(node_in, write_num, bid, wid, lane_id, debug_time);
            }
        }
#else
        // 源点属本卡才初始化并注入；否则该卡空转（步骤 A，步骤 C 由跨卡注入启动）
        if (src >= v_begin && src < v_end)
        {
            if (!lane_id)
            {
                node_data[src + 1 - v_begin] = 0;
                // 注意：不设置 last_processed[src]=0（保持 DIST_MAX="未处理"标志，
                // 否则 v3 work 去重 0<0 false → src 永不处理）
            }
            node_in[0] = node_struct(src + 1, 0);
            int write_num = 1;
            __syncwarp();
            mlmq.write_through(node_in, write_num, bid, wid, lane_id, debug_time);
        }
#endif
        if (!lane_id){
            *(mlmq.run_begin) = 1;
            // printf("write through\n");
        }
        __threadfence();
    }
    else
    {
        while (!*(mlmq.run_begin))
        {
            __threadfence();
        }
    }

    // nodes that are already in node_in buffer
    int node_in_num = 0;

    int total_work = 0;
    int total_comp = 0;
    int on_the_fly_num = 0;
#if (L3_LOCAL_YIELD_BATCHES > 0)
    int local_yield_batches = 0;
#if (L3_LOCAL_YIELD_DIAG == true)
    bool local_yield_reported = false;
#endif
#endif
#if (L3_RX_EXPRESS == true)
    // Only global_wid==0 polls the ring.  After one express batch, force one
    // ordinary mlmq.read attempt before allowing another express batch.
    // bit 0: the fairness ordinary-read opportunity is still owed;
    // bit 1: this warp currently holds an express slot until process return.
    int express_state = 0;
    unsigned long long express_ticket = 0;
#if (L3_RX_EXPRESS_DIAG == true)
    int express_count = 0;
#endif
#if (L3_RX_EXPRESS_TEST_DELAY == true)
    int express_test_delay = (global_wid == 0) ? L3_RX_EXPRESS_DELAY_LOOPS : 0;
#endif
#endif
#if (L3_RX_L2_PULL == true)
    unsigned rx_l2_pull_seen_seq = 0;
    bool normal_service_owed = false;
#endif
#if (L3_ADMISSION_BUDGET == true)
    l3_admission_cursor admission;
    unsigned deferred_batches=0;
#endif
#if (L3_WORKER_RECOVERY == true)
    int worker_recovery_seen = 0;
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    // 当前 work warp 是否持有一个 receive-frontier claim；claim 的 inflight
    // 计数必须覆盖后续 simple_process，防 manager 提前回收 inbox slot。
    int async_rx_claim_slot = -1;
#endif
#if (L3_TILE_LOAN == true)
    int loan_in_generation = 0;
    int loan_poll_count = 0;
#if (L3_PRODUCER_STOP_POLL == true)
    bool loan_producer_closed=false;
#endif
#if (L3_TILE_LOAN_DIAG == true)
    int loan_diag_polls = 0;
#endif
#endif
#if (L3_TERM_ONLY_WORKER == false)
    bool bulk_quiesced = false;
#endif
#if (BULK_ROUND == true && L3_TERM_HANDSHAKE == true && \
     BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && \
     SEED_BARRIER == false)
    int l3_term_seen = 0;
#endif
#if (L3_IDLE_TOKEN_PROBE == true && L3_PROGRESS_DIAG == true)
    unsigned long long idle_probe_skips = 0;
#endif
#if (GLOBAL_ROUND_ASYNC == true)
    // 异步路径只在终止探测时使用 bulk_quiesce_req。请求值是单调递增的
    // probe token，而不是普通 BULK/GLOBAL_ROUND 使用的 0/1 电平；这样
    // manager 在发现新活动后撤销请求、又立即发起下一次 probe 时，work
    // warp 不会把新请求误认为已经 ACK 过的旧请求。
    int bulk_quiesce_seen = 0;
#endif
#if (BULK_DIAG == true)
    bool bulk_quiesce_diag_seen = false;
#endif

#if (WORK_CLOCK == true)
        unsigned start_time2 = clock();
#endif

        // local_exit: true for end
    while (!local_exit)
    {
#if (L3_TILE_LOAN == true)
        bool loan_control_action = false;
        bool loan_poll_due = false;
#if (L3_MULTI_PRODUCER == true)
        const bool loan_producer_warp=wid==0 && bid<4096;
        if(global_wid==0 && n_gpu>1 && l3_tile_loan_enabled && !lane_id &&
           ((bulk_quiesce_req && l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req)) ||
            (l3_term_req && l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req)))) {
            if(!l3_gate_close(l3_channel.loan_gate,l3_channel.loan_epoch) ||
               !l3_gate_close(l3_channel.peer_loan_gate,l3_channel.loan_epoch)) asm("trap;");
        }
        __syncwarp();
#else
        const bool loan_producer_warp=global_wid==0;
#endif
        if (loan_producer_warp && n_gpu > 1 && l3_tile_loan_enabled
#if (L3_PRODUCER_STOP_POLL == true)
            && (global_wid==0 || !loan_producer_closed)
#endif
           )
        {
            if (loan_poll_count <= 0)
            {
                loan_poll_due = true;
                loan_poll_count = L3_TILE_LOAN_POLL_INTERVAL;
            }
            else
                --loan_poll_count;
        }
#if (L3_PRODUCER_STOP_POLL == true)
        // Query gate never reopens. Ordinary producers can stop polling;
        // global0 must retain its timer and service/apply/reap responsibility.
        if(loan_poll_due && !loan_producer_closed) {
            int closed=0;
            if(!lane_id) closed=(l3_gate_load(l3_channel.loan_gate)&L3_GATE_CLOSED)!=0;
            loan_producer_closed=__shfl_sync(FULL_MASK,closed,0)!=0;
            if(loan_producer_closed && !lane_id) l3_channel.loan_producers[8192+bid]=1;
        }
#endif
        // Polling the peer-owned slot is itself a P2P atomic read.  Keep the
        // home-side apply/reap at a local q1/scratch safe point, but do not
        // require q2==0 there: an outstanding loan deliberately remains in
        // q2's read_done credit until APPLY.  Incoming service may proceed
        // when q2 is empty, or when its only outstanding credit is this GPU's
        // own outgoing loan; this also permits simultaneous opposite loans.
        if (global_wid == 0 && n_gpu > 1 && l3_tile_loan_enabled && loan_poll_due)
        {
            const bool loan_worker_safe =
                node_in_num == 0 && on_the_fly_num == 0 &&
                mlmq.get_local_queue_size(wid) == 0;
            if (loan_worker_safe)
            {
#if (L3_TILE_LOAN_DIAG == true)
                if (!lane_id && loan_diag_polls < 32)
                {
                    const int out_state = l3_tile_loan_load_state(
                        l3_channel.loan_out.state);
                    const int in_state = l3_tile_loan_load_state(
                        l3_channel.loan_in.state);
                    const int out_ready = l3_channel.loan_out.ready == NULL
                                              ? -1
                                              : atomicAdd(
                                                    l3_channel.loan_out.ready,
                                                    0);
                    const int in_ready = l3_channel.loan_in.ready == NULL
                                             ? -1
                                             : atomicAdd(
                                                   l3_channel.loan_in.ready,
                                                   0);
                    printf("L3_LOAN_POLL g%d q=%d lq=%d out=%d in=%d "
                           "out_ready=%d in_ready=%d\n",
                           v_begin, mlmq.get_global_queue_size(),
                           mlmq.get_local_queue_size(wid), out_state,
                           in_state, out_ready, in_ready);
                    ++loan_diag_polls;
                }
#endif
                // Home applies a returned batch before the helper is ACKed;
                // busy work continues through the ordinary path.
                loan_control_action = l3_tile_loan_reap_acked(
                    l3_channel, lane_id);
#if (L3_CONTINUATION == true)
                loan_control_action = l3_continuation_apply(mlmq,l3_channel,
                    bid,wid,lane_id,debug_time,v_begin,v_end,node_data,node_in,
                    total_work) || loan_control_action;
#elif (L3_OWNER_COMMIT == true)
                loan_control_action =
                    l3_owner_commit_reap_out<QUEUE_TYPE>(
                        mlmq, l3_channel, bid, wid, lane_id, debug_time,
                        v_begin, v_end, node_data, last_processed, node_in,
                        node_in_num, on_the_fly_num, total_work) ||
                    loan_control_action;
#else
                loan_control_action =
                    l3_tile_loan_apply_out<QUEUE_TYPE>(
                        mlmq, l3_channel, bid, wid, lane_id, debug_time,
                        v_begin, v_end, node_data, last_processed, node_in,
                        node_in_num, on_the_fly_num, total_work) ||
                    loan_control_action;
#endif

                int loan_q2_size = mlmq.get_global_queue_size();
                int loan_out_credit = 0;
                const int loan_out_state =
                    l3_tile_loan_load_state(l3_channel.loan_out.state);
                if (loan_out_state != L3_TILE_LOAN_FREE &&
                    l3_channel.loan_out.seed_count != NULL)
                    loan_out_credit = atomicAdd(
                        l3_channel.loan_out.seed_count, 0);
                const bool loan_in_accept_allowed =
                    loan_q2_size <= max(0, loan_out_credit);
                const int loan_in_state = l3_tile_loan_load_state(
                    l3_channel.loan_in.state);
                // READY has not been accepted and may be gated by local q2
                // credit. After acceptance, completion/ACK cannot wait for
                // that credit.
                if (loan_in_state != L3_TILE_LOAN_READY ||
#if (L3_MULTI_PRODUCER == true)
                    (l3_gate_load(l3_channel.loan_gate)&L3_GATE_CLOSED) ||
#endif
                    loan_in_accept_allowed)
#if (L3_CONTINUATION == true)
                    loan_control_action = l3_continuation_service(
                        l3_channel,lane_id,loan_in_generation) || loan_control_action;
#elif (L3_OWNER_COMMIT == true)
                    loan_control_action =
                        l3_owner_commit_service_incoming<QUEUE_TYPE>(
                            mlmq, l3_channel, bid, wid, lane_id, debug_time,
                            v_begin, v_end, node_data, node_in, node_in_num,
                            on_the_fly_num, total_work,
                            loan_in_generation) || loan_control_action;
#else
                    loan_control_action = l3_tile_loan_service_incoming(
                        l3_channel, lane_id, loan_in_generation) ||
                        loan_control_action;
#endif
                if (loan_control_action)
                {
                    __syncwarp();
                    continue;
                }
                l3_tile_loan_update_ready(
                    l3_channel, global_wid, lane_id, node_in_num,
                    on_the_fly_num,
                    mlmq.get_local_queue_size(wid) == 0,
                    loan_q2_size == 0);
            }
        }
#endif
#if (L3_RX_EXPRESS_TEST_DELAY == true)
        // Diagnostic-only finite pause: leave the normal workers and RX
        // manager running so a two-slot ring can exercise fallback.  It is
        // deliberately outside production builds and cannot loop forever.
        if (global_wid == 0 && express_test_delay > 0)
        {
            --express_test_delay;
            __syncwarp();
            continue;
        }
#endif
#if (BULK_DIAG == true)
        if (global_wid == 0 && !lane_id)
        {
            unsigned long long wt = atomicAdd(&g_bulk_work_tick, 1ull);
            if ((wt % BULK_WORK_DIAG_K) == 0)
                printf("BULK_WORK g%d tick=%llu q=%d lq=%d in=%d fly=%d req=%d\\n",
                       v_begin, wt, mlmq.get_global_queue_size(),
                       mlmq.get_local_queue_size(wid), node_in_num, on_the_fly_num,
                       bulk_quiesce_req ? l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) : -1);
        }
#endif
#if (SEED_BARRIER == true)
        // 解法0: 接收卡 seed_inject_done 前不读 L2——防 first_pos 提前推进，
        //   使种子注入期写桶 base=0（正确落桶），注入完成后一次干净扫掠。
        //   源卡/单卡不受限（源卡自算子图，无种子注入期）。
        if (n_gpu > 1 && !(src >= v_begin && src < v_end)
            && seed_inject_done != NULL && *(volatile int *)seed_inject_done == 0)
        {
            __threadfence();
            continue;
        }
#endif
#if (WORK_CLOCK == true)
        start_time = clock();
#endif


        // if (!global_wid && !lane_id)
        //     printf("before mlmq reading data: fly %d size %d\n", on_the_fly_num, mlmq.get_global_queue_size());

#if (L3_TILE_LOAN == true)
        bool loan_preloaded = false;
        if (loan_producer_warp && n_gpu > 1 && l3_tile_loan_enabled &&
#if (L3_PRODUCER_STOP_POLL == true)
            !loan_producer_closed &&
#endif
            loan_poll_due && node_in_num == 0 &&
            on_the_fly_num == 0)
        {
            loan_preloaded = l3_tile_loan_try_publish<QUEUE_TYPE>(
                mlmq, l3_channel, bid, wid, lane_id, debug_time,
                RowPtr, v_begin, v_local, node_in,
                node_in_num, on_the_fly_num);
        }
#endif

        if (node_in_num == 0
#if (L3_TILE_LOAN == true)
            && !loan_preloaded
#endif
           )
        {
#if (L3_RX_L2_PULL == true)
            bool rx_l2_pull_taken = false;
            if (n_gpu > 1 && wid == 0
                && !normal_service_owed
                && rx_commit_seq != NULL)
            {
                unsigned seq_now = 0;
                if (!lane_id)
                    seq_now = atomicAdd(rx_commit_seq, 0u);
                seq_now = __shfl_sync(FULL_MASK, seq_now, 0);
                if (seq_now != rx_l2_pull_seen_seq)
                {
                    rx_l2_pull_seen_seq = seq_now;
#if (L3_RX_L2_PULL_DIAG == true)
                    if (!lane_id)
                        l3_rx_l2_pull_add_stat(
                            rx_l2_pull_stats,
                            L3_RX_L2_PULL_HINT_SEEN);
#endif
                    int l1_ready = 0;
                    if (!lane_id)
                        l1_ready = mlmq.get_local_queue_size(wid) > 0;
                    l1_ready = __shfl_sync(FULL_MASK, l1_ready, 0);
                    int pull_service_allowed = 1;
#if (L3_RX_L2_PULL_SINGLE_CLAIM == true)
                    if (rx_l2_pull_enabled && l1_ready
                        && rx_pull_claim_seq != NULL)
                    {
                        if (!lane_id)
                        {
                            unsigned claim_now = atomicAdd(
                                rx_pull_claim_seq, 0u);
                            if (claim_now >= seq_now
                                || atomicCAS(rx_pull_claim_seq,
                                             claim_now, seq_now) != claim_now)
                                pull_service_allowed = 0;
                        }
                        pull_service_allowed = __shfl_sync(
                            FULL_MASK, pull_service_allowed, 0);
#if (L3_RX_L2_PULL_DIAG == true)
                        if (!lane_id && !pull_service_allowed)
                            l3_rx_l2_pull_add_stat(
                                rx_l2_pull_stats,
                                L3_RX_L2_PULL_CLAIM_LOST);
#endif
                    }
#endif
                    if (rx_l2_pull_enabled && l1_ready
                        && pull_service_allowed)
                    {
#if (L3_RX_L2_PULL_DIAG == true)
                        if (!lane_id)
                        {
                            l3_rx_l2_pull_add_stat(
                                rx_l2_pull_stats,
                                L3_RX_L2_PULL_PULL_ATTEMPT);
                            if (on_the_fly_num > 0)
                                l3_rx_l2_pull_add_stat(
                                    rx_l2_pull_stats,
                                    L3_RX_L2_PULL_OLD_INFLIGHT,
                                    static_cast<unsigned long long>(on_the_fly_num));
                            int l2_available = mlmq.get_global_queue_size();
                            if (l2_available > 0)
                                l3_rx_l2_pull_add_stat(
                                    rx_l2_pull_stats,
                                    L3_RX_L2_PULL_L2_AVAILABLE,
                                    static_cast<unsigned long long>(l2_available));
                        }
#endif
                        int pull_num = 0;
#if (L3_RX_L2_PULL_DIAG == true)
                        const int old_inflight_before = on_the_fly_num;
                        int l2_available_before = 0;
                        if (!lane_id)
                            l2_available_before = mlmq.get_global_queue_size();
#endif
                        mlmq.q2.read(node_in, pull_num, bid, wid,
                                     lane_id, debug_time);
                        node_in_num = pull_num;
                        on_the_fly_num += pull_num;
                        __syncwarp();
                        node_in_num = __shfl_sync(FULL_MASK, node_in_num, 0);
                        on_the_fly_num = __shfl_sync(
                            FULL_MASK, on_the_fly_num, 0);
                        if (node_in_num > 0)
                        {
                            rx_l2_pull_taken = true;
                            normal_service_owed = true;
#if (L3_RX_L2_PULL_DIAG == true)
                            if (!lane_id)
                            {
                                l3_rx_l2_pull_add_stat(
                                    rx_l2_pull_stats,
                                    L3_RX_L2_PULL_PULL_RECORDS,
                                    static_cast<unsigned long long>(node_in_num));
                                if (old_inflight_before > 0)
                                    l3_rx_l2_pull_add_stat(
                                        rx_l2_pull_stats,
                                        L3_RX_L2_PULL_OLD_INFLIGHT_WITH_RECORD,
                                        static_cast<unsigned long long>(old_inflight_before));
                                if (l2_available_before > 0)
                                    l3_rx_l2_pull_add_stat(
                                        rx_l2_pull_stats,
                                        L3_RX_L2_PULL_L2_AVAILABLE_WITH_RECORD,
                                        static_cast<unsigned long long>(l2_available_before));
                            }
#endif
                        }
#if (L3_RX_L2_PULL_DIAG == true)
                        else if (!lane_id)
                            l3_rx_l2_pull_add_stat(
                            rx_l2_pull_stats,
                                L3_RX_L2_PULL_EMPTY_PULL);
#endif
                    }
                }
            }
            rx_l2_pull_taken = __shfl_sync(FULL_MASK, rx_l2_pull_taken, 0);
#else
            bool rx_l2_pull_taken = false;
#endif
#if (L3_RX_EXPRESS == true)
            bool express_taken = false;
            if (rx_express_enabled && n_gpu > 1 && global_wid == 0 && !(express_state & 1)
                && on_the_fly_num == 0
                && mlmq.get_local_queue_size(wid) == 0)
            {
                express_taken = l3_rx_express_try_dequeue(
                    rx_express, node_in, node_in_num,
                    express_ticket, lane_id);
                if (express_taken)
                {
#if (L3_RX_EXPRESS_DIAG == true)
                    express_count = node_in_num;
#endif
                    express_state = (express_state & ~1) | 2;
                }
            }
            express_taken = __shfl_sync(FULL_MASK, express_taken, 0);
#else
            bool express_taken = false;
#endif
            if (!express_taken && !rx_l2_pull_taken)
            {
#if (L3_WAIT_DIAG == true)
            wait_diag.tick(1);
#endif
            mlmq.read(node_in, node_in_num, on_the_fly_num, bid, wid, lane_id, debug_time);
#if (L3_WAIT_DIAG == true)
            wait_diag.finish_read(node_in_num>0);
#endif
#if (L3_RX_L2_PULL == true)
            if (wid == 0 && normal_service_owed)
            {
#if (L3_RX_L2_PULL_DIAG == true)
                if (!lane_id)
                    l3_rx_l2_pull_add_stat(
                        rx_l2_pull_stats,
                        L3_RX_L2_PULL_ORDINARY_AFTER_PULL);
#endif
                normal_service_owed = false;
            }
#endif
#if (L3_RX_EXPRESS == true)
            if (global_wid == 0)
            {
                l3_rx_express_note_normal_batch(
                    rx_express, node_in_num, global_wid, lane_id);
                // This read is the bounded fairness opportunity after an
                // express batch, even when the ordinary queue is empty.
                express_state &= ~1;
            }
#endif
#if (BULK_DIAG == true)
            if (global_wid == 0 && !lane_id
                && mlmq.get_global_queue_size() > 0
                && node_in_num == 0)
            {
                unsigned long long wt2 = atomicAdd(&g_bulk_work_tick, 0ull);
                if ((wt2 & 0xfffffu) == 0)
                    printf("BULK_READ_EMPTY g%d tick=%llu q=%d lq=%d fly=%d\\n",
                           v_begin, wt2, mlmq.get_global_queue_size(),
                           mlmq.get_local_queue_size(wid), on_the_fly_num);
            }
#endif
            __syncwarp();
            // on_the_fly_num 是 warp 私有变量；lane0 负责 read_done 记账，
            // 其清零结果必须广播，否则其他 lane 会保留旧值并阻止 quiesce ack。
            on_the_fly_num = __shfl_sync(FULL_MASK, on_the_fly_num, 0);
#if (L3_WORK_DIAG == true)
            // Starvation attribution for the dual-GPU throughput gap: one
            // counter per read attempt outcome, warp-reduced at kernel exit.
            if (!lane_id)
            {
                int qsz = mlmq.get_global_queue_size();
                if (node_in_num == 0)
                {
                    if (qsz > 0) ++worker_metrics.empty_reads;
                    else ++worker_metrics.idle_iters;
                }
                else
                {
                    ++worker_metrics.busy_iters;
                }
            }
#endif
#if (L3_IDLE_BACKOFF == true)
            // Dual-GPU idle backoff (L0/L2 interface).  With two GPUs the L2
            // drains between bucket waves while the peer's next batch or the
            // local bucket manager has not published yet; measured idle
            // iterations grow 15-30x over single GPU.  All work warps then
            // spin-read the delta queue counters, stalling the very managers
            // and L3/termination warps that must publish new work.  Back off
            // briefly after consecutive fully-idle reads so producers get the
            // memory system back.  Single-GPU keeps the original path.
            if (n_gpu > 1 && node_in_num == 0 && on_the_fly_num == 0
                && mlmq.get_local_queue_size(wid) == 0)
            {
                if (idle_streak < 4) ++idle_streak;
                else
                {
                    // ~ 64ns * 2^k, capped at ~2us; sm_70+ warp-synchronous.
                    unsigned backoff_ns = 64u << (idle_streak < 6 ? idle_streak - 4 : 2);
                    __nanosleep(backoff_ns);
                    idle_streak = idle_streak >= 6 ? 6 : idle_streak + 1;
                }
            }
            else
            {
                idle_streak = 0;
            }
#endif
#if (BULK_ROUND == true && BULK_FRONTIER_ENABLED == true)
            // epoch 接收条目不经过 dirty/write_through；L2 当前没有可读批次时，
            // work warp 直接领取 frontier，仍交给现有 simple_process 维护 last_processed。
            if (n_gpu > 1 && node_in_num == 0 && on_the_fly_num == 0)
            {
                node_in_num = bulk_claim_frontier(
                    bulk_frontier, bulk_frontier_head, bulk_frontier_tail,
                    node_in, node_size, lane_id);
            }
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
            // async receive frontier：本地 L2 没有可读条目时，work warp 直接领取
            // manager 已激活的本地 inbox。候选在 helper 内先 atomicMin，只有真正
            // 胜出的条目进入 node_in；不再经过 dirty bitmap / write_through。
            if (n_gpu > 1 && node_in_num == 0 && on_the_fly_num == 0)
            {
                async_rx_claim_slot = -1;
                node_in_num = async_rx_claim_frontier_batch(
                    async_rx_inbox, async_rx_inbox_count, async_rx_inbox_epoch,
                    async_rx_inbox_read_head, async_rx_inbox_inflight,
                    async_rx_inbox_state, async_rx_inbox_generation,
                    async_rx_active_slot, v_begin, v_local, node_data,
                    node_in, node_size, lane_id, async_rx_claim_slot);
            }
#endif
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
            // manager-owned receive frontier：manager 已把 inbox 候选 atomicMin
            // 到本卡 node_data，并置 ready bitmap。只让一个 work warp 负责领取，
            // 避免所有 work warp 重复扫描；领取后仍走现有 simple_process，
            // 因而保留 last_processed、队列选择和本地/远程松弛语义。
            if (n_gpu > 1 && global_wid == 0 && node_in_num == 0
                && on_the_fly_num == 0)
            {
                int rx_dwords = (v_local + 31) / 32;
                int rx_hwords = (rx_dwords + 31) / 32;
                collect_dirty_slice(
                    async_rx_ready_bitmap, async_rx_ready_hint,
                    0, rx_hwords * 32,
                    v_local, v_begin, node_data, node_in, node_in_num,
                    node_size);
                // collect_dirty_slice 的扫描者是 lane0；其余 lane 的
                // node_in_num 必须同步，否则后面的 simple_process 会在
                // 同一 work warp 内发生控制流分歧。
                node_in_num = __shfl_sync(FULL_MASK, node_in_num, 0);
                __syncwarp();
            }
#endif
            // if (!lane_id && node_in_num > 0)
            //     printf("%d read out %d\n", global_wid, node_in_num);
            // if(!lane_id) printf("mlmq node_in %d on_the_fly %d\n", node_in_num, on_the_fly_num);
            // if (on_the_fly_num && node_in_num == 0)//&& node_in_num == 0
            // {
            //     if (!lane_id)
            //     {
            //         mlmq.update_done(on_the_fly_num);
            //         on_the_fly_num = 0;
            //     }
            // }
            }
        }

        // if ( !lane_id)
        // printf("after mlmq reading data num %d fly %d size %d\n", node_in_num, on_the_fly_num, mlmq.get_global_queue_size());
        // if(!lane_id) printf("on_the_fly_num %d\n", on_the_fly_num);
        __syncwarp();
#if (WORK_CLOCK == true)
        end_time = clock();
        read_time += end_time - start_time;
        start_time = clock();
#endif

        if (node_in_num > 0)
        {
#if (L3_ADMISSION_BUDGET == true)
            if(n_gpu>1) {
                int minimum=DIST_MAX,horizon=DIST_MAX,force=0;
                for(int i=lane_id;i<node_in_num;i+=WARP_SIZE) {
                    int local=node_in[i].id-v_begin;
                    if(local>=1 && local<=v_local)
                        minimum=min(minimum,int(*((volatile VALUE_TYPE *)&node_data[local])));
                }
                for(int step=16;step;step/=2)minimum=min(minimum,__shfl_down_sync(FULL_MASK,minimum,step));
                minimum=__shfl_sync(FULL_MASK,minimum,0);
                int allowed=1;
                if(!lane_id) {
                    horizon=atomicAdd(g_l3_admission+2,0);
                    force=l3_term_req && l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req)!=0;
                    bool was_armed=admission.armed;
                    allowed=admission.allow(clock64(),minimum,horizon,force);
                    if(!allowed && !was_armed)++deferred_batches;
                }
                allowed=__shfl_sync(FULL_MASK,allowed,0);
                // Retain input AND L2 in-flight responsibility. In particular,
                // do not fall through to the update_done block below.
                if(!allowed)continue;
            }
#endif
            VALUE_TYPE *remote_cand_use = remote_cand;
            unsigned *remote_mark_use = remote_mark;
            unsigned *mark_hint_use = mark_hint;
            unsigned *mark_hint2_use = mark_hint2;
#if (GLOBAL_ROUND_ASYNC == true)
            // 每次 simple_process 持有一个 bank lease。manager 可以在两个
            // process 调用之间切 bank，但不会在本次边遍历中清空正在使用的 bank。
            int *remote_pending_use = NULL;
            int async_bank_id = -1;
            if (n_gpu > 1 && cand_ctl.active_bank != NULL
                && cand_bank0.cand != NULL && cand_bank1.cand != NULL)
            {
                async_bank_id = async_candidate_acquire(cand_ctl);
                if (async_bank_id == 1)
                {
                    remote_cand_use = cand_bank1.cand;
                    remote_mark_use = cand_bank1.mark;
                    mark_hint_use = cand_bank1.hint;
                    mark_hint2_use = cand_bank1.hint2;
                    remote_pending_use = cand_bank1.pending;
                }
                else if (async_bank_id == 0)
                {
                    remote_cand_use = cand_bank0.cand;
                    remote_mark_use = cand_bank0.mark;
                    mark_hint_use = cand_bank0.hint;
                    mark_hint2_use = cand_bank0.hint2;
                    remote_pending_use = cand_bank0.pending;
                }
            }
#endif
#if (WORK_SEG_PROFILE == true)
            unsigned long long wk_t0 = 0;
            if (!lane_id && global_wid == 0) wk_t0 = clock64();   // 诊断: work 活跃段计时（64 位不回绕）
#endif
#if (L3_WAIT_DIAG == true)
            wait_diag.tick(3);
#endif
#if (L3_WORK_DIAG == true)
#if (L3_WORK_COUNT_ONLY == false)
            unsigned long long worker_segment_start = 0;
            if (!lane_id) {
                worker_segment_start = clock64();
                if (!worker_metrics.calls)
                    worker_metrics.first = worker_segment_start - worker_start;
                ++worker_metrics.calls;
            }
#else
            if (!lane_id) ++worker_metrics.calls;
#endif
#endif
#if (WORK_COUNT == true)
            simple_process<QUEUE_TYPE>(m, nnz, RowPtr, ColIdx, edge_data, node_data, 
            node_in, node_out, node_in_num, mlmq, qshm_size + nshm_size, bid, wid, lane_id, debug_time, total_work, total_comp, v_begin, v_end,
            remote_cand_use, remote_mark_use, mark_hint_use, mark_hint2_use, peer_cache, peer_v_begin, peer_v_local,
#if (GLOBAL_ROUND_ASYNC == true)
            remote_pending_use,
#endif
#if (GHOST_DEPTH > 0)
            ghost_id_to_idx, ghost_node_data, ghost_mark,
#endif
            last_processed
#if (DQ_SPARSE_PHASE == true)
            , &block_phase
#endif
#if (L3_WORK_DIAG == true)
            , source_expand_counts
            , worker_metrics
#endif
            );
#else
            simple_process<QUEUE_TYPE>(m, nnz, RowPtr, ColIdx, edge_data, node_data,
            node_in, node_out, node_in_num, mlmq, qshm_size + nshm_size, bid, wid, lane_id, debug_time, v_begin, v_end,
            remote_cand_use, remote_mark_use, mark_hint_use, mark_hint2_use, peer_cache, peer_v_begin, peer_v_local,
#if (GLOBAL_ROUND_ASYNC == true)
            remote_pending_use,
#endif
#if (GHOST_DEPTH > 0)
            ghost_id_to_idx, ghost_node_data, ghost_mark,
#endif
            last_processed
#if (DQ_SPARSE_PHASE == true)
            , &block_phase
#endif
#if (L3_WORK_DIAG == true)
            , source_expand_counts
            , worker_metrics
#endif
            );
#endif
#if (L3_LOCAL_YIELD_BATCHES > 0)
            // simple_process consumed its input and published every output.
            // Keep original on_the_fly_num live throughout the spill below.
            if (n_gpu > 1) {
                if (mlmq.get_local_queue_size(wid) == 0) {
                    local_yield_batches = 0;
                } else if (++local_yield_batches >= L3_LOCAL_YIELD_BATCHES) {
                    int spilled = mlmq.spill_local_to_l2(bid, wid, lane_id, debug_time);
                    local_yield_batches = 0;
#if (L3_LOCAL_YIELD_DIAG == true)
                    if (!local_yield_reported && !lane_id)
                        printf("L3_LOCAL_YIELD gpu_begin=%d warp=%d spilled=%d budget=%d\n",
                               v_begin, global_wid, spilled, L3_LOCAL_YIELD_BATCHES);
                    local_yield_reported = true;
#endif
                }
            }
#endif
#if (L3_WAIT_DIAG == true)
            wait_diag.finish_work();
#endif
#if (L3_WORK_DIAG == true && L3_WORK_COUNT_ONLY == false)
            if (!lane_id) {
                const auto worker_segment_end = clock64();
                worker_metrics.active += worker_segment_end - worker_segment_start;
                worker_metrics.last = worker_segment_end - worker_start;
            }
#endif
#if (WORK_SEG_PROFILE == true)
            if (!lane_id && global_wid == 0)
                atomicAdd(&g_wk_active_clks, clock64() - wk_t0);
#endif
#if (GLOBAL_ROUND_ASYNC == true)
            if (async_bank_id >= 0)
                async_candidate_release(cand_ctl, async_bank_id);
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
            if (async_rx_claim_slot >= 0)
            {
                // simple_process 已经读取并处理完本批 inbox 条目；此后
                // manager 才可以把 READING slot 置 DONE 并允许发送侧复用。
                atomicSub(async_rx_inbox_inflight + async_rx_claim_slot, 1);
                async_rx_claim_slot = -1;
            }
#endif
#if (L3_RX_EXPRESS == true)
            if (express_state & 2)
            {
                // simple_process has completed all edge relaxations and
                // ordinary queue submissions.  Release only at this point;
                // termination must continue to see the slot as live before it.
                l3_rx_express_release(
                    rx_express, express_ticket,
#if (L3_RX_EXPRESS_DIAG == true)
                    express_count,
#else
                    0,
#endif
                    lane_id);
                express_state = (express_state & ~2) | 1;
            }
#endif
       }

#if (WORK_CLOCK == true)
        end_time = clock();
        process_time += end_time - start_time;
#endif

        // if (!lane_id)
        //     printf("After processing fly %d local_size %d global_size %d\n", on_the_fly_num, mlmq.get_local_queue_size(wid),
        //     mlmq.get_global_queue_size());
    //    __syncwarp();
    //    if(!lane_id) printf("on_the_fly_num %d local queue %d\n", on_the_fly_num, mlmq.get_local_queue_size(wid));
        // __syncwarp();
        if (on_the_fly_num && mlmq.get_local_queue_size(wid) == 0)
        {
            if (!lane_id)
            {
                mlmq.update_done(on_the_fly_num);
                on_the_fly_num = 0;
            }
        }

        // if(!lane_id){
        //     // printf("update done");
        //     if (on_the_fly_num && mlmq.get_local_queue_size(wid) == 0){
        //         // printf("update done %d\n", on_the_fly_num);
        //         mlmq.update_done(on_the_fly_num);
        //         on_the_fly_num = 0;
        //     }
        // }
        __syncwarp();
            on_the_fly_num = __shfl_sync(FULL_MASK, on_the_fly_num, 0);
#if (L3_IDLE_TOKEN_PROBE == true)
        // Neither handshake may ACK a warp holding input, L1, or ancestor
        // credit. Test this original prerequisite before reading requests.
        // Busy warps continue draining normally; no request is acknowledged,
        // cleared, or cached here. A later idle iteration reads the live token.
        const bool probe_local_idle = node_in_num == 0 && on_the_fly_num == 0
                                   && mlmq.get_local_queue_size(wid) == 0;
        const bool probe_allowed = __ballot_sync(FULL_MASK, probe_local_idle) == FULL_MASK;
#if (L3_PROGRESS_DIAG == true)
        if (!lane_id && n_gpu > 1 && !probe_allowed) ++idle_probe_skips;
#endif
#endif
#if (L3_RX_EXPRESS == true)
            bool express_empty_for_worker = true;
            if (global_wid == 0)
            {
                if (!lane_id)
                    express_empty_for_worker = l3_rx_express_empty(rx_express);
                express_empty_for_worker = __shfl_sync(
                    FULL_MASK, express_empty_for_worker, 0);
            }
#endif
#if (BULK_ROUND == true && L3_TERM_ONLY_WORKER == false)
        // BULK_ROUND 的静止握手：manager 置 request 后，work warp 继续排空自己
        // 的 L1/on-the-fly 项；只有 node_in、on_the_fly 和本地队列都为空才 ack。
        // ack 后不再调用 read/process，直到 manager 完成本轮 inbox 发布并清 request。
#if (BULK_DIAG == true)
        int bulk_req_now = (bulk_quiesce_req != NULL)
                         ? l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) : 0;
        int bulk_lq_now = mlmq.get_local_queue_size(wid);
        if (bulk_req_now == 0)
            bulk_quiesce_diag_seen = false;
        else if (!bulk_quiesced && !bulk_quiesce_diag_seen && !lane_id
                 && (node_in_num != 0 || on_the_fly_num != 0 || bulk_lq_now != 0))
        {
            unsigned long long p = atomicAdd(&g_bulk_quiesce_diag_prints, 1ull);
            if (p < 128)
            {
                int fhead = -1;
                int ftail = -1;
#if (BULK_FRONTIER_ENABLED == true)
                if (bulk_frontier_head != NULL)
                    fhead = atomicAdd(bulk_frontier_head, 0);
                if (bulk_frontier_tail != NULL)
                    ftail = atomicAdd(bulk_frontier_tail, 0);
#endif
                printf("BULK_QUIESCE_WAIT g%d wid=%d lq=%d in=%d fly=%d frontier=%d/%d req=%d\n",
                       v_begin, global_wid, bulk_lq_now, node_in_num,
                       on_the_fly_num, fhead, ftail, bulk_req_now);
            }
            bulk_quiesce_diag_seen = true;
        }
#endif
#if (GLOBAL_ROUND_ASYNC == true)
        int bulk_req_token_now = (bulk_quiesce_req != NULL)
                               ? l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) : 0;
#endif
        bool bulk_frontier_is_empty = true;
#if (BULK_FRONTIER_ENABLED == true)
        if (n_gpu > 1)
            bulk_frontier_is_empty = bulk_frontier_empty(
                bulk_frontier_head, bulk_frontier_tail, lane_id);
#endif
        bool bulk_quiesce_ready =
            (n_gpu > 1 && bulk_quiesce_req != NULL && bulk_quiesce_ack != NULL
#if (L3_IDLE_TOKEN_PROBE == true)
             && probe_allowed
#endif
#if (GLOBAL_ROUND_ASYNC == true)
             && bulk_req_token_now != 0
#else
             && l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) != 0
#endif
             && node_in_num == 0 && on_the_fly_num == 0
             && mlmq.get_local_queue_size(wid) == 0
#if (L3_TILE_LOAN == true)
             && l3_tile_loan_worker_idle(l3_channel, global_wid, lane_id,l3_tile_loan_enabled)
#endif
#if (L3_RX_EXPRESS == true)
             && express_empty_for_worker
#endif
             && bulk_frontier_is_empty);
#if (BULK_DIAG == true)
        unsigned bulk_ready_mask = __ballot_sync(FULL_MASK, bulk_quiesce_ready);
        unsigned in_empty_mask = __ballot_sync(FULL_MASK, node_in_num == 0);
        unsigned fly_empty_mask = __ballot_sync(FULL_MASK, on_the_fly_num == 0);
        unsigned lq_empty_mask = __ballot_sync(
            FULL_MASK, mlmq.get_local_queue_size(wid) == 0);
        if (bulk_req_now != 0 && !lane_id
            && bulk_ready_mask != 0 && bulk_ready_mask != FULL_MASK)
        {
            unsigned long long p = atomicAdd(&g_bulk_quiesce_diag_prints, 1ull);
            if (p < 128)
                printf("BULK_QUIESCE_DIVERGE g%d wid=%d ready=0x%08x in0=0x%08x fly0=0x%08x lq0=0x%08x\n",
                       v_begin, global_wid, bulk_ready_mask,
                       in_empty_mask, fly_empty_mask, lq_empty_mask);
        }
#endif
        if (bulk_quiesce_ready)
        {
#if (GLOBAL_ROUND_ASYNC == true)
            // Async probe 使用 token，不能复用普通路径的 bool
            // bulk_quiesced：manager 可能在 token=1 期间发现新活动并把
            // req 清零，随后立即发布 token=2。
            int req_token = bulk_req_token_now;
            if (req_token != bulk_quiesce_seen)
            {
                if (!lane_id) l3_atomic_fetch_add_acq_rel<cuda::thread_scope_device>(bulk_quiesce_ack, 1);
                __syncwarp();
                bulk_quiesce_seen = req_token;
            }
            while (l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) == req_token
                   && l3_atomic_load_acquire<cuda::thread_scope_device>(global_exit) == 0)
            {
                mlmq.update_local_info(lane_id);
                __threadfence();
            }
            bulk_quiesced = false;
#else
            if (!bulk_quiesced)
            {
                if (!lane_id) l3_atomic_fetch_add_acq_rel<cuda::thread_scope_device>(bulk_quiesce_ack, 1);
                __syncwarp();
                bulk_quiesced = true;
            }
            while (l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) != 0 && l3_atomic_load_acquire<cuda::thread_scope_device>(global_exit) == 0)
            {
                mlmq.update_local_info(lane_id);
                __threadfence();
            }
            bulk_quiesced = false;
#endif
            continue;
        }
#endif
        // printf("%d still running\n", lane_id);
#if (L3_WORKER_RECOVERY == true && L3_ACK_SCAN == false)
        if(n_gpu>1 && wid==0)
            l3_worker_recovery_poll(node_data,last_processed,v_local,
                                       bid,gridDim.x,lane_id,worker_recovery_seen);
#endif
#if (BULK_ROUND == true && L3_TERM_HANDSHAKE == true && \
     BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && \
     SEED_BARRIER == false)
        // 默认 BULK 终止握手：请求 token 非零时，work warp 先完成当前
        // simple_process，再把自己的 token 写入独立 ACK 槽；manager 只有在
        // 全部槽位匹配当前 token 后才能发布 READY。按 warp 记录 token，避免
        // 取消后快速重试时把上一轮 ACK 混入新请求。
        if (n_gpu > 1 && l3_term_req != NULL && l3_term_ack_slots != NULL
#if (L3_IDLE_TOKEN_PROBE == true)
            && probe_allowed
#endif
           )
        {
            int term_req_now = l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req);
            if (term_req_now == 0)
            {
                l3_term_seen = 0;
            }
            else
            {
                bool term_local_ready =
                    (node_in_num == 0 && on_the_fly_num == 0
                     && mlmq.get_local_queue_size(wid) == 0
#if (L3_TILE_LOAN == true)
                     && l3_tile_loan_worker_idle(l3_channel, global_wid, lane_id,l3_tile_loan_enabled)
#endif
#if (L3_RX_EXPRESS == true)
                     && express_empty_for_worker
#endif
                     );
                unsigned term_ready_mask = __ballot_sync(
                    FULL_MASK, term_local_ready);
                if (term_ready_mask == FULL_MASK)
                {
                    if (l3_term_seen != term_req_now)
                    {
                        if (!lane_id)
                            l3_atomic_store_release<cuda::thread_scope_device>(l3_term_ack_slots + global_wid, term_req_now);
                        l3_term_seen = term_req_now;
                    }
                    __syncwarp();
#if (L3_WAIT_DIAG == true)
                    wait_diag.tick(4);
#endif
                    while (
#if (L3_ACK_SCAN == true)
                           l3_ack_scan_token_live(l3_term_req,term_req_now,global_exit,lane_id)
#else
                           l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req) == term_req_now && l3_atomic_load_acquire<cuda::thread_scope_device>(global_exit) == 0
#endif
                           )
                    {
#if (L3_WORKER_RECOVERY == true)
#if (L3_ACK_WIDE_SCAN == true)
                        l3_worker_recovery_poll(node_data,last_processed,v_local,
                            bid,gridDim.x,lane_id,worker_recovery_seen,wid);
#else
                        if(wid==0)
                            l3_worker_recovery_poll(node_data,last_processed,v_local,
                                                       bid,gridDim.x,lane_id,worker_recovery_seen);
#endif
#endif
                        __threadfence();
                    }
#if (L3_WAIT_DIAG == true)
                    wait_diag.tick(0);
#endif
                    __syncwarp();
                    continue;
                }
            }
        }
#endif
    }

#if (L3_ADMISSION_BUDGET == true)
    if(n_gpu>1 && !lane_id)atomicAdd(g_l3_admission+3,int(deferred_batches));
#endif
#if (L3_IDLE_TOKEN_PROBE == true && L3_PROGRESS_DIAG == true)
    if (!lane_id) atomicAdd(&g_l3_progress.idle_probe_skips, idle_probe_skips);
#endif
#if (L3_WAIT_DIAG == true)
    wait_diag.save(bid);
#endif
#if (DQ_SPARSE_PHASE == true)
    if (!lane_id && wid == bid % WARP_NUM_PER_BLOCK) {
        block_phase.end = clock64();
        g_dq_phase[bid] = block_phase;
    }
#endif
#if (L3_WORK_DIAG == true)
    for (int offset = 16; offset; offset /= 2) {
        worker_metrics.popped += __shfl_down_sync(FULL_MASK, worker_metrics.popped, offset);
        worker_metrics.expanded += __shfl_down_sync(FULL_MASK, worker_metrics.expanded, offset);
        worker_metrics.edges += __shfl_down_sync(FULL_MASK, worker_metrics.edges, offset);
        worker_metrics.owner_destination_edges +=
            __shfl_down_sync(FULL_MASK, worker_metrics.owner_destination_edges, offset);
        worker_metrics.cross_owner_destination_edges +=
            __shfl_down_sync(FULL_MASK, worker_metrics.cross_owner_destination_edges, offset);
        worker_metrics.empty_reads += __shfl_down_sync(FULL_MASK, worker_metrics.empty_reads, offset);
        worker_metrics.idle_iters += __shfl_down_sync(FULL_MASK, worker_metrics.idle_iters, offset);
        worker_metrics.busy_iters += __shfl_down_sync(FULL_MASK, worker_metrics.busy_iters, offset);
    }
    if (!lane_id) {
#if (L3_WORK_COUNT_ONLY == false)
        worker_metrics.span = clock64() - worker_start;
#else
        // Count-only mode: participation sentinel, NOT elapsed cycles.
        worker_metrics.span = 1;
#endif
        g_l3_work_metrics[global_wid] = worker_metrics;
    }
#endif
#if (WORK_CLOCK == true)
        total_time += clock() - start_time2;
#endif

#if (PROFILE_COUNT == true)
    if (!lane_id)
    {
        atomicAdd(profile, (debug_time[0] + read_time) / WORK_WARP_NUM);
        atomicAdd(profile + 1, total_time / WORK_WARP_NUM);
    }
#endif

#if (WORK_COUNT == true)
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        total_work += __shfl_down_sync(0xffffffff, total_work, offset);
        total_comp += __shfl_down_sync(0xffffffff, total_comp, offset);
    }

    if (!lane_id)
    {
        atomicAdd(global_work_count, total_work);
        atomicAdd(global_comp_count, total_comp);
    }
#endif

#if (WORK_CLOCK == true)
    // Performance profiling
    // Notice: When profiling is enabled, performance will decrease due to calls to printf
    if (!global_wid && !lane_id)
    {
        printf("average write back granularity %.3f\n", 1.0 * debug_time[2] / debug_time[1]);
    }
#endif

}

// 方案2(P3)+H1: backstop 连续读全扫（替代 stride 读，cache 友好 + int4 向量化）。
// 每 lane 负责区间 [beg,end) 内的连续分片，扫 node_data < last_processed，发现改进置 dirty+hint。
// 返回本 lane 是否发现改进（volatile 读 node_data 保跨卡改进可见，last_processed 普通读可 cache）。
__device__ __forceinline__ bool backstop_scan_range(
    VALUE_TYPE *node_data, VALUE_TYPE *last_processed,
    unsigned *dirty_bitmap, unsigned *dirty_hint,
    int beg, int end, int lane_id)
{
    bool found = false;
    // 合并 stride（lane 相邻地址 + int4 对齐）——替代旧"每 lane 连续段"（32 lane 地址分散
    //   → 缓存行 32x 浪费，volatile 下 latency-bound 实测 ~20ms/次）。
    //   volatile 读保 P2P/system 级可见性（SESSION 20 教训，禁 __ldcg 降级）。
    int fa = (beg + 3) & ~3;             // 首个 4 对齐索引（node_data[j0] j0≡0 mod4 才 16B 对齐）
    if (fa < beg) fa = beg;
    // 头部标量 [beg, fa)
    for (int j = beg + lane_id; j < fa && j < end; j += WARP_SIZE)
    {
        VALUE_TYPE nd = *((volatile VALUE_TYPE *)&node_data[j]);
#if (L3_COMPLETED_ROWS == true)
        if (l3_recovery_needs_work(j,nd,last_processed[j]))
#else
        if (nd < last_processed[j])
#endif
        {
            l3_system_mark_publish(&dirty_bitmap[(j - 1) >> 5], 1u << ((j - 1) & 31));
            l3_system_mark_publish(&dirty_hint[(j - 1) >> 10], 1u << (((j - 1) >> 5) & 31));
            found = true;
        }
    }
    // 主体: coalesced stride int4（lane i 读 [fa+4i, fa+4i+4)，128 步长 → 每轮 512B 连续）
    for (int j0 = fa + lane_id * 4; j0 + 3 < end; j0 += WARP_SIZE * 4)
    {
        volatile int4 *nd4p = (volatile int4 *)&node_data[j0];
        int4 lp4 = *((int4 *)&last_processed[j0]);
        int nx = nd4p->x, ny = nd4p->y, nz = nd4p->z, nw = nd4p->w;
#if (L3_COMPLETED_ROWS == true)
        if (l3_recovery_needs_work(j0,nx,lp4.x))
#else
        if (nx < lp4.x)
#endif
        {
            l3_system_mark_publish(&dirty_bitmap[(j0 - 1) >> 5], 1u << ((j0 - 1) & 31));
            l3_system_mark_publish(&dirty_hint[(j0 - 1) >> 10], 1u << (((j0 - 1) >> 5) & 31));
            found = true;
        }
#if (L3_COMPLETED_ROWS == true)
        if (l3_recovery_needs_work(j0+1,ny,lp4.y))
#else
        if (ny < lp4.y)
#endif
        {
            l3_system_mark_publish(&dirty_bitmap[j0 >> 5], 1u << (j0 & 31));
            l3_system_mark_publish(&dirty_hint[j0 >> 10], 1u << ((j0 >> 5) & 31));
            found = true;
        }
#if (L3_COMPLETED_ROWS == true)
        if (l3_recovery_needs_work(j0+2,nz,lp4.z))
#else
        if (nz < lp4.z)
#endif
        {
            l3_system_mark_publish(&dirty_bitmap[(j0 + 1) >> 5], 1u << ((j0 + 1) & 31));
            l3_system_mark_publish(&dirty_hint[(j0 + 1) >> 10], 1u << (((j0 + 1) >> 5) & 31));
            found = true;
        }
#if (L3_COMPLETED_ROWS == true)
        if (l3_recovery_needs_work(j0+3,nw,lp4.w))
#else
        if (nw < lp4.w)
#endif
        {
            l3_system_mark_publish(&dirty_bitmap[(j0 + 2) >> 5], 1u << ((j0 + 2) & 31));
            l3_system_mark_publish(&dirty_hint[(j0 + 2) >> 10], 1u << (((j0 + 2) >> 5) & 31));
            found = true;
        }
    }
    // 尾部 [end-4, end) 重扫（覆盖主循环未覆盖的尾部；幂等 atomicOr 无害）
    for (int j = end - 4; j < end; j++)
    {
        if (j < beg) continue;
        VALUE_TYPE nd = *((volatile VALUE_TYPE *)&node_data[j]);
#if (L3_COMPLETED_ROWS == true)
        if (l3_recovery_needs_work(j,nd,last_processed[j]))
#else
        if (nd < last_processed[j])
#endif
        {
            l3_system_mark_publish(&dirty_bitmap[(j - 1) >> 5], 1u << ((j - 1) & 31));
            l3_system_mark_publish(&dirty_hint[(j - 1) >> 10], 1u << (((j - 1) >> 5) & 31));
            found = true;
        }
    }
    return found;
}

// H1(方向A): backstop 协作全扫（warp0 触发 + 注入 warp 分担）。
#if (L3_RECOVERY_MODE > 0)
// Every helper has a private epoch cursor; it need not observe the cleared
// request between consecutive scans. No queue/credit waits inside a segment.
__device__ __forceinline__ bool l3_recovery_service(
    l3_recovery_cursor &cursor, int rank, int warps, int size, int lane,
    VALUE_TYPE *distances, VALUE_TYPE *processed, unsigned *dirty, unsigned *hint,
    volatile int *request, int *done, int *found)
{
    int epoch = 0;
    if (!lane) epoch = atomicAdd((int *)request, 0);
    epoch = __shfl_sync(FULL_MASK, epoch, 0);
    if (!epoch) return false;
    if (cursor.fresh(epoch)) {
        const auto range = l3_recovery_slice(size, rank, warps);
        bool improved = backstop_scan_range(distances, processed, dirty, hint,
                                            range.begin, range.end, lane);
        const unsigned mask = __ballot_sync(FULL_MASK, improved);
#if (L3_RECOVERY_TEST_DELAY == true)
        if (epoch == 1 && rank == warps - 1) {
            const auto begin = clock64();
            while (clock64() - begin < 250000ull) { }
#if (L3_PROGRESS_DIAG == true)
            if (!lane) ++g_l3_progress.recovery_delays;
#endif
            }
        }
#endif
        // Global dirty publication precedes shared completion, for every lane.
        __threadfence();
        __syncwarp();
        if (!lane) {
            if (mask) atomicOr(found, 1);
            __threadfence_block();
            atomicAdd(done, 1);
        }
        cursor.completed(epoch);
        __syncwarp();
    }
    // Stay available while this epoch is live, but never wait for observing 0.
    return true;
}
#endif
// 总 warp = 1 + INJECT_WARP_NUM，每 warp 连续段 = v_local / n_warp，余数给前几个。
// warp0 = 段 0；注入 warp i（inj_id=i） = 段 i+1。仅 warp0 调用，返回整卡是否发现改进。
__device__ __forceinline__ bool backstop_collaborative(
    VALUE_TYPE *node_data, VALUE_TYPE *last_processed,
    unsigned *dirty_bitmap, unsigned *dirty_hint,
    volatile int *bs_req, int *bs_done, int *bs_found,
    int v_local, int lane_id
#if (L3_WORKER_RECOVERY == true)
    , int recovery_blocks
#endif
#if (L3_RECOVERY_MODE > 0)
    , int extra_helpers, int *request_epoch
#endif
    )
{
#if (L3_PROGRESS_DIAG == true)
    const unsigned long long progress_backstop_start = clock64();
#endif
#if (L3_WORKER_RECOVERY == true)
    bool distributed_found=l3_worker_recovery_request(dirty_bitmap,dirty_hint,recovery_blocks,lane_id);
#if (L3_PROGRESS_DIAG == true)
    if(!lane_id) {
        g_l3_progress.backstop_cycles+=clock64()-progress_backstop_start;
        ++g_l3_progress.backstop_calls;
        g_l3_progress.backstop_positive+=distributed_found;
    }
#endif
    return distributed_found;
#else
    const int n_warp = 1 + INJECT_WARP_NUM
#if (L3_RECOVERY_MODE > 0)
        + extra_helpers
#endif
        ;
    int per = v_local / n_warp;
    int rem = v_local % n_warp;
    int seg0_beg = 1;
    int seg0_end = seg0_beg + per + (rem > 0 ? 1 : 0);
    // 1. 置请求（清结果计数，再广播请求）
    if (!lane_id)
    {
#if (L3_LIVE_SNAPSHOT == true)
        l3_live_backstop_state(70, bs_req, bs_done, bs_found);
#endif
        *bs_found = 0;
        *bs_done = 0;
    }
    __threadfence_block();
    if (!lane_id)
    {
#if (L3_RECOVERY_MODE > 0)
        assert(*request_epoch >= 0 && *request_epoch < INT_MAX);
        atomicExch((int *)bs_req, ++*request_epoch);
#else
        *bs_req = 1;
#endif
#if (L3_LIVE_SNAPSHOT == true)
        l3_live_backstop_state(71, bs_req, bs_done, bs_found);
#endif
    }
    __threadfence_block();
    // 2. warp0 自扫段 0
    bool my_found = backstop_scan_range(node_data, last_processed, dirty_bitmap, dirty_hint,
                                        seg0_beg, seg0_end, lane_id);
    // 3. 等注入 warp 完成（lane0 轮询 + __syncwarp 广播）
    if (!lane_id)
    {
        while (*(volatile int *)bs_done < n_warp - 1)
        {
#if (L3_LIVE_SNAPSHOT == true)
            l3_live_backstop_state(72, bs_req, bs_done, bs_found);
#endif
            __threadfence();
        }
#if (L3_LIVE_SNAPSHOT == true)
        l3_live_backstop_state(73, bs_req, bs_done, bs_found);
#endif
    }
    __syncwarp();
    // 4. 归约 + 清请求
#if (L3_RECOVERY_MODE > 0)
    if (!lane_id) assert(atomicAdd(bs_done, 0) == n_warp - 1);
#endif
    bool found = my_found || (*(volatile int *)bs_found != 0);
#if (L3_PROGRESS_DIAG == true)
    const unsigned found_mask = __ballot_sync(FULL_MASK, found);
    if (!lane_id && found_mask) ++g_l3_progress.backstop_positive;
#endif
    if (!lane_id)
    {
        *bs_req = 0;
#if (L3_LIVE_SNAPSHOT == true)
        l3_live_backstop_state(74, bs_req, bs_done, bs_found);
#endif
    }
    __threadfence_block();
    // This helper is called only by the RX/termination manager in direct RX.
#if (L3_PROGRESS_DIAG == true)
    if (!lane_id) {
        g_l3_progress.backstop_cycles += clock64() - progress_backstop_start;
        ++g_l3_progress.backstop_calls;
    }
#endif
    return found;
#endif
}

// E3: dirty 区间收集（warp0 / 注入 warp 共用），串行 lane0 两级扫描（design_v3 §16）
// 第一级扫本 slice 的 dirty_hint（每 32 个 dirty word 一个 hint bit，稀疏），
// 命中块（bit=1）再扫该 32-words block 的 dirty_bitmap，CAS 非零 word→0，
// 收集到 node_in[0..out_num)，DIST_MAX（L3 在途）位 restore，容量触顶回填剩余位。
// 保守清零（§16.5 风险1）：块消费完且未触顶 → volatile 复扫块全 0 →
//   atomicCAS(hint, hv_old, 0)（hv_old=块扫描前快照，并发置位→CAS 失败保留）。
// 注：slice 必须 32-word 块对齐（c_beg/c_end 为 32 倍数，hint word 单消费者，§16.5 风险2）。
// 注：lane 并行扫描尝试（prefix-sum / atomicAdd 两版）均有未定位的卡死问题，弃用。
__device__ ASYNC_RX_COLLECT_ATTR void collect_dirty_slice(
    unsigned *dirty_bitmap, unsigned *dirty_hint, int c_beg, int c_end, int v_local, int v_begin,
    VALUE_TYPE *node_data, NODE_TYPE *node_in, int &out_num, int max_per_lane)
{
    const int lid = threadIdx.x & 31;   // lane id（manage kernel 的 lane_id）
    out_num = 0;
    if (lid == 0)
    {
        const int dwords = (v_local + 31) / 32;
        // c_beg/c_end 为 32 倍数（块对齐），hint word 索引即 c>>5
        for (int hb = (c_beg >> 5); hb < (c_end >> 5); hb++)
        {
            unsigned hv = l3_atomic_load_relaxed<cuda::thread_scope_system>(&dirty_hint[hb]);
            if (!hv) continue;
            bool block_cleared = true;      // 块内所有 dirty word 均已消费（或本就为空）
            bool cap_hit = false;           // 容量触顶 → 跳过保守清零
            int w0 = hb << 5;               // 块 = [w0, w0+32)（slice 块对齐保证不越界到别块）
            int w = w0;
            for (; w < w0 + 32 && !cap_hit; w++)
            {
                if (w >= dwords) break;     // 对齐上取整可能越界，clamp
                int val = l3_atomic_load_relaxed<cuda::thread_scope_system>(&dirty_bitmap[w]);
                if (!val) continue;
                if (!l3_system_mark_claim(&dirty_bitmap[w], (unsigned)val)) continue;
                block_cleared = false;
                for (int b = 0; b < 32; b++)
                {
                    if (!(val & (1u << b))) continue;
                    if (out_num >= max_per_lane)
                    {
                        // 容量触顶：回填当前 word 剩余 bit（b..31），保 dirty/hint 置位，下轮继续
                        l3_system_mark_publish(&dirty_bitmap[w], val & ~((1u << b) - 1));
                        cap_hit = true;
                        break;
                    }
                    int j = w * 32 + b;
                    if (j >= v_local) continue;
                    VALUE_TYPE nd = *((volatile VALUE_TYPE *)&node_data[j + 1]);
                    if (nd != DIST_MAX)
                    {
                        node_in[out_num] = node_struct(v_begin + j + 1, nd);
#if (GLOBAL_ROUND_DIRECT_TRACE == true)
                        if (v_begin + j + 1 == 193)
                        {
                            atomicAdd(&g_direct_inj_193, 1ull);
                            atomicMin(&g_direct_inj_193_min, (int)nd);
                        }
                        else if (v_begin + j + 1 == 362)
                        {
                            atomicAdd(&g_direct_inj_362, 1ull);
                            atomicMin(&g_direct_inj_362_min, (int)nd);
                        }
#endif
                        out_num++;
                    }
                    else
                    {
                        l3_system_mark_publish(&dirty_bitmap[w], 1u << b);
                    }
                }
            }
            // 保守清零：块消费完且未触顶 → volatile 复扫块全 0 才 CAS(hint, hv, 0)
            // SESSION 22 根治 ABA 假阴：CAS 清后立即复扫 dirty_bitmap，非零则重新置 hint
            //   （peer 置 dirty 时 hint or 幂等 → CAS 误清 → 复扫发现非零重新置 hint 兜底）
            if (block_cleared && !cap_hit)
            {
                bool all_zero = true;
                for (w = w0; w < w0 + 32; w++)
                {
                    if (w >= dwords) break;
                    if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w]) != 0) { all_zero = false; break; }
                }
                if (all_zero)
                {
                    l3_atomic_compare_exchange_acq_rel<cuda::thread_scope_system>(&dirty_hint[hb], hv, 0u);
                    bool dirty_again = false;
                    for (w = w0; w < w0 + 32; w++)
                    {
                        if (w >= dwords) break;
                        if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w]) != 0) { dirty_again = true; break; }
                    }
                    if (dirty_again)
                        l3_system_mark_publish(&dirty_hint[hb], hv);
                }
            }
        }
    }
    __syncwarp();
    out_num = __shfl_sync(FULL_MASK, out_num, 0);
}

// SESSION 22 修复：dirty_hint ABA 假阴兜底（与 mark_hint SESSION 20 同构）。
// 直接扫 c_beg..c_end 的 dirty_bitmap（不依赖 dirty_hint），CAS 取走非零 word，
// 逐 bit 消费（node_data != DIST_MAX 收集，== DIST_MAX restore 并同步置 hint，L3 在途）。
__device__ __forceinline__ void collect_dirty_slice_nohint(
    unsigned *dirty_bitmap, unsigned *dirty_hint, int c_beg, int c_end, int v_local, int v_begin,
    VALUE_TYPE *node_data, NODE_TYPE *node_in, int &out_num, int max_per_lane)
{
    const int lid = threadIdx.x & 31;
    out_num = 0;
    if (lid == 0)
    {
        const int dwords = (v_local + 31) / 32;
        bool cap_hit = false;
        for (int w = c_beg; w < c_end && !cap_hit; w++)
        {
            if (w >= dwords) break;
            int val = l3_atomic_load_relaxed<cuda::thread_scope_system>(&dirty_bitmap[w]);
            if (!val) continue;
            if (!l3_system_mark_claim(&dirty_bitmap[w], (unsigned)val)) continue;
            for (int b = 0; b < 32; b++)
            {
                if (!(val & (1u << b))) continue;
                if (out_num >= max_per_lane)
                {
                    l3_system_mark_publish(&dirty_bitmap[w], val & ~((1u << b) - 1));
                    cap_hit = true;
                    break;
                }
                int j = w * 32 + b;
                if (j >= v_local) continue;
                VALUE_TYPE nd = *((volatile VALUE_TYPE *)&node_data[j + 1]);
                if (nd != DIST_MAX)
                {
                    node_in[out_num] = node_struct(v_begin + j + 1, nd);
                    out_num++;
                }
                else
                {
                    // restore：L3 flush 在途未落位，下轮再试（同步置 hint，保正常路径可见）
                    l3_system_mark_publish(&dirty_bitmap[w], 1u << b);
                    l3_system_mark_publish(&dirty_hint[w >> 5], 1u << (w & 31));
                }
            }
        }
    }
    __syncwarp();
    out_num = __shfl_sync(FULL_MASK, out_num, 0);
}

// exp-sp_async: 注入顶点 BF 化 helper（SP Async "本地 Dijkstra + 跨进程 BF" 思想映射）
// 对输入 buffer 每个顶点的本地出边做原子松弛（BF 式，绕开 delta 桶序假设）。
// 只松弛【本地邻居】（dst_v 属本卡），远程邻居跳过——远程候选由 L3 正常路径处理，
// 避免注入 BF 产生反向 remote_cand 造成跨卡乒乓。
// 松弛赢的本地邻居：①【置 dirty_bitmap/hint】（队列感知：dirty → 注入 → write_through
//   → L2 → work 处理，last_processed 由 work 正常更新，保持 L2 队列协议完整）；
//   ②【写入输出 buffer bf_out】（供深度 BF 下一层扩散，容量 bf_cap）。
// 返回输出 buffer 的顶点数（下一层输入），0 表示无可扩散。
//
// 【无分歧设计】采用与 work kernel large 顶点一致的模式：
// 先每 lane 收集一个输入顶点的 (first_edge,node_len)，__ballot_sync 得到 large_mask
// （node_len>0 的 lane），while(large_mask) 内 find_ms_bit 选 leader 广播其出边范围，
// 全 warp 协作松弛该顶点的所有出边，然后 set_bits 清除 leader。全程无 continue 分歧，
// 每个 __syncwarp() 所有 lane 同步到达。
// 输出收集：bf_slot_ptr（调用方共享内存，per 注入 warp 专属）atomicAdd 槽分配。
__device__ __forceinline__ int inject_bf_relax(
    NODE_TYPE *node_in, int node_in_num,
    int *RowPtr, int *ColIdx, VALUE_TYPE *edge_data, VALUE_TYPE *node_data,
    int v_begin, int v_end,
    VALUE_TYPE *remote_cand, unsigned *remote_mark, unsigned *mark_hint, unsigned *mark_hint2,
    VALUE_TYPE *peer_cache, int peer_v_begin,
    unsigned *dirty_bitmap, unsigned *dirty_hint, int v_local,
    NODE_TYPE *bf_out, int bf_cap, int *bf_slot_ptr
#if (WORK_COUNT == true)
    , int &total_work
#endif
)
{
    const int lid = threadIdx.x & 31;
    if (!lid) *bf_slot_ptr = 0;
    __syncwarp();
    int n_set = 0;
    for (int base = 0; base < node_in_num; base += WARP_SIZE)
    {
        int idx = base + lid;
        int u = 0;
        int first_edge = 0, node_len = 0;
        if (idx < node_in_num)
        {
            int gid1 = node_in[idx].id;          // 全局 1-based
            if (gid1 - 1 >= v_begin && gid1 - 1 < v_end)
            {
                u = gid1;
                first_edge = RowPtr[gid1 - 1 - v_begin];
                node_len = RowPtr[gid1 - v_begin] - first_edge;
            }
        }
        __syncwarp();
        unsigned large_mask = __ballot_sync(FULL_MASK, node_len >= 1);
        while (large_mask != 0)
        {
            unsigned large_lane = find_ms_bit(large_mask);
            int lu = __shfl_sync(FULL_MASK, u, large_lane);
            int lfirst = __shfl_sync(FULL_MASK, first_edge, large_lane);
            int llen = __shfl_sync(FULL_MASK, node_len, large_lane);
            // 全 warp 协作松弛该顶点的出边（stride）
            for (int e = lid; e < llen; e += WARP_SIZE)
            {
                int dst_v = ColIdx[lfirst + e] + 1;   // 全局 1-based
                if (!(dst_v - 1 >= v_begin && dst_v - 1 < v_end)) continue;  // 远程跳过
                VALUE_TYPE new_dist = *((volatile VALUE_TYPE *)&node_data[lu - v_begin]) + edge_data[lfirst + e];
                VALUE_TYPE old_dist = *((volatile VALUE_TYPE *)&node_data[dst_v - v_begin]);
                if (new_dist >= old_dist) continue;
                VALUE_TYPE update_res;
#ifdef TYPE_INT
                update_res = atomicMin(&node_data[dst_v - v_begin], new_dist);
#else
                update_res = atomicMin_float(&node_data[dst_v - v_begin], new_dist);
#endif
                if (new_dist < update_res)
                {
#if (WORK_COUNT == true)
                    total_work++;
                    atomicAdd(&g_local_work, (work_count_type)1);
                    atomicAdd(&g_hist[dst_v - v_begin], 1u);
#endif
                    // 队列感知：改进者置 dirty（本卡），work 侧 last_processed 正常
                    int j = dst_v - 1 - v_begin;      // 局部 0-based
                    l3_system_mark_publish(&dirty_bitmap[j >> 5], 1u << (j & 31));
                    l3_system_mark_publish(&dirty_hint[j >> 10], 1u << ((j >> 5) & 31));
                    n_set++;
                    // 输出到 bf_out（下一层 BF 输入）：共享槽 atomicAdd 分配
                    int slot = atomicAdd(bf_slot_ptr, 1);
                    if (slot < bf_cap)
                        bf_out[slot] = node_struct(dst_v, new_dist);
                }
            }
            __syncwarp();
            large_mask = set_bits(large_mask, 0, large_lane, 1);
        }
        __syncwarp();
    }
    __syncwarp();
    int bf_cnt = __shfl_sync(FULL_MASK, *bf_slot_ptr, 0);
    if (bf_cnt > bf_cap) bf_cnt = bf_cap;
    (void)n_set;
    return bf_cnt;
}

#if (GHOST_DEPTH > 0)
// phaseI 方案 C: ghost 出边松弛 helper（BF 式，leader 无分歧模式同 inject_bf_relax）。
// 输入 g_in[] 为 ghost 索引列表；松弛每个 ghost 的出边（ghost 子图 CSR）:
//   返回边（dst∈本卡）→ atomicMin(node_data) 赢 → 置 dirty（注入路径 write_through 处理）；
//   ghost 间边（dst 也是 ghost）→ atomicMin(ghost_node_data) 赢 → 写入 bf_out（下一层输入）。
// 返回下一层 ghost 数（0 = 无可扩散）。
__device__ __forceinline__ int ghost_bf_relax(
    int *g_in, int g_in_num,
    int *g_row_start, int *g_col, VALUE_TYPE *g_edge_data,
    int *g_id_to_idx, int peer_v_begin,
    VALUE_TYPE *node_data, unsigned *dirty_bitmap, unsigned *dirty_hint,
    int v_begin, int v_end, int v_local,
    VALUE_TYPE *ghost_node_data,
    VALUE_TYPE *remote_cand, unsigned *remote_mark, unsigned *mark_hint, unsigned *mark_hint2,
    int *bf_out, int bf_cap, int *bf_slot_ptr)
{
    const int lid = threadIdx.x & 31;
    if (!lid) *bf_slot_ptr = 0;
    __syncwarp();
    for (int base = 0; base < g_in_num; base += WARP_SIZE)
    {
        int idx = base + lid;
        int gidx = -1, first_edge = 0, node_len = 0;
        if (idx < g_in_num)
        {
            gidx = g_in[idx];
            if (gidx >= 0)
            {
                first_edge = g_row_start[gidx];
                node_len = g_row_start[gidx + 1] - first_edge;
            }
        }
        __syncwarp();
        unsigned large_mask = __ballot_sync(FULL_MASK, node_len >= 1);
        while (large_mask != 0)
        {
            unsigned large_lane = find_ms_bit(large_mask);
            int lg = __shfl_sync(FULL_MASK, gidx, large_lane);
            int lfirst = __shfl_sync(FULL_MASK, first_edge, large_lane);
            int llen = __shfl_sync(FULL_MASK, node_len, large_lane);
            for (int e = lid; e < llen; e += WARP_SIZE)
            {
                int dst_v = g_col[lfirst + e];          // 全局 1-based
                VALUE_TYPE nd = *((volatile VALUE_TYPE *)&ghost_node_data[lg]) + g_edge_data[lfirst + e];
                if (dst_v - 1 >= v_begin && dst_v - 1 < v_end)
                {
                    // 返回边: 改进本卡顶点（队列感知置 dirty，注入路径 write_through → L2 → work）
                    VALUE_TYPE old_dist = *((volatile VALUE_TYPE *)&node_data[dst_v - v_begin]);
                    if (nd < old_dist)
                    {
                        VALUE_TYPE update_res;
#ifdef TYPE_INT
                        update_res = atomicMin(&node_data[dst_v - v_begin], nd);
#else
                        update_res = atomicMin_float(&node_data[dst_v - v_begin], nd);
#endif
                        if (nd < update_res)
                        {
#if (WORK_COUNT == true)
                            atomicAdd(&g_local_work, (work_count_type)1);
                            atomicAdd(&g_hist[dst_v - v_begin], 1u);
#endif
                            int j = dst_v - 1 - v_begin;
                            l3_system_mark_publish(&dirty_bitmap[j >> 5], 1u << (j & 31));
                            l3_system_mark_publish(&dirty_hint[j >> 10], 1u << ((j >> 5) & 31));
                        }
                    }
                }
                else if (dst_v - 1 >= peer_v_begin && g_id_to_idx != NULL)
                {
                    // ghost 间边: 改进其他 ghost → 下一层输入
                    int g2 = g_id_to_idx[dst_v - 1 - peer_v_begin];
                    if (g2 >= 0)
                    {
                        VALUE_TYPE old_dist = *((volatile VALUE_TYPE *)&ghost_node_data[g2]);
                        if (nd < old_dist)
                        {
                            VALUE_TYPE update_res = atomicMin(&ghost_node_data[g2], nd);
                            if (nd < update_res)
                            {
                                // 同步写远程候选（灌值/异步 flush 携带 ghost 探索值给对端权威）
                                int r_lidx = dst_v - peer_v_begin;      // 局部 1-based
                                VALUE_TYPE rc_old = atomicMin(&remote_cand[r_lidx], nd);
                                if (nd < rc_old)
                                {
                                    int r0 = dst_v - 1 - peer_v_begin;   // 局部 0-based
                                    l3_device_mark_publish(&remote_mark[r0 >> 5], 1u << (r0 & 31));
                                    if (mark_hint != NULL)
                                        atomicOr(&mark_hint[r0 >> 10], 1u << ((r0 >> 5) & 31));
                                    if (mark_hint2 != NULL)
                                        atomicOr(&mark_hint2[r0 >> 15], 1u << ((r0 >> 10) & 31));
                                }
                                int slot = atomicAdd(bf_slot_ptr, 1);
                                if (slot < bf_cap) bf_out[slot] = g2;
                            }
                        }
                    }
                }
            }
            __syncwarp();
            large_mask = set_bits(large_mask, 0, large_lane, 1);
        }
        __syncwarp();
    }
    __syncwarp();
    int cnt = __shfl_sync(FULL_MASK, *bf_slot_ptr, 0);
    if (cnt > bf_cap) cnt = bf_cap;
    return cnt;
}
#endif

#if (SEED_BARRIER == true && SEED_PHASE2_PARALLEL == true)
// phase2 发布路径：L3 warp 与注入 warp 按 remote_mark word 做固定分片。
// 每个 word 只有一个发布 warp 负责，word 内的 bit 仍由 lane 顺序消费；因此不需要
// 共享动态收集缓冲，也不需要每个候选对 peer_seed_list_cnt 做 P2P atomicAdd。
//
// seed_pub_busy 记录发布 warp 在途状态。phase2→3 只有在所有发布 warp 完成
// 候选提取、列表写入和 system fence 后才能通过，从而避免越过尚未完成的 P2P 写。
__device__ __forceinline__ void seed_phase2_publish_warp(
    int pub_id, int pub_warp_num, int lane_id, int m,
    int *phase, int *seed_pub_busy,
    int peer_v_begin, int peer_v_local,
    VALUE_TYPE *remote_cand, unsigned *remote_mark,
    unsigned *mark_hint, unsigned *mark_hint2,
    VALUE_TYPE *peer_cache, VALUE_TYPE *peer_node_data,
    NODE_TYPE *seed_list, int *seed_list_cnt)
{
    int phase_now = 0;
    if (!lane_id)
        phase_now = *(volatile int *)phase;
    phase_now = __shfl_sync(FULL_MASK, phase_now, 0);
    if (phase_now != 2)
        return;

    if (!lane_id)
        atomicAdd(seed_pub_busy, 1);
    __syncwarp();

    // phase 可能在登记后被 warp0 改写；登记后再检查，确保不会漏掉在途 warp。
    if (!lane_id)
        phase_now = *(volatile int *)phase;
    phase_now = __shfl_sync(FULL_MASK, phase_now, 0);
    if (phase_now != 2)
    {
        if (!lane_id)
            atomicSub(seed_pub_busy, 1);
        __syncwarp();
        return;
    }

    const int mark_words = (peer_v_local + 31) / 32;
    const int mark_hint_words = (mark_words + 31) / 32;
    const int mark_hint2_words = (mark_hint_words + 31) / 32;
    bool published = false;

    // 固定、不重叠的三级 hint 分片：每个 h2 块只由一个发布 warp 处理，
    // warp 内 32 个 lane 分别处理该块的 32 个 mark_hint word。
    for (int h2 = pub_id; h2 < mark_hint2_words; h2 += pub_warp_num)
    {
        unsigned hv2 = l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint2[h2]);
        if (!hv2) continue;

        const int h0 = h2 * WARP_SIZE;
        for (int h = h0 + lane_id; h < h0 + WARP_SIZE && h < mark_hint_words; h += WARP_SIZE)
        {
            unsigned hv = l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint[h]);
            if (!hv) continue;

            const int w0 = h * WARP_SIZE;
            for (int w = w0; w < w0 + WARP_SIZE && w < mark_words; w++)
            {
                unsigned mv = l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w]);
                if (!mv) continue;
                if (!l3_device_mark_claim(&remote_mark[w], mv)) continue;
                published = true;

                for (int b = 0; b < WARP_SIZE; b++)
                {
                    if (!(mv & (1u << b))) continue;
                    int lidx = w * WARP_SIZE + b + 1;  // peer 局部 1-based
                    if (lidx > peer_v_local) continue;

                    VALUE_TYPE nd = atomicExch(&remote_cand[lidx], DIST_MAX);
                    if (nd == DIST_MAX)
                    {
                        // mark 先于候选被取走的竞争情况：恢复候选信号和两级 hint。
                        unsigned bit = 1u << b;
                        l3_device_mark_publish(&remote_mark[w], bit);
                        atomicOr(&mark_hint[h], 1u << (w & 31));
                        atomicOr(&mark_hint2[h2], 1u << (h & 31));
                        continue;
                    }

                    // peer_cache 保持发送侧单调过滤语义。
#if (BULK_NO_CACHE == false)
                    atomicMin(&peer_cache[lidx], nd);
#endif
#if (SEED_PHASE2_LIST_ONLY == true)
                    // 接收卡在 seed gate 解除前会把紧凑列表物化到本卡 node_data，
                    // 这里省掉一次随机 P2P 权威写。
#else
                    peer_node_data[lidx] = nd;
#endif

                    // seed list 保存在源卡本地，只有本卡 atomicAdd；每个 peer 顶点
                    // 最多保留一个最终候选，列表容量由 m+1 覆盖。
                    int slot = atomicAdd(seed_list_cnt, 1);
                    if (slot < m + 1)
                        seed_list[slot] = node_struct(peer_v_begin + lidx, nd);
                }
            }

            // 并发置位会改变 hint 值，CAS 失败则保留下轮扫描入口。
            bool h_empty = true;
            for (int w = w0; w < w0 + WARP_SIZE && w < mark_words; w++)
                if (l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w]) != 0) { h_empty = false; break; }
            if (h_empty)
                atomicCAS(&mark_hint[h], hv, 0);
        }

        __syncwarp();
        if (!lane_id)
        {
            bool h2_empty = true;
            for (int h = h0; h < h0 + WARP_SIZE && h < mark_hint_words; h++)
                if (l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint[h]) != 0) { h2_empty = false; break; }
            if (h2_empty)
                atomicCAS(&mark_hint2[h2], hv2, 0);
        }
        __syncwarp();
    }

    // seed-list-only 时发布者只写本卡 peer_cache/seed_list，system fence 留到
    // phase2→3 的一次性列表复制前执行；旧双写回退路径仍需每轮 system fence。
    bool any_published = __any_sync(FULL_MASK, published);
    __syncwarp();
#if (SEED_PHASE2_LIST_ONLY == true)
    if (any_published)
        __threadfence_block();
#else
    if (any_published)
        __threadfence_system();
#endif
    __threadfence_block();
    if (!lane_id)
        atomicSub(seed_pub_busy, 1);
    __syncwarp();

    // 避免多个发布 warp 在空轮中紧循环，给 warp0 留出观察 phase2 空闲窗口的机会。
    if (!any_published)
    {
        for (int spin = 0; spin < 128; spin++)
            __threadfence();
    }
}
#endif

#if (BULK_ROUND == true)
// BULK inbox 的 epoch 槽位协议。
// epoch 从 1 开始；同一槽位只会承载相差 BULK_INBOX_SLOTS 的 epoch。
// 因此发送方复用槽位前等待该槽位的旧 epoch ack，接收方只接受严格的
// rx_epoch+1，避免双槽复用时把较新的槽误当成当前消息。
// Slot mapping and ownership primitives live in l3/l3_transport.cuh.

// 发送端尝试取得一个可写槽位。只有发送端能把 DONE 回收为 FREE；接收端在完成
// apply 后才发布 DONE，因此 WRITING/READY/READING 槽位绝不会被覆盖。
//
// 这里必须是非阻塞尝试：L3 warp 不能在 peer 槽位上无限自旋。若接收侧正在
// 消费上一批，发送侧应把本批候选放回 remote_cand/remote_mark，继续让双方的
// manager/work warp 推进；否则两个方向各自等待对端 ACK 时，会把整个持久化
// kernel 锁在一个看似“有活动”的 l3_busy 状态。
// 接收端严格领取 expected generation。READY -> READING 的 CAS 使领取动作
// 幂等；若另一个接收路径已经领取，当前调用必须等待下一次轮询而不能重复 apply。
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
// async receive frontier：从已经由 manager 标记为 READING 的本地 inbox slot
// 领取一批条目。候选列表按远程顶点去重，因此每个条目最多代表一个目标顶点；
// 接收侧先在本卡 node_data 上 atomicMin，只有真正胜出的条目才进入 node_in。
//
// inflight 的生命周期覆盖后续 simple_process：调用者拿到非空 node_in 后，必须
// 在 simple_process 完成后调用 atomicSub(inbox_inflight + claim_slot, 1)。这样
// manager 不会在 work warp 仍使用该 slot 时发布 DONE，发送侧也不会复用槽位。
__device__ __forceinline__ int async_rx_claim_frontier_batch(
    NODE_TYPE *inbox, int *inbox_count, int *inbox_epoch,
    int *inbox_read_head,
    int *inbox_inflight, int *inbox_state, int *inbox_generation,
    int *active_slot_ptr, int v_begin, int v_local,
    VALUE_TYPE *node_data, NODE_TYPE *node_in, int capacity,
    int lane_id, int &claim_slot)
{
    claim_slot = -1;
    if (inbox == NULL || inbox_count == NULL || inbox_epoch == NULL
        || inbox_read_head == NULL
        || inbox_inflight == NULL || inbox_state == NULL
        || inbox_generation == NULL || active_slot_ptr == NULL
        || node_in == NULL || capacity <= 0)
        return 0;

    int slot = -1;
    if (!lane_id)
        slot = atomicAdd(active_slot_ptr, 0);
    slot = __shfl_sync(FULL_MASK, slot, 0);
    if (slot < 0 || slot >= BULK_INBOX_SLOTS)
        return 0;

    int state = atomicAdd(inbox_state + slot, 0);
    int generation = atomicAdd(inbox_generation + slot, 0);
    int published = atomicAdd(inbox_count + slot, 0);
    int epoch = atomicAdd(inbox_epoch + slot, 0);
    // generation 是发送者发布的严格代际标签；count 已在 READY 之前发布，
    // 这里只接受 manager 当前激活的 READING 槽。再次读 state/generation
    // 可覆盖 manager 完成旧槽、发送者复用同槽的边界窗口。
    if (state != BULK_SLOT_READING || generation != epoch)
        return 0;
    if (published < 0)
        published = 0;
    if (published > v_local + 1)
        published = v_local + 1;

    int start = -1;
    if (!lane_id)
    {
        // 先登记 inflight，再复核槽位所有权。manager 可能恰好把 READING
        // 转为 CLOSING；此时本 claim 必须撤销，不能在 manager 已开始回收
        // 槽位后再推进 read_head 或读取 inbox 数据。
        atomicAdd(inbox_inflight + slot, 1);
        __threadfence();
        int state_after = atomicAdd(inbox_state + slot, 0);
        int generation_after = atomicAdd(inbox_generation + slot, 0);
        int epoch_after = atomicAdd(inbox_epoch + slot, 0);
        if (state_after == BULK_SLOT_READING
            && generation_after == generation
            && epoch_after == epoch)
        {
            // manager 看到 read_head 已覆盖 count 时，必然也能看到本次
            // claim 的 inflight，不能提前完成槽位回收。
            start = atomicAdd(inbox_read_head + slot, capacity);
        }
        else
        {
            atomicSub(inbox_inflight + slot, 1);
            __threadfence();
        }
    }
    start = __shfl_sync(FULL_MASK, start, 0);

    if (start < 0)
    {
        __syncwarp();
        return 0;
    }

    if (start >= published)
    {
        if (!lane_id)
            atomicSub(inbox_inflight + slot, 1);
        __syncwarp();
        return 0;
    }

    int count = mlq_min(capacity, published - start);
    unsigned win = 0;
    int win_id = 0;
    VALUE_TYPE win_dist = DIST_MAX;
    for (int t = lane_id; t < count; t += WARP_SIZE)
    {
        NODE_TYPE item = inbox[slot * (v_local + 1) + start + t];
        int local_id = item.id - v_begin;
        if (local_id < 1 || local_id > v_local)
            continue;

        VALUE_TYPE nd = item.get_data();
        VALUE_TYPE old_dist;
#ifdef TYPE_INT
        old_dist = atomicMin(&node_data[local_id], nd);
#else
        old_dist = atomicMin_float(&node_data[local_id], nd);
#endif
        if (nd < old_dist)
        {
            win |= 1u << lane_id;
            win_id = item.id;
            win_dist = nd;
        }
    }

    unsigned win_mask = __ballot_sync(FULL_MASK, win != 0);
    int out_num = count_bit(win_mask);
    int out_pos = count_bit(set_bits(win_mask, 0, lane_id, 32));
    if (win != 0)
        node_in[out_pos] = node_struct(win_id, win_dist);
    __syncwarp();

    if (out_num == 0)
    {
        // 本批条目已经被本卡更小的距离淘汰，不需要进入 simple_process，
        // 因而可以在 helper 内直接释放 inflight。
        if (!lane_id)
            atomicSub(inbox_inflight + slot, 1);
        __syncwarp();
        return 0;
    }

    claim_slot = slot;
    return out_num;
}
#endif

#if (GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true)
// GLOBAL_ROUND 接收侧的直接批量应用路径。
// 旧 BULK_ROUND 是 atomicMin -> dirty_bitmap -> 注入 warp 扫描 -> write_through，
// 对一个已经发布的 inbox 形成重复搬运。manager warp 直接完成 atomicMin，
// 并把真正改进的条目批量写入本地 L2；GLOBAL_ROUND 使用 quiesce 窗口，
// GLOBAL_ROUND_ASYNC 则依赖 atomicMin 与 MLMQ 的并发 MRMW 写入保证。
template <typename QUEUE_TYPE>
__device__ __forceinline__ int global_round_consume_inbox_direct(
    NODE_TYPE *inbox, int *inbox_count, int *inbox_epoch, int *inbox_ack,
    int *inbox_state, int *inbox_generation,
    int &rx_epoch, VALUE_TYPE *node_data, VALUE_TYPE *peer_cache_feedback,
    int v_begin, int v_local,
    NODE_TYPE *batch_buf, int batch_cap, int *apply_count,
    QUEUE_TYPE &mlmq, unsigned *debug_time, int lane_id
#if (GLOBAL_ROUND_FRONTIER == true)
    , NODE_TYPE *frontier, int *frontier_tail,
    int *frontier_append_base, int *frontier_append_count
#endif
    )
{
    if (inbox == NULL || inbox_count == NULL || inbox_epoch == NULL || inbox_ack == NULL
        || inbox_state == NULL || inbox_generation == NULL
        || batch_buf == NULL || batch_cap <= 0 || apply_count == NULL)
        return 0;
#if (GLOBAL_ROUND_FRONTIER == true)
    if (frontier == NULL || frontier_tail == NULL
        || frontier_append_base == NULL || frontier_append_count == NULL)
        return 0;
#endif

    int ready = bulk_inbox_ready_epoch(
        inbox_epoch, inbox_state, inbox_generation, rx_epoch);
    if (ready <= rx_epoch)
        return 0;

    int slot = bulk_inbox_slot(ready);
    bool claimed = false;
    if (!lane_id)
        claimed = bulk_inbox_claim_read(inbox_state, inbox_generation, inbox_epoch,
                                        slot, ready);
    claimed = __shfl_sync(FULL_MASK, claimed, 0);
    if (!claimed)
        return 0;

    NODE_TYPE *slot_inbox = inbox + slot * (v_local + 1);
    int cnt = atomicAdd(inbox_count + slot, 0);
    if (cnt < 0) cnt = 0;
    if (cnt > v_local) cnt = v_local;
#if (GLOBAL_ROUND_FRONTIER == true)
    // append_base/count 属于本次 inbox 消费，和已发布 tail 分离：先读取旧 tail
    // 作为预留区起点，再由所有 lane 通过 append_count 分配槽位。tail 在所有
    // 条目写完并完成 fence 前保持不变，work warp 因而看不到半成品。
    if (!lane_id)
    {
        *frontier_append_base = atomicAdd(frontier_tail, 0);
        *frontier_append_count = 0;
    }
    __syncwarp();
#endif

    int improved_total = 0;
    for (int s0 = 0; s0 < cnt; s0 += batch_cap)
    {
        int n = mlq_min(batch_cap, cnt - s0);
        if (!lane_id)
            atomicExch(apply_count, 0);
        __syncwarp();

        for (int t = lane_id; t < n; t += WARP_SIZE)
        {
            NODE_TYPE item = slot_inbox[s0 + t];
            int local_id = item.id - v_begin;
            if (local_id < 1 || local_id > v_local)
                continue;

            VALUE_TYPE nd = item.get_data();
            VALUE_TYPE old_dist;
#if (GLOBAL_ROUND_DIRECT_STORE == true)
            // manager 只有在所有 work/injection warp ack 后才进入此函数；
            // 此时 node_data 没有并发写者，普通读写避免每条候选的 RMW 延迟。
            old_dist = *((volatile VALUE_TYPE *)&node_data[local_id]);
            bool improved_item = nd < old_dist;
            if (improved_item)
                node_data[local_id] = nd;
#else
#ifdef TYPE_INT
            old_dist = atomicMin(&node_data[local_id], nd);
#else
            old_dist = atomicMin_float(&node_data[local_id], nd);
#endif
            bool improved_item = nd < old_dist;
#endif
            if (improved_item)
            {
#if (GLOBAL_ROUND_FRONTIER == true)
                int base = *frontier_append_base;
                int out = atomicAdd(frontier_append_count, 1);
                int capacity = (base >= 0 && base <= v_local) ? (v_local - base + 1) : 0;
                if (out < capacity)
                    frontier[base + out] = item;
#else
                int out = atomicAdd(apply_count, 1);
                if (out < batch_cap)
                    batch_buf[out] = item;
#endif
            }
#if (GLOBAL_ROUND_PEER_CACHE_FEEDBACK == true)
            // apply 发生在双方 work/injection 已 quiesce 的窗口；发送卡此时不会
            // 并发更新自己的 peer_cache。把接收卡当前权威值反馈回发送卡，令后续
            // round 的候选过滤拥有对端已收敛状态。这里用普通 monotonic store，
            // 避免对每条 inbox 消息再增加一次跨卡 atomic RMW。
            if (peer_cache_feedback != NULL)
            {
                VALUE_TYPE authoritative = improved_item ? nd : old_dist;
                VALUE_TYPE cached = *((volatile VALUE_TYPE *)&peer_cache_feedback[local_id]);
                if (authoritative < cached)
                    peer_cache_feedback[local_id] = authoritative;
            }
#endif
        }

        __syncwarp();
#if (GLOBAL_ROUND_FRONTIER == true)
        // frontier 条目直接交给 work warp；不再把它们写入本地 L2 queue。
        // 此处的 fence 只是保证条目写入先于最终 tail 发布；tail 仍保持旧值。
        __threadfence();
#else
        int write_num = atomicAdd(apply_count, 0);
        if (write_num > 0)
        {
            int wn = write_num;
            mlmq.write_through(batch_buf, wn, 0, 0, lane_id, debug_time);
        }
        __syncwarp();
        improved_total += write_num;
#endif
    }

#if (GLOBAL_ROUND_FRONTIER == true)
    // 只有此处才发布 tail。append_count 可能包含多个 batch 的改进项，发布时
    // 统一截断到 frontier 容量，避免越界条目影响后续 work claim。
    __threadfence();
    __syncwarp();
    if (!lane_id)
    {
        int base = *frontier_append_base;
        int appended = *frontier_append_count;
        int capacity = (base >= 0 && base <= v_local) ? (v_local - base + 1) : 0;
        if (appended < 0) appended = 0;
        if (appended > capacity) appended = capacity;
        atomicExch(frontier_tail, base + appended);
    }
    __threadfence_system();
    __syncwarp();
#endif

    __syncwarp();
    if (!lane_id)
        bulk_inbox_finish_read(inbox_state, inbox_ack, slot, ready);
    __syncwarp();
    rx_epoch = ready;
    return improved_total;
}

#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
// manager-owned receive frontier：manager 领取一个 generation-safe inbox，
// 将真正胜出的候选写入 node_data，并把本卡 ready bitmap 置位；work warp
// 后续通过 collect_dirty_slice 领取这些顶点，避免 manager 直接向 MLMQ
// write_through，也避免在每个 work warp 中维护 inbox claim/inflight 状态。
__device__ __forceinline__ int global_round_consume_inbox_manager_frontier(
    NODE_TYPE *inbox, int *inbox_count, int *inbox_epoch, int *inbox_ack,
    int *inbox_state, int *inbox_generation,
    int &rx_epoch, VALUE_TYPE *node_data, VALUE_TYPE *peer_cache_feedback,
    int v_begin, int v_local, unsigned *ready_bitmap, unsigned *ready_hint,
    int lane_id)
{
    if (inbox == NULL || inbox_count == NULL || inbox_epoch == NULL
        || inbox_ack == NULL || inbox_state == NULL
        || inbox_generation == NULL || node_data == NULL
        || ready_bitmap == NULL || ready_hint == NULL)
        return 0;

    int ready = bulk_inbox_ready_epoch(
        inbox_epoch, inbox_state, inbox_generation, rx_epoch);
    if (ready <= rx_epoch)
        return 0;

    int slot = bulk_inbox_slot(ready);
    bool claimed = false;
    if (!lane_id)
        claimed = bulk_inbox_claim_read(
            inbox_state, inbox_generation, inbox_epoch, slot, ready);
    claimed = __shfl_sync(FULL_MASK, claimed, 0);
    if (!claimed)
        return 0;

    NODE_TYPE *slot_inbox = inbox + slot * (v_local + 1);
    int count = atomicAdd(inbox_count + slot, 0);
    if (count < 0) count = 0;
    if (count > v_local) count = v_local;

    int lane_improved = 0;
    for (int t = lane_id; t < count; t += WARP_SIZE)
    {
        NODE_TYPE item = slot_inbox[t];
        int local_id = item.id - v_begin;
        if (local_id < 1 || local_id > v_local)
            continue;

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
            atomicOr(&ready_bitmap[j >> 5], 1u << (j & 31));
            atomicOr(&ready_hint[j >> 10], 1u << ((j >> 5) & 31));
            lane_improved++;
        }
#if (GLOBAL_ROUND_PEER_CACHE_FEEDBACK == true)
        if (peer_cache_feedback != NULL)
        {
            VALUE_TYPE authoritative = improved_item ? nd : old_dist;
            VALUE_TYPE cached = *((volatile VALUE_TYPE *)&peer_cache_feedback[local_id]);
            if (authoritative < cached)
                peer_cache_feedback[local_id] = authoritative;
        }
#endif
    }

    int improved_total = lane_improved;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        improved_total += __shfl_down_sync(FULL_MASK, improved_total, offset);
    improved_total = __shfl_sync(FULL_MASK, improved_total, 0);
    __threadfence_system();
    if (!lane_id)
        bulk_inbox_finish_read(inbox_state, inbox_ack, slot, ready);
    __syncwarp();
    rx_epoch = ready;
    return improved_total;
}
#endif

#if (GLOBAL_ROUND_PARALLEL_APPLY == true)
// GLOBAL_ROUND_PARALLEL_APPLY：接收侧每个 L2 manager warp 处理 inbox 的不相交
// 条目区间。GLOBAL_ROUND 的发送端已经按 peer 目标顶点聚合，因此同一 round
// 中每个目标至多一条消息；在 quiesce 窗口内可用普通读写替代 atomicMin。
__device__ __forceinline__ void global_round_apply_inbox_warp(
    int request, int apply_warp_id, int apply_warp_total,
    NODE_TYPE *inbox, int *inbox_count,
    VALUE_TYPE *node_data, VALUE_TYPE *peer_cache_feedback,
    int v_begin, int v_local,
    NODE_TYPE *frontier, int *frontier_append_base,
    int *frontier_append_count, int *apply_done, int lane_id)
{
    if (inbox == NULL || inbox_count == NULL || node_data == NULL
        || frontier == NULL || frontier_append_base == NULL
        || frontier_append_count == NULL || apply_done == NULL
        || apply_warp_total <= 0)
        return;

    int slot = bulk_inbox_slot(request);
    NODE_TYPE *slot_inbox = inbox + slot * (v_local + 1);
    int count = 0;
    if (!lane_id)
        count = atomicAdd(inbox_count + slot, 0);
    count = __shfl_sync(FULL_MASK, count, 0);
    if (count < 0) count = 0;
    if (count > v_local) count = v_local;

    int base = atomicAdd(frontier_append_base, 0);
    int capacity = (base >= 0 && base <= v_local) ? (v_local - base + 1) : 0;
    int stride = apply_warp_total * WARP_SIZE;
    for (int t = apply_warp_id * WARP_SIZE + lane_id; t < count; t += stride)
    {
        NODE_TYPE item = slot_inbox[t];
        int local_id = item.id - v_begin;
        if (local_id < 1 || local_id > v_local)
            continue;

        VALUE_TYPE nd = item.get_data();
        VALUE_TYPE old_dist = *((volatile VALUE_TYPE *)&node_data[local_id]);
        if (nd < old_dist)
        {
            node_data[local_id] = nd;
            int out = atomicAdd(frontier_append_count, 1);
            if (out < capacity)
                frontier[base + out] = item;
        }
#if (GLOBAL_ROUND_PEER_CACHE_FEEDBACK == true)
        // 与 serial apply 相同：发送卡 work 在本 round apply 期间保持 quiesce，
        // 因而可用单调普通写把当前权威值反馈给发送卡 cache。
        if (peer_cache_feedback != NULL)
        {
            VALUE_TYPE authoritative = (nd < old_dist) ? nd : old_dist;
            VALUE_TYPE cached = *((volatile VALUE_TYPE *)&peer_cache_feedback[local_id]);
            if (authoritative < cached)
                peer_cache_feedback[local_id] = authoritative;
        }
#endif
    }

    // done 计数发布前，保证本 warp 的 node_data/frontier 写入对 work block 可见。
    __syncwarp();
    __threadfence();
    if (!lane_id)
        atomicAdd(apply_done, 1);
}
#endif
#endif

#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
// GLOBAL_ROUND dense 发送侧：quiesce 后把 remote_cand 的完整数组搬到对端
// staging。remote_cand 每个对端顶点至多保留一个 atomicMin 结果，因此接收侧
// 不需要传 id，只按本地局部索引扫描。epoch/count/ack 仍复用 sparse inbox 的
// 发布元数据；数据 -> count -> epoch 的顺序由 system fence 保证。
__device__ __forceinline__ bool global_round_publish_dense(
    int epoch, int lane_id, int peer_v_local,
    VALUE_TYPE *remote_cand, unsigned *remote_mark,
    unsigned *mark_hint, unsigned *mark_hint2,
    VALUE_TYPE *peer_dense_inbox, VALUE_TYPE *peer_cache,
    int *peer_inbox_count, int *peer_inbox_epoch, int *peer_inbox_ack,
    int *peer_inbox_state, int *peer_inbox_generation,
    int *send_count, volatile int *publish_done)
{
    if (remote_cand == NULL || peer_dense_inbox == NULL
        || peer_inbox_count == NULL || peer_inbox_epoch == NULL
        || peer_inbox_ack == NULL || peer_inbox_state == NULL
        || peer_inbox_generation == NULL || send_count == NULL
        || publish_done == NULL)
        return false;

    int slot = bulk_inbox_slot(epoch);
    bool claimed = false;
    if (!lane_id)
        claimed = bulk_inbox_try_acquire_write(
            peer_inbox_state, peer_inbox_generation, peer_inbox_ack,
            slot, epoch);
    claimed = __shfl_sync(FULL_MASK, claimed, 0);
    if (!claimed)
        return false;

    int lane_count = 0;
    for (int lidx = lane_id + 1; lidx <= peer_v_local; lidx += WARP_SIZE)
    {
        VALUE_TYPE nd = atomicExch(&remote_cand[lidx], DIST_MAX);
        peer_dense_inbox[lidx] = nd;
        if (nd != DIST_MAX)
        {
            lane_count++;
#if (BULK_NO_CACHE == false)
            if (peer_cache != NULL)
                atomicMin(&peer_cache[lidx], nd);
#endif
        }
    }

    // Dense 路径不再消费 mark；清零仅用于保持诊断/回退状态一致，不能在
    // producer 尚未 quiesce 时调用本函数。
    if (remote_mark != NULL)
    {
        int mark_words = (peer_v_local + 31) / 32;
        for (int w = lane_id; w < mark_words; w += WARP_SIZE)
            l3_atomic_exchange_acq_rel<cuda::thread_scope_device>(
                &remote_mark[w], 0u);
    }
    if (mark_hint != NULL)
    {
        int mark_words = (peer_v_local + 31) / 32;
        int hint_words = (mark_words + 31) / 32;
        for (int w = lane_id; w < hint_words; w += WARP_SIZE)
            atomicExch(&mark_hint[w], 0u);
    }
    if (mark_hint2 != NULL)
    {
        int mark_words = (peer_v_local + 31) / 32;
        int hint_words = (mark_words + 31) / 32;
        int hint2_words = (hint_words + 31) / 32;
        for (int w = lane_id; w < hint2_words; w += WARP_SIZE)
            atomicExch(&mark_hint2[w], 0u);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        lane_count += __shfl_down_sync(FULL_MASK, lane_count, offset);
    int count = __shfl_sync(FULL_MASK, lane_count, 0);

    __syncwarp();
    __threadfence_system();
    __syncwarp();
    if (!lane_id)
        l3_atomic_store_relaxed<cuda::thread_scope_system>(peer_inbox_count + slot, count);
    __threadfence_system();
    if (!lane_id)
        l3_atomic_store_relaxed<cuda::thread_scope_system>(peer_inbox_epoch + slot, epoch);
    __threadfence_system();
    if (!lane_id)
        l3_atomic_store_release<cuda::thread_scope_system>(peer_inbox_state + slot, BULK_SLOT_READY);
    __threadfence_system();
    if (!lane_id)
    {
        atomicExch((int *)send_count, count);
        l3_atomic_store_release<cuda::thread_scope_block>(
            (int *)publish_done, epoch);
    }
    __syncwarp();
    return true;
}

// GLOBAL_ROUND dense 接收侧：各 manager warp 处理不相交的目标顶点区间。
// sender 在发布 epoch 前已经写完整数组并 fence；work/injection 在 apply
// 窗口内保持 quiesce，因此 direct store 不会和本地 producer 冲突。
__device__ __forceinline__ void global_round_apply_dense_warp(
    int request, int apply_warp_id, int apply_warp_total,
    VALUE_TYPE *dense_inbox, int *inbox_count,
    VALUE_TYPE *node_data, int v_begin, int v_local,
    NODE_TYPE *frontier, int *frontier_append_base,
    int *frontier_append_count, int *apply_done, int lane_id)
{
    if (dense_inbox == NULL || inbox_count == NULL || node_data == NULL
        || frontier == NULL || frontier_append_base == NULL
        || frontier_append_count == NULL || apply_done == NULL
        || apply_warp_total <= 0)
        return;

    int slot = bulk_inbox_slot(request);
    int count = 0;
    if (!lane_id)
        count = atomicAdd(inbox_count + slot, 0);
    count = __shfl_sync(FULL_MASK, count, 0);
    if (count < 0) count = 0;

    int base = atomicAdd(frontier_append_base, 0);
    int capacity = (base >= 0 && base <= v_local) ? (v_local - base + 1) : 0;
    int stride = apply_warp_total * WARP_SIZE;
    for (int lidx = apply_warp_id * WARP_SIZE + lane_id + 1;
         lidx <= v_local; lidx += stride)
    {
        VALUE_TYPE nd = *((volatile VALUE_TYPE *)&dense_inbox[lidx]);
        if (nd == DIST_MAX)
            continue;

        VALUE_TYPE old_dist = *((volatile VALUE_TYPE *)&node_data[lidx]);
        if (nd < old_dist)
        {
            node_data[lidx] = nd;
            int out = atomicAdd(frontier_append_count, 1);
            if (out < capacity)
                frontier[base + out] = node_struct(v_begin + lidx, nd);
        }
    }

    __syncwarp();
    __threadfence();
    if (!lane_id)
        atomicAdd(apply_done, 1);
}
#endif

#if (GLOBAL_ROUND_MULTI_PACK == true)
// GLOBAL_ROUND_MULTI_PACK 的最终发布阶段。remote_mark/remote_cand 已由多个
// manager warp 收集到 send_list；所有 worker 分片复制 inbox，worker 0 只负责
// 槽位 ACK 和 data -> count -> epoch 的最终发布。
__device__ __forceinline__ bool global_round_publish_collected_multi(
    int epoch, int worker_id, int worker_total, int lane_id,
    int peer_v_local, NODE_TYPE *send_list, NODE_TYPE *peer_inbox,
    int *peer_inbox_count, int *peer_inbox_epoch, int *peer_inbox_ack,
    int *peer_inbox_state, int *peer_inbox_generation,
    int *send_count, volatile int *worker_done, volatile int *copy_start,
    int *copy_done, volatile int *publish_done)
{
    if (worker_id < 0 || worker_id >= worker_total || send_list == NULL
        || peer_inbox == NULL || peer_inbox_count == NULL
        || peer_inbox_epoch == NULL || peer_inbox_ack == NULL
        || peer_inbox_state == NULL || peer_inbox_generation == NULL
        || send_count == NULL || worker_done == NULL || copy_start == NULL
        || copy_done == NULL || publish_done == NULL)
        return false;

    while (l3_atomic_load_acquire<cuda::thread_scope_block>((int *)worker_done) < worker_total)
        __threadfence();

    int slot = bulk_inbox_slot(epoch);
    if (worker_id == 0)
    {
        bool claimed = false;
        if (!lane_id)
            claimed = bulk_inbox_try_acquire_write(
                peer_inbox_state, peer_inbox_generation, peer_inbox_ack,
                slot, epoch);
        claimed = __shfl_sync(FULL_MASK, claimed, 0);
        if (!claimed)
            return false;

        if (!lane_id)
        {
            atomicExch(copy_done, 0);
            __threadfence();
            l3_atomic_store_release<cuda::thread_scope_block>((int *)copy_start, epoch);
        }
    }

    // worker 0 发布 copy_start 后，所有 worker 才能读取 send_list。worker_done
    // 的递增位于每个 worker 的 __threadfence 之后，保证候选条目已可见。
    while (l3_atomic_load_acquire<cuda::thread_scope_block>((int *)copy_start) < epoch)
        __threadfence();
    __threadfence();

    int cnt = atomicAdd(send_count, 0);
    if (cnt < 0) cnt = 0;
    if (cnt > peer_v_local) cnt = peer_v_local;

    NODE_TYPE *slot_inbox = peer_inbox + slot * (peer_v_local + 1);
    int global_lane = worker_id * WARP_SIZE + lane_id;
    int global_stride = worker_total * WARP_SIZE;
    for (int i = global_lane; i < cnt; i += global_stride)
        slot_inbox[i] = send_list[i];
    __syncwarp();
    __threadfence_system();

    if (!lane_id)
        l3_atomic_fetch_add_acq_rel<cuda::thread_scope_block>(copy_done, 1);
    while (l3_atomic_load_acquire<cuda::thread_scope_block>(copy_done) < worker_total)
        __threadfence();

    if (worker_id == 0)
    {
        __threadfence_system();
        if (!lane_id)
            l3_atomic_store_relaxed<cuda::thread_scope_system>(peer_inbox_count + slot, cnt);
        __threadfence_system();
        if (!lane_id)
            l3_atomic_store_relaxed<cuda::thread_scope_system>(peer_inbox_epoch + slot, epoch);
        __threadfence_system();
        if (!lane_id)
            l3_atomic_store_release<cuda::thread_scope_system>(peer_inbox_state + slot, BULK_SLOT_READY);
        __threadfence_system();
        if (!lane_id)
        {
            atomicExch(send_count, cnt);
            l3_atomic_store_release<cuda::thread_scope_block>(
                (int *)publish_done, epoch);
        }
    }
    __syncwarp();
    return true;
}

// 每个参与 worker 是一个完整 warp。remote_mark 的 word 按全局 warp-lane
// 编号切分，remote_cand 仍按顶点 atomicExch 取走，因此每个顶点最多进入
// send_list 一次；worker 0 等所有 worker 完成后负责最终 publish。
__device__ __forceinline__ bool global_round_multi_pack_worker(
    int epoch, int worker_id, int worker_total, int lane_id,
    int peer_v_begin, int peer_v_local,
    VALUE_TYPE *remote_cand, unsigned *remote_mark,
    unsigned *mark_hint, unsigned *mark_hint2,
    VALUE_TYPE *peer_node_data, NODE_TYPE *send_list,
    NODE_TYPE *peer_inbox, int *peer_inbox_count,
    int *peer_inbox_epoch, int *peer_inbox_ack,
    int *peer_inbox_state, int *peer_inbox_generation,
    int *send_count, volatile int *worker_done,
    volatile int *copy_start, int *copy_done,
    volatile int *publish_done)
{
    if (worker_total <= 0 || worker_id < 0 || worker_id >= worker_total
        || remote_cand == NULL || remote_mark == NULL || send_list == NULL
        || worker_done == NULL || copy_start == NULL || copy_done == NULL
        || publish_done == NULL)
        return false;

    int mark_words = (peer_v_local + 31) / 32;
    int global_lane = worker_id * WARP_SIZE + lane_id;
    int global_stride = worker_total * WARP_SIZE;
    for (int w = global_lane; w < mark_words; w += global_stride)
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
#if (GLOBAL_ROUND_CANDIDATE_PREFILTER == true)
            // node_data 只会单调下降。此处即使读到对端较旧的较大值，也只会
            // 保留冗余候选；读到不大的值时丢弃是安全的，避免发布阶段再串行
            // 扫描 send_list。
            if (peer_node_data != NULL && epoch > 1)
            {
                VALUE_TYPE peer_dist =
                    l3_atomic_load_relaxed<cuda::thread_scope_system>(
                        &peer_node_data[lidx]);
                if (nd >= peer_dist)
                    continue;
            }
#endif
            int out = atomicAdd(send_count, 1);
            if (out < peer_v_local)
                send_list[out] = node_struct(peer_v_begin + lidx, nd);
        }
    }

    int mark_hint_words = (mark_words + 31) / 32;
    for (int h = global_lane; h < mark_hint_words; h += global_stride)
    {
        if (mark_hint != NULL)
            atomicExch(&mark_hint[h], 0u);
    }
    int mark_hint2_words = (mark_hint_words + 31) / 32;
    for (int h = global_lane; h < mark_hint2_words; h += global_stride)
    {
        if (mark_hint2 != NULL)
            atomicExch(&mark_hint2[h], 0u);
    }

    __syncwarp();
    __threadfence();
    if (!lane_id)
        l3_atomic_fetch_add_acq_rel<cuda::thread_scope_block>((int *)worker_done, 1);

    bool published = global_round_publish_collected_multi(
        epoch, worker_id, worker_total, lane_id, peer_v_local,
        send_list, peer_inbox, peer_inbox_count, peer_inbox_epoch,
        peer_inbox_ack, peer_inbox_state, peer_inbox_generation,
        send_count, worker_done, copy_start, copy_done,
        publish_done);
    if (!published && !lane_id)
        l3_atomic_store_release<cuda::thread_scope_block>(
            (int *)publish_done, epoch);
    return published;
}
#endif

#endif

#if (BULK_FRONTIER_ENABLED == true)
#define BULK_FRONTIER_CONSUME_ARGS \
    , bulk_frontier, bulk_frontier_head, bulk_frontier_tail, &bulk_frontier_append_base, &bulk_frontier_append_count
#else
#define BULK_FRONTIER_CONSUME_ARGS
#endif

template <typename QUEUE_TYPE>
// E3: launch_bounds 约束 768 线程（L2DQ manage_warp_num=16 → 32*(1+16+1+6)=768）。
//   A100 SM regfile 65536/768=85 → ptxas 在 85 regs 内调度（必要时 spill），
//   防 hint 两级扫描 + 各队列实例化寄存器超限导致 launch 失败。
__global__ void __launch_bounds__(768) manage_block_kernel(int m, int nnz, int *RowPtr, int *ColIdx, VALUE_TYPE *edge_data, int src, QUEUE_TYPE mlmq, mlmq_setup setup,
    int qshm_size, int nshm_size, int rshm_size, int *global_exit,
    int v_begin, int v_end, int v_local,
    VALUE_TYPE *node_data, unsigned *dirty_bitmap, unsigned *dirty_hint, VALUE_TYPE *last_processed,
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
    unsigned *async_rx_ready_bitmap, unsigned *async_rx_ready_hint,
#endif
    int *local_idle, l3_channel_view l3_channel,
    int *phase, int *seed_ready, int *peer_seed_ready, int *seed_inject_done,
    NODE_TYPE *seed_list, int *seed_list_cnt, NODE_TYPE *peer_seed_list, int *peer_seed_list_cnt,
#if (GLOBAL_ROUND_ASYNC == true)
    async_candidate_bank cand_bank0, async_candidate_bank cand_bank1,
    async_candidate_control cand_ctl,
#endif
#if (GHOST_DEPTH > 0)
    int ghost_num, int *ghost_row_start, int *ghost_col, VALUE_TYPE *ghost_edge_data,
    VALUE_TYPE *ghost_node_data, int *ghost_id_to_idx, unsigned *ghost_mark,
#endif
    work_count_type *global_work_count, int *global_comp_count,
    int n_gpu,
#if (L3_RX_EXPRESS == true)
    int rx_express_enabled,
#endif
#if (BULK_ROUND == true)
    int bulk_work_warp_total,
#endif
    unsigned long long *mgmt_profile)
{
    int local_tid = threadIdx.x;
    int local_wid = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;

    // CP2a compatibility aliases: all existing code below keeps its current
    // data flow while the kernel ABI now receives one explicit channel view.
    int *peer_local_idle = l3_channel.peer_local_idle;
    VALUE_TYPE *peer_node_data = l3_channel.peer_node_data;
    unsigned *peer_dirty_bitmap = l3_channel.peer_dirty_bitmap;
    unsigned *peer_dirty_hint = l3_channel.peer_dirty_hint;
    int peer_v_begin = l3_channel.peer_v_begin;
    int peer_v_local = l3_channel.peer_v_local;
    VALUE_TYPE *remote_cand = l3_channel.candidate_values;
    unsigned *remote_mark = l3_channel.candidate_mark;
    unsigned *mark_hint = l3_channel.candidate_hint;
    unsigned *mark_hint2 = l3_channel.candidate_hint2;
    VALUE_TYPE *peer_cache = l3_channel.peer_cache;
    VALUE_TYPE *peer_cache_feedback = l3_channel.peer_cache_feedback;
    unsigned long long *remote_eff = l3_channel.remote_eff;
    unsigned long long *peer_remote_eff = l3_channel.peer_remote_eff;
#if (BULK_ROUND == true)
    int *bulk_quiesce_req = l3_channel.quiesce_req;
    int *bulk_quiesce_ack = l3_channel.quiesce_ack;
    int *l3_term_req = l3_channel.term_req;
    int *l3_term_state = l3_channel.term_state;
    int *l3_term_ack_slots = l3_channel.term_ack_slots;
    int *peer_l3_term_state = l3_channel.peer_term_state;
    NODE_TYPE *bulk_send_list = l3_channel.send_list;
    NODE_TYPE *bulk_inbox = l3_channel.rx_payload;
    int *bulk_inbox_count = l3_channel.rx_count;
    int *bulk_inbox_epoch = l3_channel.rx_epoch;
    int *bulk_inbox_ack = l3_channel.rx_ack;
    int *bulk_inbox_state = l3_channel.rx_state;
    int *bulk_inbox_generation = l3_channel.rx_generation;
    NODE_TYPE *peer_bulk_inbox = l3_channel.tx_payload;
    int *peer_bulk_inbox_count = l3_channel.tx_count;
    int *peer_bulk_inbox_epoch = l3_channel.tx_epoch;
    int *peer_bulk_inbox_ack = l3_channel.tx_ack;
    int *peer_bulk_inbox_state = l3_channel.tx_state;
    int *peer_bulk_inbox_generation = l3_channel.tx_generation;
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
    int *bulk_inbox_read_head = l3_channel.rx_read_head;
    int *bulk_inbox_inflight = l3_channel.rx_inflight;
    int *bulk_inbox_active_slot = l3_channel.rx_active_slot;
#endif
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
    VALUE_TYPE *bulk_dense_inbox = l3_channel.dense_rx_payload;
    VALUE_TYPE *peer_bulk_dense_inbox = l3_channel.dense_tx_payload;
#endif
#if (BULK_FRONTIER_ENABLED == true)
    NODE_TYPE *bulk_frontier = l3_channel.rx_frontier;
    int *bulk_frontier_head = l3_channel.rx_frontier_head;
    int *bulk_frontier_tail = l3_channel.rx_frontier_tail;
#endif
#endif
#if (L3_RX_EXPRESS == true)
    l3_rx_express_ring rx_express = l3_channel.rx_express;
#endif

    __shared__ int manager_end;
#if (L3_ADMISSION_BUDGET == true)
    unsigned long long admission_last=0;
#endif
    // v3: L3 flush 在途标志（warp0 终止检测读 / L3 flush warp 写，block 内共享）。
    // phase2 并行发布时旧标志仍保留给 phase3，另用计数记录发布 warp 在途。
    __shared__ int l3_busy;
    __shared__ int seed_pub_busy;
    // E2: 注入在途计数（注入 warp write_through 前 +n / 后 -n，warp0 终止检测读）
    __shared__ int inj_pending;
    // E2: peer idle 结果（lane0 读 P2P 后广播到全 warp，二次确认用全 lane 并行扫描）
    __shared__ volatile int peer_idle_s;
    // H1: backstop 协作全扫（warp0 请求 / 注入 warp 分担，见 backstop_collaborative）
    __shared__ volatile int bs_req;
    __shared__ int bs_done;
    __shared__ int bs_found;
#if (L3_ACK_SCAN == true)
    __shared__ int ack_scan_checked_token;
#endif
#if (L3_RECOVERY_MODE > 0)
    __shared__ int bs_epoch;
#endif
    // ASYNC_FB（方案 B）: 终止 quiet 窗口状态（warp0 专用：eff 计数快照 + 连续稳定轮数）
    __shared__ volatile unsigned long long s_eff_base;
    __shared__ volatile unsigned long long s_peer_eff_base;
    __shared__ volatile int s_eff_quiet;
#if (BULK_ROUND == true)
    // BULK_ROUND：warp0 请求 work 静止；L3 warp 完成列表打包/发布后回写 done。
    __shared__ volatile int bulk_pack_req;
    __shared__ volatile int bulk_publish_done;
    __shared__ volatile int bulk_pack_worker_done;
    // GLOBAL_ROUND_MULTI_PACK：候选提取完成后，worker 0 发布复制开始信号；
    // 所有 worker 分片复制 inbox，copy_done 汇合后再发布 count/epoch。
    __shared__ volatile int bulk_pack_copy_start;
    __shared__ int bulk_pack_copy_done;
    __shared__ int bulk_send_count;
    __shared__ int bulk_apply_count;
    // 默认 BULK 终止握手：每个 manage injection warp 写自己的 token ACK，
    // 避免共享计数在取消/重试时出现旧 ACK 混入新请求。
    __shared__ int l3_term_inject_ack[INJECT_WARP_NUM];
    // TX must own no journal and stop claiming marks before final inspection.
    __shared__ int l3_term_tx_ack;
    __shared__ int l3_drain_requested;
    // GLOBAL_ROUND：注入 warp 也必须先到安全点，避免 manager 看到 inj_pending=0
    // 的瞬间，某个注入 warp 仍在收集 dirty 并随后写入 L2/产生下一轮候选。
    __shared__ int bulk_inject_ack;
#if (GLOBAL_ROUND_PARALLEL_APPLY == true)
    // GLOBAL_ROUND_PARALLEL_APPLY：L2 manager warps 的 round apply 请求与完成数。
    __shared__ volatile int global_apply_req;
    __shared__ int global_apply_done;
#endif
    // 通用路径：本卡最后一个已发布 epoch。终止前必须确认对端 ack 到该 epoch。
    __shared__ volatile int bulk_tx_epoch_published;
#if (BULK_FRONTIER_ENABLED == true)
    // manager warp 在一次 inbox 消费中先压实 frontier，再发布 tail。
    __shared__ int bulk_frontier_append_base;
    __shared__ int bulk_frontier_append_count;
#endif
#endif

    if (!local_wid && !lane_id)
    {
#if (L3_PROGRESS_DIAG == true)
        g_l3_progress.start = clock64();
#endif
        l3_atomic_store_release<cuda::thread_scope_block>(&manager_end, 0);
        l3_busy = 0;
        seed_pub_busy = 0;
        inj_pending = 0;
        peer_idle_s = 0;
        bs_req = 0;
        bs_done = 0;
        bs_found = 0;
#if (L3_ACK_SCAN == true)
        ack_scan_checked_token = 0;
#endif
#if (L3_RECOVERY_MODE > 0)
        bs_epoch = 0;
#endif
        s_eff_base = 0;
        s_peer_eff_base = 0;
        s_eff_quiet = 0;
#if (BULK_ROUND == true)
        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_req, 0);
        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_publish_done, 0);
        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_worker_done, 0);
        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_copy_start, 0);
        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_copy_done, 0);
        bulk_send_count = 0;
        bulk_apply_count = 0;
        for (int i = 0; i < INJECT_WARP_NUM; i++)
            l3_atomic_store_release<cuda::thread_scope_block>(
                l3_term_inject_ack + i, 0);
        l3_atomic_store_release<cuda::thread_scope_block>(&l3_term_tx_ack, 0);
        l3_atomic_store_release<cuda::thread_scope_block>(&l3_drain_requested, 0);
        l3_atomic_store_release<cuda::thread_scope_block>(&bulk_inject_ack, 0);
#if (GLOBAL_ROUND_PARALLEL_APPLY == true)
        global_apply_req = 0;
        global_apply_done = 0;
#endif
        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published, 0);
#if (BULK_FRONTIER_ENABLED == true)
        bulk_frontier_append_base = 0;
        bulk_frontier_append_count = 0;
#endif
        if (bulk_quiesce_req != NULL) l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 0);
        if (bulk_quiesce_ack != NULL) l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
#endif
        while (!*(mlmq.run_begin))
        { __threadfence(); }
    }
    __syncwarp();

    __syncthreads();

    if (local_wid == 0)
    {
        // === 注入 + 终止 warp（wid=0，复用 work 的共享内存布局） ===
        extern __shared__ int s[];
        // init_device 只初始化 L1/shared 状态；manage 在 n=1 时不需要 L1，
        // 但注入路径需要（后续用到时再初始化）。这里统一初始化以支持注入。
        mlmq.init_device(0, 0, lane_id, setup);

        NODE_TYPE *node_buf = (NODE_TYPE*)(s + qshm_size / sizeof(int));
        // E2: node_in 串行 lane0 收集，容量 node_size。warp0 用 [0,node_size)，
        //   注入 warp i 用 [(1+i)*node_size, ...)。node_out 预留区（未使用）上限 1024。
        NODE_TYPE *node_in = node_buf;                                  // wid=0
        NODE_TYPE *node_out = node_buf + 2 * node_size * WARP_NUM_PER_BLOCK;

    int node_in_num = 0;
        int total_work = 0;
        int total_comp = 0;
        unsigned debug_time[3] = {0, 0, 0};
        int dwords = (v_local + 31) / 32;
        // E3: 注入区间切片按 hint-word（32 个 dirty word 一块）切分，保证块对齐
        //   （hint word 单消费者，§16.5 风险2）。消费者 = warp0(slice0) + INJECT_WARP_NUM 个注入 warp。
        //   剩余 hint word 分给前 h_rem 个消费者，小图 hwords<7 时 warp0 至少覆盖 word0。
#if (BULK_ROUND == true)
        // BULK_ROUND 将 dirty 的全部切片交给注入 warp；manager warp0 只负责
        // epoch/终止协议，不直接并发写 delta queue。
        int ncons = INJECT_WARP_NUM;
#else
        int ncons = 1 + INJECT_WARP_NUM;
#endif
        int hwords = (dwords + 31) / 32;
        int h_slice = hwords / ncons;
        int h_rem = hwords % ncons;
        int hb_beg = 0;
        int hb_end = h_slice + (h_rem > 0 ? 1 : 0);
        int c_beg = hb_beg * 32;
        int c_end = hb_end * 32;
        // E1a: backstop 全区间扫描降频（每 BACKSTOP_K 轮扫一次，终止前强制补扫）
        // E4b: 16 → 64（del_n23 w0 backstop 94% 瓶颈。正确性不依赖 K：终止前
        //   pre_ready 强制补扫 + 二次确认无条件全扫兜底）
        // E5: 64 → 256（E4 复查: del_n23 w0 backstop 59.5s/120s=78%，969 次×61ms。
        //   降频 → 243 次，预计 ~15s + w17 L2 争用联动收益）
        // E5b: 256 → 512（实测 K=256 仍超时：轮率 1457/s × ~50 万轮，backstop=轮数/K×61ms
        //   =131s(52%)。提 K → backstop 预期 ~65s）
#if (ASYNC_FB == true)
        // B 开销优化: 接收卡 backstop 全扫降频 512→2048（B 模式终止由 pre_ready+confirm 兜底）
        const int BACKSTOP_K = 2048;
#else
        const int BACKSTOP_K = 512;
#endif
            int backstop_round = 0;
#if (L3_BOUNDED_RECOVERY == true)
            l3_bounded_scan_cursor recovery_cursor;
            // At most 512 KiB logical dist/processed reads per active service.
            // Fixed budget; no graph-name policy or per-graph tuning.
            const int RECOVERY_VERTEX_BUDGET = 65536;
#endif
            l3_confirm_policy confirm_policy;
#if (BULK_ROUND == true)
        // 接收侧 epoch 游标由 warp0 维护；L3 发布后先写 inbox，再写 epoch，
        // warp0 只负责 atomicMin+dirty，注入 warp 负责真正写入本地 L2。
        int bulk_rx_epoch = 0;
#endif
#if (SEED_BARRIER == true)
        int wait_round = 0;   // 等待期 l2_empty 轮检查降频计数（小图 overhead 优化）
#endif
#if (SEED_BARRIER == true)
#endif
        // HANG_DIAG 独立计数（不依赖 MANAGE_PROFILE）
        unsigned diag_iter = 0;
#if (HANG_DIAG == true)
        bool diag_inbox_seen = false;
        bool diag_term_seen = false;
        bool diag_loop_seen = false;
#endif

#if (MANAGE_PROFILE == true)
        // V0.5 profile: 注入 / backstop / 终止检查 三环节 clock() 差分累积
        unsigned long long p_inject = 0, p_backstop = 0, p_term = 0, p_total = 0;
        unsigned p_iter = 0;
        // 方案2(P2) 细分：mark_empty 全扫 / pre_ready+二次确认 backstop 全扫
        unsigned long long p_mark = 0, p_confirm = 0, p_dirty = 0;
        unsigned p_confirm_cnt = 0, p_pre_cnt = 0;
#endif

#if (BULK_ROUND == true && GLOBAL_ROUND == true)
        // GLOBAL_ROUND：只在本卡真正到达本地安全点后启动一轮。与旧 BULK_ROUND
        // 每个 L3_BATCH 都发布不同，这里把整个 remote_cand/remote_mark 生命周期
        // 收拢到一次 full pack，避免中间距离被对端反复注入。
        if (n_gpu > 1)
        {
            int global_round = 0;
            int global_active_round = 0;
            bool global_pack_armed = false;
#if (GLOBAL_ROUND_DIAG == true)
            unsigned global_round_diag = 0;
#endif
#if (GLOBAL_ROUND_PROFILE == true)
            timeline_clock_type gr_local_start = 0;
            timeline_clock_type gr_quiesce_start = 0;
            timeline_clock_type gr_pack_start = 0;
            timeline_clock_type gr_inbox_start = 0;
            timeline_clock_type gr_apply_start = 0;
            timeline_clock_type gr_release_start = 0;
            bool gr_local_wait_active = false;
            bool gr_pack_profiled = false;
            bool gr_inbox_profiled = false;
#endif

            while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
            {
#if (GLOBAL_ROUND_PROFILE == true)
                if (!lane_id && global_active_round == 0 && !gr_local_wait_active)
                {
                    gr_local_start = TIMELINE_CLOCK();
                    gr_local_wait_active = true;
                }
#endif
                bool local_empty = false;
                if (global_active_round == 0)
                {
                    int q_now = mlmq.get_global_queue_size();
                    q_now = __shfl_sync(FULL_MASK, q_now, 0);
                    int inj_now = atomicAdd(&inj_pending, 0);
                    bool d_lane_empty = true;
                    int dwords = (v_local + 31) / 32;
                    bool frontier_empty = true;
#if (GLOBAL_ROUND_FRONTIER == true)
                    frontier_empty = bulk_frontier_empty(
                        bulk_frontier_head, bulk_frontier_tail, lane_id);
#endif
                    if (q_now == 0 && inj_now == 0 && node_in_num == 0
                        && frontier_empty)
                    {
                        for (int w = lane_id; w < dwords; w += WARP_SIZE)
                        {
                            if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w]) != 0)
                                d_lane_empty = false;
                        }
                        unsigned dmask = __ballot_sync(FULL_MASK, d_lane_empty);
                        local_empty = (dmask == FULL_MASK);
                    }
                }

                if (global_active_round == 0)
                {
                    if (!local_empty)
                    {
                        __threadfence();
                        continue;
                    }

                    global_active_round = global_round + 1;
                    global_pack_armed = false;
                    if (!lane_id)
                    {
#if (GLOBAL_ROUND_PROFILE == true)
                        global_round_profile_add(&g_gr_local_empty,
                                                 &g_gr_local_empty_max,
                                                 gr_local_start, TIMELINE_CLOCK());
                        gr_local_wait_active = false;
                        gr_quiesce_start = TIMELINE_CLOCK();
                        gr_pack_profiled = false;
                        gr_inbox_profiled = false;
#endif
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                        l3_atomic_store_release<cuda::thread_scope_block>(&bulk_inject_ack, 0);
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 1);
                        timeline_idle_write_reason(local_idle, 0, 20);
                    }
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }

                // work warp 和 injection warp 都到安全点后，才允许 L3 full pack。
                int work_ack = l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_ack);
                int inject_ack = l3_atomic_load_acquire<cuda::thread_scope_block>(&bulk_inject_ack);
                int pending = atomicAdd(&inj_pending, 0);
                if (!global_pack_armed)
                {
                    if (work_ack < bulk_work_warp_total
                        || inject_ack < INJECT_WARP_NUM || pending != 0)
                    {
                        __threadfence();
                        continue;
                    }

#if (GLOBAL_ROUND_FRONTIER == true)
                    // 不能在 request 刚置位时 reset：work warp 可能已经读到旧
                    // head/tail，尚未完成 CAS。等全部 work/injection warp ack 后，
                    // 所有 claim 都已结束，再复位 frontier，避免迟到 CAS 在新 round
                    // 中重新成功领取旧槽位。
                    bulk_frontier_reset(bulk_frontier_head, bulk_frontier_tail, lane_id);
#endif
                    if (!lane_id)
                    {
#if (GLOBAL_ROUND_PROFILE == true)
                        global_round_profile_add(&g_gr_quiesce,
                                                 &g_gr_quiesce_max,
                                                 gr_quiesce_start, TIMELINE_CLOCK());
#endif
                        bulk_send_count = 0;
                        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_publish_done, 0);
                        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_worker_done, 0);
                        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_copy_start, 0);
                        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_copy_done, 0);
                        // L3 warp 只在此时看到新的 round id；此前即使 remote_mark
                        // 非空也不能消费，因 work/injection 可能仍在收尾。
                        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_req, global_active_round);
#if (GLOBAL_ROUND_PROFILE == true)
                        gr_pack_start = TIMELINE_CLOCK();
#endif
                    }
                    global_pack_armed = true;
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }

                if (l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_publish_done) < global_active_round)
                {
                    __threadfence();
                    continue;
                }

#if (GLOBAL_ROUND_PROFILE == true)
                if (!lane_id && !gr_pack_profiled)
                {
                    timeline_clock_type now = TIMELINE_CLOCK();
                    global_round_profile_add(&g_gr_pack, &g_gr_pack_max,
                                             gr_pack_start, now);
                    gr_inbox_start = now;
                    gr_pack_profiled = true;
                }
#endif

                int ready = bulk_inbox_ready_epoch(
                    bulk_inbox_epoch, bulk_inbox_state, bulk_inbox_generation,
                    bulk_rx_epoch);
                if (ready < global_active_round)
                {
                    __threadfence_system();
                    continue;
                }
                if (ready > global_active_round)
                {
                    if (!lane_id)
                    {
                        printf("GLOBAL_ROUND_PROTOCOL_ERROR g%d expected=%d got=%d rx=%d\\n",
                               v_begin, global_active_round, ready, bulk_rx_epoch);
                        l3_atomic_store_release<cuda::thread_scope_block>(&manager_end, 1);
                        l3_atomic_store_release<cuda::thread_scope_device>(global_exit, 1);
                    }
                    __syncwarp();
                    continue;
                }

#if (GLOBAL_ROUND_PROFILE == true)
                if (!lane_id && !gr_inbox_profiled)
                {
                    timeline_clock_type now = TIMELINE_CLOCK();
                    global_round_profile_add(&g_gr_inbox_wait,
                                             &g_gr_inbox_wait_max,
                                             gr_inbox_start, now);
                    gr_apply_start = now;
                    gr_inbox_profiled = true;
                }
#endif

                int rx_slot = bulk_inbox_slot(global_active_round);
                int rx_count = 0;
                if (!lane_id)
                    rx_count = l3_atomic_load_relaxed<cuda::thread_scope_system>(bulk_inbox_count + rx_slot);
                rx_count = __shfl_sync(FULL_MASK, rx_count, 0);
                if (rx_count < 0) rx_count = 0;
                if (rx_count > v_local)
                    rx_count = v_local;

                bool peer_round_activity = false;
#if (GLOBAL_ROUND_DIRECT_P2P == true)
                // Direct P2P 路径的候选已经在发送卡 quiesce 窗口内 atomicMin
                // 到本卡 node_data，并置好本卡 dirty/hint。这里仅确认对端已经
                // 发布同一 round 的完成标记，再回写 ack 允许其复用 epoch slot。
                int rx_improved = 0;
                int frontier_appended = 0;
                // 发送卡在 marker 前已做 system fence；接收卡再做一次
                // system fence，确保随后解除 work/injection gate 时读取到
                // 对端 atomicMin 落位的权威距离，而不是旧缓存行。
                __threadfence_system();
                // Direct P2P 的接收端不再经过 inbox apply。保守构建只在本轮
                // marker 确实携带候选时做兜底扫描，修复任何“距离已落位但 dirty
                // 信号未被注入路径看到”的顶点；空 marker 不会产生新的跨卡写，
                // 因而不重复扫描整个分区。关闭该宏时依赖发送侧
                // atomicMin→dirty→marker 的 system-fence 顺序与本轮双向 ACK 握手。
                bool direct_repair_lane = false;
#if (GLOBAL_ROUND_DIRECT_EXACT_LIST == true)
                // 精确列表是发送侧 atomicMin 真正胜出的条目；在解除 gate 前
                // 再置一次本卡 dirty/hint，修复跨卡 dirty 原子偶发未被注入侧
                // 及时观察的窗口。这里按候选数扫描，而不是扫描整个分区。
                NODE_TYPE *direct_slot_inbox =
                    bulk_inbox + rx_slot * (v_local + 1);
                for (int i = lane_id; i < rx_count; i += WARP_SIZE)
                {
                    NODE_TYPE item = direct_slot_inbox[i];
                    int local_id = item.id - v_begin;
                    if (local_id < 1 || local_id > v_local)
                        continue;
                    int r0 = local_id - 1;
                    l3_system_mark_publish(&dirty_bitmap[r0 >> 5], 1u << (r0 & 31));
                    l3_system_mark_publish(&dirty_hint[r0 >> 10],
                                           1u << ((r0 >> 5) & 31));
                }
                __syncwarp();
#endif
#if (GLOBAL_ROUND_DIRECT_BACKSTOP == true)
                int direct_sent_count = atomicAdd(&bulk_send_count, 0);
                // 非空接收轮可能刚落位跨卡改进；完全空的候选轮是终止
                // 前的精确复核点。仅“单向发送、无接收”的中间轮跳过全扫。
                if (rx_count > 0 || direct_sent_count == 0)
                    direct_repair_lane = backstop_scan_range(
                        node_data, last_processed, dirty_bitmap, dirty_hint,
                        1, v_local + 1, lane_id);
#endif
                unsigned direct_repair_mask = __ballot_sync(
                    FULL_MASK, direct_repair_lane);
                if (direct_repair_mask != 0)
                    rx_improved = 1;
                __syncwarp();
                bool direct_rx_activity = (rx_count > 0 || rx_improved > 0);
                if (!lane_id)
                    l3_atomic_store_release<cuda::thread_scope_system>(
                        bulk_inbox_ack + rx_slot,
                        global_active_round |
                        (direct_rx_activity
                             ? GLOBAL_ROUND_DIRECT_ACK_ACTIVITY : 0));
                if (!lane_id)
                    l3_atomic_store_release<cuda::thread_scope_system>(bulk_inbox_state + rx_slot, BULK_SLOT_DONE);
                __threadfence_system();

                // 两张卡都先回写“我已消费你发来的 marker”，再等待对端回写
                // “我发给你的 marker”。ACK 高位携带对端在本轮观察到的活动，
                // 防止一张卡在 round 边界发现 backstop repair 时，另一张卡
                // 依据自己的空轮提前结束。
                int peer_ack_raw = 0;
                while (true)
                {
                    if (!lane_id)
                        peer_ack_raw = l3_atomic_load_acquire<cuda::thread_scope_system>(peer_bulk_inbox_ack + rx_slot);
                    peer_ack_raw = __shfl_sync(FULL_MASK, peer_ack_raw, 0);
                    int peer_ack_epoch =
                        peer_ack_raw & GLOBAL_ROUND_DIRECT_ACK_EPOCH_MASK;
                    if (peer_ack_epoch >= global_active_round)
                        break;
                    __threadfence_system();
                }
                peer_round_activity =
                    (peer_ack_raw & GLOBAL_ROUND_DIRECT_ACK_ACTIVITY) != 0;
                bulk_rx_epoch = global_active_round;
                __syncwarp();
#elif (GLOBAL_ROUND_PARALLEL_APPLY == true)
                // work/injection 已全部 quiesce；让每个 L2 manager warp 分片应用
                // inbox，完成后由 warp0 一次性发布 frontier tail 和 inbox ack。
                bool apply_claimed = false;
                if (!lane_id)
                    apply_claimed = bulk_inbox_claim_read(
                        bulk_inbox_state, bulk_inbox_generation, bulk_inbox_epoch,
                        rx_slot, global_active_round);
                apply_claimed = __shfl_sync(FULL_MASK, apply_claimed, 0);
                if (!apply_claimed)
                {
                    if (!lane_id)
                        printf("GLOBAL_ROUND_PROTOCOL_ERROR g%d claim round=%d rx=%d\\n",
                               v_begin, global_active_round, bulk_rx_epoch);
                    l3_atomic_store_release<cuda::thread_scope_block>(&manager_end, 1);
                    l3_atomic_store_release<cuda::thread_scope_device>(global_exit, 1);
                    __syncwarp();
                    continue;
                }
                if (!lane_id)
                {
                    bulk_frontier_append_base = atomicAdd(bulk_frontier_tail, 0);
                    bulk_frontier_append_count = 0;
                    global_apply_done = 0;
                    __threadfence();
#if (GLOBAL_ROUND_PROFILE == true)
                    gr_apply_start = TIMELINE_CLOCK();
#endif
                    atomicExch((int *)&global_apply_req, global_active_round);
                }
                __threadfence();
                __syncwarp();
                while (atomicAdd(&global_apply_done, 0) < mlmq.manage_warp_num())
                    __threadfence();

                int frontier_appended = atomicAdd(&bulk_frontier_append_count, 0);
                int frontier_base = atomicAdd(&bulk_frontier_append_base, 0);
                int frontier_capacity = (frontier_base >= 0 && frontier_base <= v_local)
                                      ? (v_local - frontier_base + 1) : 0;
                if (frontier_appended < 0) frontier_appended = 0;
                if (frontier_appended > frontier_capacity)
                    frontier_appended = frontier_capacity;
                __threadfence();
                __syncwarp();
                if (!lane_id)
                    atomicExch(bulk_frontier_tail, frontier_base + frontier_appended);
                __threadfence_system();
                __syncwarp();
                if (!lane_id)
                    bulk_inbox_finish_read(
                        bulk_inbox_state, bulk_inbox_ack,
                        rx_slot, global_active_round);
                __threadfence_system();
                bulk_rx_epoch = global_active_round;
                if (!lane_id)
                    atomicExch((int *)&global_apply_req, 0);
                __threadfence();
                __syncwarp();
                int rx_improved = frontier_appended;
#else
                // GLOBAL_ROUND 的 node_in 只有 32 条，若直接复用会导致大 inbox
                // 被拆成数千次小 write_through。node_out 是 manage warp 专用的
                // shared-memory 预留区，在本协议中不被其他 warp 使用，改用它做
                // 大批量压实缓冲，减少本地 L2 queue 的调用/同步次数。
                int rx_improved = global_round_consume_inbox_direct(
                    bulk_inbox, bulk_inbox_count, bulk_inbox_epoch, bulk_inbox_ack,
                    bulk_inbox_state, bulk_inbox_generation,
                    bulk_rx_epoch, node_data, peer_cache_feedback, v_begin, v_local,
                    node_out, GLOBAL_ROUND_APPLY_BATCH, &bulk_apply_count,
                    mlmq, debug_time, lane_id
#if (GLOBAL_ROUND_FRONTIER == true)
                    , bulk_frontier, bulk_frontier_tail,
                    &bulk_frontier_append_base, &bulk_frontier_append_count
#endif
                    );
#if (GLOBAL_ROUND_FRONTIER == true)
                int frontier_appended = atomicAdd(&bulk_frontier_append_count, 0);
                int frontier_base = atomicAdd(&bulk_frontier_append_base, 0);
                int frontier_capacity = (frontier_base >= 0 && frontier_base <= v_local)
                                      ? (v_local - frontier_base + 1) : 0;
                if (frontier_appended < 0) frontier_appended = 0;
                if (frontier_appended > frontier_capacity)
                    frontier_appended = frontier_capacity;
                // frontier 路径不走 write_through，consume 函数本身不会返回改进数。
                rx_improved = frontier_appended;
#else
                int frontier_appended = 0;
#endif
#endif
                int sent = atomicAdd(&bulk_send_count, 0);
                if (sent < 0) sent = 0;
                if (sent > peer_v_local)
                    sent = peer_v_local;

#if (GLOBAL_ROUND_STATS == true)
                if (!lane_id)
                {
                    atomicAdd(&g_global_round_count, 1ull);
                    atomicAdd(&g_global_round_sent, (unsigned long long)sent);
                    atomicAdd(&g_global_round_recv, (unsigned long long)rx_count);
                    atomicAdd(&g_global_round_improved, (unsigned long long)rx_improved);
                    atomicAdd(&g_global_round_frontier_append,
                              (unsigned long long)frontier_appended);
                }
#endif

                bool round_nonempty = (sent > 0 || rx_count > 0 || rx_improved > 0
                                       || peer_round_activity);
                global_round = global_active_round;
                global_active_round = 0;
                global_pack_armed = false;
#if (GLOBAL_ROUND_PROFILE == true)
                if (!lane_id)
                    gr_release_start = TIMELINE_CLOCK();
#endif
                if (!lane_id)
                {
                    l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 0);
                    l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                    l3_atomic_store_release<cuda::thread_scope_block>(&bulk_inject_ack, 0);
                    timeline_idle_write_reason(local_idle, round_nonempty ? 0 : 1,
                                                round_nonempty ? 21 : 22);
                }
                __threadfence_system();
                __syncwarp();

#if (GLOBAL_ROUND_PROFILE == true)
                if (!lane_id)
                {
                    timeline_clock_type now = TIMELINE_CLOCK();
                    global_round_profile_add(&g_gr_apply, &g_gr_apply_max,
                                             gr_apply_start, gr_release_start);
                    global_round_profile_add(&g_gr_release, &g_gr_release_max,
                                             gr_release_start, now);
                    atomicAdd(&g_gr_profile_rounds, 1ull);
                }
#endif

#if (GLOBAL_ROUND_DIAG == true)
                if (!lane_id)
                {
                    global_round_diag++;
                    if (global_round_diag <= 64 || (global_round_diag & 255u) == 0)
                        printf("GLOBAL_ROUND_DONE g%d round=%d sent=%d recv=%d improved=%d\\n",
                               v_begin, global_round, sent, rx_count, rx_improved);
                }
#endif

                // 两边都已经发布了本轮列表；本轮完全为空时不会再有生产者，
                // 因而可以直接结束。非空轮解除 work gate，进入下一轮本地计算。
                if (!round_nonempty)
                {
                    if (!lane_id)
                    {
                        timeline_idle_write(local_idle, 1);
                        l3_atomic_store_release<cuda::thread_scope_block>(&manager_end, 1);
                        l3_atomic_store_release<cuda::thread_scope_device>(global_exit, 1);
                        atomicExch(&g_t_term, TIMELINE_CLOCK());
                    }
                    __syncwarp();
                    continue;
                }
            }
        }
        else
#elif (BULK_ROUND == true && BULK_EPOCH == true)
        if (n_gpu > 1)
        {
            // BULK_ROUND 的 epoch 由每张卡独立推进；inbox ack 保证发送侧不会覆盖
            // 接收侧尚未消费的上一条消息。空 epoch 时保持 quiesce request，直到
            // 对端也报告空闲，避免两次 idle 之间又产生新的远程候选。
            int bulk_epoch = 0;
            int bulk_rx_epoch = 0;
            int bulk_active_epoch = 0;
            int bulk_round_rx_improved = 0;
            bool bulk_local_idle = false;
            int bulk_idle_stable = 0;
#if (BULK_FRONTIER == true)
            bool bulk_frontier_reset_done = false;
#endif
#if (BULK_DIAG == true)
            unsigned bulk_diag_iter = 0;
#endif

            while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
            {
#if (BULK_DIAG == true)
                bulk_diag_iter++;
                if (!lane_id && (bulk_diag_iter % BULK_DIAG_K) == 0)
                {
                    int in_epoch = (bulk_inbox_epoch != NULL)
                                 ? bulk_inbox_ready_epoch(
                                     bulk_inbox_epoch, bulk_inbox_state,
                                     bulk_inbox_generation, bulk_rx_epoch) : -1;
                    int in_ack = (bulk_inbox_ack != NULL)
                               ? l3_atomic_load_acquire<cuda::thread_scope_system>(bulk_inbox_ack + bulk_inbox_slot(bulk_rx_epoch + 1)) : -1;
                    int qsz = mlmq.get_global_queue_size();
                    int fhead = (bulk_frontier_head != NULL) ? atomicAdd(bulk_frontier_head, 0) : -1;
                    int ftail = (bulk_frontier_tail != NULL) ? atomicAdd(bulk_frontier_tail, 0) : -1;
                    printf("BULK g%d it=%u epoch=%d active=%d rx=%d in=%d/%d q=%d req=%d wack=%d/%d pack=%d done=%d send=%d idle=%d peer=%d frontier=%d/%d\\n",
                           v_begin, bulk_diag_iter, bulk_epoch, bulk_active_epoch, bulk_rx_epoch,
                           in_epoch, in_ack, qsz,
                           bulk_quiesce_req ? l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) : -1,
                           bulk_quiesce_ack ? l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_ack) : -1,
                           bulk_work_warp_total, l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_pack_req),
                           l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_publish_done), atomicAdd(&bulk_send_count, 0),
                           l3_atomic_load_acquire<cuda::thread_scope_system>(local_idle), peer_local_idle ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) : -1,
                           fhead, ftail);
                }
                if (bulk_diag_iter >= 5000000u)
                {
                    if (!lane_id)
                    {
                        printf("BULK_DIAG_ABORT g%d active=%d epoch=%d rx=%d\\n",
                               v_begin, bulk_active_epoch, bulk_epoch, bulk_rx_epoch);
                        l3_atomic_store_release<cuda::thread_scope_block>(&manager_end, 1);
                        l3_atomic_store_release<cuda::thread_scope_device>(global_exit, 1);
                    }
                    __syncwarp();
                    continue;
                }
#endif
                int ready = (bulk_inbox_epoch != NULL)
                          ? bulk_inbox_ready_epoch(
                              bulk_inbox_epoch, bulk_inbox_state,
                              bulk_inbox_generation, bulk_rx_epoch) : bulk_rx_epoch;

#if (BULK_FRONTIER == false)
                // 接收不应等待本卡先进入 idle：对端可以在本卡仍有本地 work 时
                // 发布下一批候选；若把 inbox 延迟到 quiesce 后消费，会形成“本卡
                // 等本地队列排空、对端等本卡消费 inbox”的互等。接收卡在本地
                // work 运行期间直接 atomicMin + write_through，随后继续本地计算。
                if (!bulk_local_idle && bulk_active_epoch == 0
                    && ready > bulk_rx_epoch && node_in_num == 0)
                {
                    int got = bulk_consume_inbox(
                        bulk_inbox, bulk_inbox_count, bulk_inbox_epoch, bulk_inbox_ack,
                        bulk_inbox_state, bulk_inbox_generation,
                        bulk_rx_epoch, node_data, v_begin, v_local,
                        dirty_bitmap, dirty_hint, lane_id BULK_FRONTIER_CONSUME_ARGS);
                    if (got > 0 && !lane_id)
                        timeline_idle_write_reason(local_idle, 0, 1);
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }
#endif

                // 空 epoch 等待期间仍可接收对端消息。收到真正改进后解除本地
                // quiesce，让 work warp 处理 inbox 注入的 L2 项。
                if (bulk_local_idle)
                {
                    if (ready > bulk_rx_epoch && node_in_num == 0)
                    {
#if (BULK_FRONTIER == true)
                        // work warp 仍被 quiesce request 挡住，frontier 已安全排空；
                        // 现在复用其存储接收下一 epoch。
                        bulk_frontier_reset(bulk_frontier_head, bulk_frontier_tail, lane_id);
#endif
                        int got = bulk_consume_inbox(
                            bulk_inbox, bulk_inbox_count, bulk_inbox_epoch, bulk_inbox_ack,
                            bulk_inbox_state, bulk_inbox_generation,
                            bulk_rx_epoch, node_data, v_begin, v_local,
                            dirty_bitmap, dirty_hint, lane_id BULK_FRONTIER_CONSUME_ARGS);
                        if (got > 0)
                        {
                            bulk_round_rx_improved += got;
                            bulk_local_idle = false;
                            if (!lane_id) timeline_idle_write_reason(local_idle, 0, 1);
                            if (!lane_id)
                            {
                                l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 0);
                                l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                            }
                            __threadfence_system();
                            __syncwarp();
                            continue;
                        }
                    }

                    // 对端 idle 标志必须连续两次可见，且本卡 inbox 没有未消费的
                    // epoch，才允许退出。双方此时都仍在 quiesce request 中。
                    int pending = (bulk_inbox_epoch != NULL)
                                ? bulk_inbox_has_next(
                                    bulk_inbox_epoch, bulk_inbox_state,
                                    bulk_inbox_generation, bulk_rx_epoch) : 0;
                    int peer_idle_1 = (peer_local_idle != NULL)
                                    ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) : 0;
                    __threadfence_system();
                    int peer_idle_2 = (peer_local_idle != NULL)
                                    ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) : 0;
                    int q_now = mlmq.get_global_queue_size();
                    int inj_now = atomicAdd(&inj_pending, 0);
                    bool local_empty_now = (q_now == 0 && inj_now == 0 && node_in_num == 0);
#if (BULK_FRONTIER == true)
                    local_empty_now = local_empty_now
                        && bulk_frontier_empty(bulk_frontier_head, bulk_frontier_tail, lane_id);
#endif
                    // peer_local_idle 只表示对端当前没有本地工作，不能证明它已经
                    // 消费了本卡最后发布的 inbox epoch。若本卡在对端 ACK 前退出，
                    // 对端可能错过这条最终反馈，随后产生的回传改进也会被截断。
                    bool bulk_outbox_empty = true;
                    if (peer_bulk_inbox_ack != NULL)
                    {
                        int published = l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published);
                        if (published > 0)
                        {
                            int published_slot = bulk_inbox_slot(published);
                            __threadfence_system();
                            bulk_outbox_empty =
                                (l3_atomic_load_acquire<cuda::thread_scope_system>(peer_bulk_inbox_ack + published_slot) >= published);
                        }
                    }
                    if (!pending && peer_idle_1 == 1 && peer_idle_2 == 1
                        && local_empty_now && bulk_outbox_empty)
                    {
                        bulk_idle_stable++;
#if (BULK_DIAG == true)
                        if (!lane_id)
                            printf("BULK_TERM_CAND g%d epoch=%d rx=%d pending=%d peer=%d/%d q=%d inj=%d outbox=%d stable=%d req=%d\\n",
                                   v_begin, bulk_epoch, bulk_rx_epoch, pending,
                                   peer_idle_1, peer_idle_2, q_now, inj_now,
                                   (int)bulk_outbox_empty, bulk_idle_stable,
                                   l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req));
#endif
                        if (bulk_idle_stable < 2)
                        {
                            __threadfence_system();
                            continue;
                        }
                        if (!lane_id)
                        {
                            l3_atomic_store_release<cuda::thread_scope_block>(&manager_end, 1);
                            l3_atomic_store_release<cuda::thread_scope_device>(global_exit, 1);
                            atomicExch(&g_t_term, TIMELINE_CLOCK());
                        }
                        __syncwarp();
                        continue;
                    }
                    else
                    {
                        bulk_idle_stable = 0;
                        if (!local_empty_now)
                        {
                            bulk_local_idle = false;
                            if (!lane_id)
                            {
                                timeline_idle_write_reason(local_idle, 0, 10);
                                l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 0);
                                l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                            }
                        }
                    }
                    __threadfence();
                    continue;
                }

                // 当前 epoch 已请求 work 静止。只有全部 work warp ack 且无注入在途
                // 后才消费 inbox 和启动 L3 打包，确保 remote_mark 在整个打包期间冻结。
                if (bulk_active_epoch != 0)
                {
                    int ack = (bulk_quiesce_ack != NULL) ? l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_ack) : 0;
                    bool frontier_empty_now = true;
#if (BULK_FRONTIER == true)
                    frontier_empty_now = bulk_frontier_empty(
                        bulk_frontier_head, bulk_frontier_tail, lane_id);
#endif
                    bool frontier_wait = false;
#if (BULK_FRONTIER == true)
                    // reset 后追加的 frontier 属于当前 active epoch；work warp
                    // 仍在 request 中，必须等发布完成并清 request 后再领取，不能
                    // 在此处把它重新当作 active epoch 的前置空条件。
                    frontier_wait = !frontier_empty_now && !bulk_frontier_reset_done;
#endif
                    if (ack < bulk_work_warp_total || atomicAdd(&inj_pending, 0) != 0
                        || frontier_wait)
                    {
                        __threadfence();
                        continue;
                    }

#if (BULK_FRONTIER == true)
                    if (!bulk_frontier_reset_done)
                    {
                        bulk_frontier_reset(bulk_frontier_head, bulk_frontier_tail, lane_id);
                        bulk_frontier_reset_done = true;
                    }
#endif

                    ready = (bulk_inbox_epoch != NULL)
                          ? bulk_inbox_ready_epoch(
                              bulk_inbox_epoch, bulk_inbox_state,
                              bulk_inbox_generation, bulk_rx_epoch) : bulk_rx_epoch;
                    if (ready > bulk_rx_epoch && node_in_num == 0)
                    {
                        int got = bulk_consume_inbox(
                            bulk_inbox, bulk_inbox_count, bulk_inbox_epoch, bulk_inbox_ack,
                            bulk_inbox_state, bulk_inbox_generation,
                            bulk_rx_epoch, node_data, v_begin, v_local,
                            dirty_bitmap, dirty_hint, lane_id BULK_FRONTIER_CONSUME_ARGS);
                        bulk_round_rx_improved += got;
                    }

                    // 现在 work 已静止且 inbox 已消费，才允许 L3 warp 打包发布。
                    if (l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_pack_req) == 0)
                    {
                        if (!lane_id)
                        {
                            bulk_send_count = 0;
                            l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_publish_done, 0);
                            l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_req, bulk_active_epoch);
                        }
                        __syncwarp();
                    }

                    if (l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_publish_done) < bulk_active_epoch)
                    {
                        // L3 可能正在等待 peer 对上一 epoch 的 ack。frontier 模式下
                        // work warp 仍被 quiesce request 挡住，不能在这里继续消费后续
                        // inbox：多个 epoch 会叠加到同一条 v_local+1 的线性 frontier，
                        // 超出容量的条目会被截断。后续 epoch 留在 inbox，等本轮
                        // frontier 被 work 消费、request 解除后再进入下一轮处理。
#if (BULK_FRONTIER == false)
                        ready = (bulk_inbox_epoch != NULL)
                              ? bulk_inbox_ready_epoch(
                                  bulk_inbox_epoch, bulk_inbox_state,
                                  bulk_inbox_generation, bulk_rx_epoch) : bulk_rx_epoch;
                        if (ready > bulk_rx_epoch && node_in_num == 0)
                        {
                            int got = bulk_consume_inbox(
                                bulk_inbox, bulk_inbox_count, bulk_inbox_epoch, bulk_inbox_ack,
                                bulk_inbox_state, bulk_inbox_generation,
                                bulk_rx_epoch, node_data, v_begin, v_local,
                                dirty_bitmap, dirty_hint, lane_id BULK_FRONTIER_CONSUME_ARGS);
                            bulk_round_rx_improved += got;
                        }
#endif
                        __threadfence();
                        continue;
                    }

                    int sent = atomicAdd(&bulk_send_count, 0);
                    int rx_improved = bulk_round_rx_improved;
                    bool round_nonempty = (sent > 0 || rx_improved > 0);
                    bulk_epoch = bulk_active_epoch;
                    bulk_active_epoch = 0;
                    bulk_round_rx_improved = 0;
#if (BULK_FRONTIER == true)
                    bulk_frontier_reset_done = false;
#endif
                    if (!lane_id)
                        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_pack_req, 0);
                    __syncwarp();

                    // 发布后再次检查本地 L2/injection 状态。接收 epoch 可能刚刚
                    // 触发本地注入；如果此时仍有队列工作，不能把本轮误判为空闲。
                    int post_q = mlmq.get_global_queue_size();
                    int post_inj = atomicAdd(&inj_pending, 0);
                    bool post_local_empty = (post_q == 0 && post_inj == 0 && node_in_num == 0);
#if (BULK_FRONTIER == true)
                    post_local_empty = post_local_empty
                        && bulk_frontier_empty(bulk_frontier_head, bulk_frontier_tail, lane_id);
#endif
                    if (round_nonempty || !post_local_empty)
                    {
                        bulk_local_idle = false;
                        if (!lane_id)
                        {
                            timeline_idle_write_reason(local_idle, 0, 2);
                            l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 0);
                            l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                        }
                    }
                    else
                    {
                        // 空 epoch：保留 quiesce request，work warp 不再创建新的远程
                        // 候选；收到对端非空 inbox 时上面的 idle 分支会解除它。
                        bulk_local_idle = true;
                        if (!lane_id)
                            timeline_idle_write(local_idle, 1);
                    }
#if (BULK_DIAG == true)
                    if (!lane_id)
                        printf("BULK_ROUND_DONE g%d epoch=%d sent=%d rximp=%d nonempty=%d postq=%d postinj=%d idle=%d req=%d\\n",
                               v_begin, bulk_epoch, sent, rx_improved,
                               (int)round_nonempty, post_q, post_inj, (int)bulk_local_idle,
                               l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req));
#endif
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }

                // 独立 frontier 模式不再用可能滞后的 qsize 作为 request 前置条件。
                // work warp 会在 request 后自行排空 L1/on-the-fly/local queue，并以
                // quiesce ack 证明安全点；否则残留 read_done 计数会阻塞首个 epoch。
#if (BULK_FRONTIER == true)
                if (!bulk_local_idle && bulk_active_epoch == 0)
                {
                    int next_epoch = bulk_epoch + 1;
                    bulk_active_epoch = next_epoch;
                    bulk_round_rx_improved = 0;
                    if (!lane_id)
                    {
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 1);
                    }
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }
#endif

                bool l2_empty = (mlmq.get_global_queue_size() == 0);
#if (BULK_FRONTIER == true)
                bool frontier_empty_before_req = bulk_frontier_empty(
                    bulk_frontier_head, bulk_frontier_tail, lane_id);
#else
                bool frontier_empty_before_req = true;
#endif
                if (!l2_empty || atomicAdd(&inj_pending, 0) != 0
                    || !frontier_empty_before_req)
                {
                    if (!lane_id) timeline_idle_write_reason(local_idle, 0, 3);
                    __threadfence();
                    continue;
                }

                // request 前做一次精确 dirty 全扫。不能只看 hint：若 hint 因 ABA
                // 假阴而漏掉一个本地待注入顶点，直接进入 epoch 会造成提前终止。
                bool dirty_empty = true;
                for (int w = lane_id; w < dwords; w += WARP_SIZE)
                    if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w]) != 0) dirty_empty = false;
                unsigned dirty_mask = __ballot_sync(FULL_MASK, dirty_empty);
                if (dirty_mask != FULL_MASK)
                {
                    if (!lane_id) timeline_idle_write_reason(local_idle, 0, 4);
                    __threadfence();
                    continue;
                }

                int next_epoch = bulk_epoch + 1;
                bulk_active_epoch = next_epoch;
                bulk_round_rx_improved = 0;
                if (!lane_id)
                {
                    l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                    l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 1);
                }
                __threadfence_system();
                __syncwarp();
            }
        }
        else
#endif
#if (GLOBAL_ROUND_ASYNC == true)
        // Async 路径只在终止探测时使用 bulk_quiesce_req。请求值是单调递增的
        // probe token，而不是普通 BULK/GLOBAL_ROUND 使用的 0/1 电平；这样
        // manager 在发现新活动后撤销请求、又立即发起下一次 probe 时，work
        // warp 不会把新请求误认为已经 ACK 过的旧请求。
        int async_quiesce_token = 0;
#endif
#if (BULK_ROUND == true && L3_TERM_HANDSHAKE == true && \
     BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && \
     SEED_BARRIER == false)
        // 默认 BULK 终止请求只由 warp0 递增；work/injection ACK 记录该 token。
        int l3_term_token = 0;
#endif
#if (L3_DEFER_BUSY_PROBE == true)
        l3_probe_policy probe_policy;
#endif
#if (L3_WAIT_DIAG == true)
        l3_wait_clock manager_wait(!lane_id);
#endif
        while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
        {
#if (L3_WAIT_DIAG == true)
            if(!lane_id) manager_wait.tick(
                l3_atomic_load_acquire<cuda::thread_scope_system>(l3_term_state));
#endif
#if (L3_LIVE_SNAPSHOT == true)
            atomicAdd(&g_l3_live_state.manager_lane_iter[lane_id], 1ull);
            l3_live_manager_lane_state(lane_id, 1, -1);
            if (!lane_id)
            {
                atomicAdd(&g_l3_live_state.manager_iter, 1ull);
                l3_live_manager_state(
                    1, bulk_rx_epoch, node_in_num, -1,
                    &bs_req, &bs_done, &bs_found);
            }
#endif
#if (HANG_DIAG == true)
            if (!lane_id && !diag_loop_seen)
            {
                printf("L3_TERM_LOOP_ENTRY g%d\n", v_begin);
                diag_loop_seen = true;
            }
#endif
#if (L3_FAULT_INJECT_ACK_DELAY == true)
            // ACK 延迟模型的恢复动作必须先于本轮 READY/termination 检查，
            // 否则发送端可能持续观察到未确认的 DONE 槽位。
            if (n_gpu > 1 && bulk_inbox_state != NULL
                && bulk_inbox_ack != NULL)
            {
                if (!lane_id)
                    bulk_inbox_retry_delayed_ack(
                        bulk_inbox_state, bulk_inbox_ack);
                __threadfence_system();
                __syncwarp();
            }
#endif
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
            // manager-owned receive frontier：warp0 manager 直接消费 READY inbox，
            // 只执行 atomicMin + ready bitmap 发布；work warp 负责后续
            // simple_process，manager 不再把远程消息写入本地 MLMQ。
            if (n_gpu > 1 && bulk_inbox_epoch != NULL
                && bulk_inbox_has_next(
                    bulk_inbox_epoch, bulk_inbox_state,
                    bulk_inbox_generation, bulk_rx_epoch))
            {
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                    l3_live_manager_state(
                        10, bulk_rx_epoch, node_in_num, -1,
                        &bs_req, &bs_done, &bs_found);
#endif
                int got = global_round_consume_inbox_manager_frontier(
                    bulk_inbox, bulk_inbox_count, bulk_inbox_epoch,
                    bulk_inbox_ack, bulk_inbox_state, bulk_inbox_generation,
                    bulk_rx_epoch, node_data, peer_cache_feedback,
                    v_begin, v_local, async_rx_ready_bitmap,
                    async_rx_ready_hint, lane_id);
                if (got > 0 && !lane_id)
                    timeline_idle_write_reason(local_idle, 0, 32);
                __threadfence_system();
                __syncwarp();
                continue;
            }
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
            // async receive frontier 的 slot 生命周期由 warp0 manager 驱动：
            // READY -> READING（激活给 work warp）-> DONE（全部 claim 完成）。
            // 这里不执行 atomicMin/write_through；manager 只做轻量元数据推进。
            if (n_gpu > 1 && bulk_inbox_active_slot != NULL)
            {
                int active_slot = atomicAdd(bulk_inbox_active_slot, 0);
                if (active_slot < 0)
                {
                    int ready = bulk_inbox_ready_epoch(
                        bulk_inbox_epoch, bulk_inbox_state,
                        bulk_inbox_generation, bulk_rx_epoch);
                    if (ready > bulk_rx_epoch)
                    {
                        int slot = bulk_inbox_slot(ready);
                        bool claimed = false;
                        if (!lane_id)
                            claimed = bulk_inbox_claim_read(
                                bulk_inbox_state, bulk_inbox_generation,
                                bulk_inbox_epoch, slot, ready);
                        claimed = __shfl_sync(FULL_MASK, claimed, 0);
                        if (claimed)
                        {
                            if (!lane_id)
                            {
                                atomicExch(bulk_inbox_read_head + slot, 0);
                                atomicExch(bulk_inbox_inflight + slot, 0);
                                __threadfence_system();
                                atomicExch(bulk_inbox_active_slot, slot);
                            }
                            __threadfence_system();
                        }
                    }
                }
                else
                {
                    int count = l3_atomic_load_relaxed<cuda::thread_scope_system>(bulk_inbox_count + active_slot);
                    int head = atomicAdd(bulk_inbox_read_head + active_slot, 0);
                    int inflight = atomicAdd(bulk_inbox_inflight + active_slot, 0);
                    int epoch = l3_atomic_load_relaxed<cuda::thread_scope_system>(bulk_inbox_epoch + active_slot);
                    if (head >= count)
                    {
                        bool closing = false;
                        if (!lane_id)
                        {
                            int old = atomicCAS(
                                bulk_inbox_state + active_slot,
                                BULK_SLOT_READING, BULK_SLOT_CLOSING);
                            closing = (old == BULK_SLOT_READING
                                       || old == BULK_SLOT_CLOSING);
                        }
                        closing = __shfl_sync(FULL_MASK, closing, 0);
                        if (closing)
                        {
                            __threadfence();
                            inflight = atomicAdd(
                                bulk_inbox_inflight + active_slot, 0);
                            if (inflight != 0)
                            {
                                __threadfence();
                                continue;
                            }
                            if (!lane_id)
                            {
                                bulk_inbox_finish_read(
                                    bulk_inbox_state, bulk_inbox_ack,
                                    active_slot, epoch);
                                bulk_rx_epoch = epoch;
                                atomicExch(bulk_inbox_active_slot, -1);
                            }
                        }
                    }
                }
                bulk_rx_epoch = __shfl_sync(FULL_MASK, bulk_rx_epoch, 0);
            }
#endif
#if (BULK_ROUND == true)
#if !(GLOBAL_ROUND_ASYNC == true && \
      (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true || \
       GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true))
            // CP1e: READY 只取一次 warp-uniform 快照。诊断开关不再改变
            // 正常路径的读取次数和控制流时序。
            bool bulk_inbox_next = false;
            if (n_gpu > 1 && bulk_inbox_epoch != NULL)
                bulk_inbox_next = bulk_inbox_has_next(
                    bulk_inbox_epoch, bulk_inbox_state,
                    bulk_inbox_generation, bulk_rx_epoch);
#if (L3_LIVE_SNAPSHOT == true)
            l3_live_manager_lane_state(
                lane_id, bulk_inbox_next ? 7 : 6, -1);
#endif
            if (bulk_inbox_next)
            {
#if (L3_LIVE_SNAPSHOT == true)
                l3_live_manager_lane_state(lane_id, 8, -1);
#endif
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                    l3_live_manager_state(
                        10, bulk_rx_epoch, node_in_num, -1,
                        &bs_req, &bs_done, &bs_found);
#endif
#if (HANG_DIAG == true)
                if (!lane_id && !diag_inbox_seen)
                {
                    printf("L3_TERM_INBOX_BRANCH g%d req=%d state=%d rx=%d\n",
                           v_begin,
                           (l3_term_req != NULL) ? l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req) : -1,
                           (l3_term_state != NULL) ? l3_atomic_load_acquire<cuda::thread_scope_system>(l3_term_state) : -1,
                           bulk_rx_epoch);
                    diag_inbox_seen = true;
                }
#endif
#if (L3_TERM_HANDSHAKE == true && BULK_EPOCH == false && \
     GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && SEED_BARRIER == false)
                // 对端可能在本卡进入 QUIESCING 后才发布最后一批 inbox。
                // 先撤销本地终止请求并释放 work，再消费 inbox；否则
                // bulk_consume_inbox 的 write_through 可能与已冻结 work
                // 形成 manager/work 互等。
                int inbox_term_req_now = 0;
                if (!lane_id && l3_term_req != NULL)
                    inbox_term_req_now = l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req);
                inbox_term_req_now = __shfl_sync(
                    FULL_MASK, inbox_term_req_now, 0);
                if (inbox_term_req_now != 0)
                {
                    if (!lane_id)
                    {
#if (L3_PROGRESS_DIAG == true)
                        ++g_l3_progress.term_inbox_cancel;
#endif
                        l3_atomic_store_release<cuda::thread_scope_system>(l3_term_state, static_cast<int>(L3_TERM_ACTIVE));
                        l3_atomic_store_release<cuda::thread_scope_device>(l3_term_req, 0);
                        timeline_idle_write_reason(local_idle, 0, 42);
#if (HANG_DIAG == true)
                        printf("L3_TERM_INBOX_CANCEL g%d rx=%d\n",
                               v_begin, bulk_rx_epoch);
#endif
                    }
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }
#endif
#if (GLOBAL_ROUND_ASYNC == true)
                // 终止 probe 期间 work warp 可能已经停在安全点；若此时
                // 对端又发布最后一批 inbox，不能直接 write_through，否则
                // manager 可能因 L2 队列满而等待，而 work 又在等待 probe
                // 解除。先撤销 probe，让 work 恢复后再消费该 inbox。
                if (bulk_quiesce_req != NULL
                    && l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) != 0)
                {
                    if (!lane_id)
                    {
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 0);
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                        timeline_idle_write_reason(local_idle, 0, 42);
#if (BULK_DIAG == true)
                        printf("ASYNC_RX_CANCEL_PROBE g%d rx=%d\\n",
                               v_begin, bulk_rx_epoch);
#endif
                    }
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }
#endif
#if (GLOBAL_ROUND_ASYNC == true && GLOBAL_ROUND_WINDOWED_ASYNC == false)
                // 异步路径直接把 inbox 中真正改进的条目写入本地 L2。
                // 不置 dirty，也不等待注入 warp；work warp 可以与此处的
                // atomicMin/MLMQ MRMW write_through 并行推进。
                int got = global_round_consume_inbox_direct(
                    bulk_inbox, bulk_inbox_count, bulk_inbox_epoch, bulk_inbox_ack,
                    bulk_inbox_state, bulk_inbox_generation,
                    bulk_rx_epoch, node_data, peer_cache_feedback,
                    v_begin, v_local, node_out, GLOBAL_ROUND_APPLY_BATCH,
                    &bulk_apply_count, mlmq, debug_time, lane_id);
                if (got > 0 && !lane_id)
                    timeline_idle_write_reason(local_idle, 0, 31);
                __threadfence_system();
                __syncwarp();
#else
                // 窗口化异步路径：发送侧仍使用双 bank + generation-safe inbox，
                // 接收侧不走 manager-frontier/direct-apply。先 atomicMin 并置
                // authoritative dirty bitmap，再由现有 injection warp 走
                // write_through；这样接收事件与 work 的并发边界沿用已回归的
                // dirty 协议，而不是引入新的 frontier claim 生命周期。
#if (L3_DIRECT_RX == true)
                int got = l3_receive_to_l2(
                    bulk_inbox, bulk_inbox_count, bulk_inbox_epoch, bulk_inbox_ack,
                    bulk_inbox_state, bulk_inbox_generation, bulk_rx_epoch,
                    node_data, v_begin, v_local, node_out, mlmq, debug_time, lane_id
#if (L3_RX_LAG_DIAG == true)
                    , last_processed
#endif
#if (L3_RX_PRIORITY_BOOTSTRAP == true)
                    , !(src >= v_begin && src < v_begin + v_local)
#endif
#if (L3_RX_FEEDBACK_MODE > 0)
                    , l3_channel.rx_feedback
#endif
#if (L3_RX_EXPRESS == true)
                    , rx_express, rx_express_enabled
#endif
#if (L3_RX_L2_PULL == true)
                    , l3_channel.rx_commit_seq
#if (L3_RX_L2_PULL_DIAG == true)
                    , l3_channel.rx_l2_pull_stats
#endif
#endif
                    );
#else
                int got = bulk_consume_inbox(
                    bulk_inbox, bulk_inbox_count, bulk_inbox_epoch, bulk_inbox_ack,
                    bulk_inbox_state, bulk_inbox_generation,
                    bulk_rx_epoch, node_data, v_begin, v_local,
                    dirty_bitmap, dirty_hint, lane_id BULK_FRONTIER_CONSUME_ARGS);
#endif
#if (L3_LIVE_SNAPSHOT == true)
                l3_live_manager_lane_state(lane_id, 9, -1);
                if (!lane_id)
                    l3_live_manager_state(
                        11, bulk_rx_epoch, node_in_num, -1,
                        &bs_req, &bs_done, &bs_found);
#endif
                if (got > 0 && !lane_id)
                    timeline_idle_write_reason(local_idle, 0, 1);
                __threadfence_system();
                __syncwarp();
#endif
            }
#endif
#endif  // BULK_ROUND inbox consume/apply block
#if (MANAGE_PROFILE == true)
            unsigned p_t0 = clock();
#endif
            backstop_round++;
            diag_iter++;
#if (BULK_DIAG == true && GLOBAL_ROUND_ASYNC == true)
            if (!lane_id && (diag_iter % BULK_DIAG_K) == 0)
            {
                int req_diag = (bulk_quiesce_req != NULL)
                             ? l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) : -1;
                int ack_diag = (bulk_quiesce_ack != NULL)
                             ? l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_ack) : -1;
                printf("ASYNC_TERM_DIAG g%d it=%u req=%d ack=%d/%d q=%d idle=%d peer=%d rx=%d out=%d\\n",
                       v_begin, diag_iter, req_diag, ack_diag,
                       bulk_work_warp_total,
                       (int)mlmq.get_global_queue_size(),
                       (int)l3_atomic_load_acquire<cuda::thread_scope_system>(local_idle),
                       peer_local_idle ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) : -1,
                       bulk_rx_epoch,
                       (int)l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published));
            }
#endif
            // 1. 扫描脏位图取改进顶点（不消费 L2：L2 活由 work warps 消费）
            //    SESSION 22: 周期性 nohint 全扫 slice0，清 dirty_hint ABA 假阴残留
            //    （孤儿 dirty 落在 slice0 时，w0 无注入 warp 兜底，须自身 nohint 扫）。
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == false && \
     GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == false)
#if (BULK_ROUND == false)
            if (node_in_num == 0 && n_gpu > 1)
            {
                if ((backstop_round & 1023) == 0)
                    collect_dirty_slice_nohint(dirty_bitmap, dirty_hint, c_beg, c_end, v_local, v_begin,
                                               node_data, node_in, node_in_num, node_size);
                else
                    collect_dirty_slice(dirty_bitmap, dirty_hint, c_beg, c_end, v_local, v_begin,
                                        node_data, node_in, node_in_num, node_size);
                __syncwarp();
            }

            // 3. 注入：改进顶点 write_through 写 L2（由 work warps 正常处理）
            //    规避 manage 自身 L1 堆积（manage 不消费 L2）与 backpressure
            //    delta 桶 write 对 dist<base 会 clamp 到当前 first_pos 桶（仍被读）
            if (node_in_num > 0)
            {
                int write_num = node_in_num;
                mlmq.write_through(node_in, write_num, 0, 0, lane_id, debug_time);
                node_in_num = 0;
            }
#endif
#endif
#if (MANAGE_PROFILE == true)
            unsigned p_t1 = clock();
#endif

            // E1c: 忙碌态降频。L2 非空 => 不可能终止，跳过昂贵全扫描（dirty/mark/backstop，
            //   大图 dwords/mark_words 各 7 万 words，全扫描每轮 ~1 万次迭代，多轮即数百秒）。
            //   正确性：终止条件要求 l2_empty；dirty 丢失（注入取走未处理）由 quiet 态
            //   backstop 补扫兜底（qsize==0 时才扫，E1a 逻辑不变）。
            //   注意：忙碌态必须复位 local_idle（防 peer 读到过期 idle 提前终止）。
            int l2_size_now = 0;
            if (!lane_id)
                l2_size_now = mlmq.get_global_queue_size();
            l2_size_now = __shfl_sync(FULL_MASK, l2_size_now, 0);
            bool l2_empty = (l2_size_now == 0);
#if (L3_ADMISSION_BUDGET == true)
            if(n_gpu>1 && !lane_id)
                l3_admission_publish(mlmq.q2,l3_channel.admission,l3_channel.peer_admission,l2_size_now,admission_last);
#endif
#if (L3_LIVE_SNAPSHOT == true)
            l3_live_manager_lane_state(
                lane_id, l2_empty ? 22 : 21, (int)l2_empty);
            if (!lane_id)
                l3_live_manager_state(
                    l2_empty ? 20 : 21, bulk_rx_epoch, node_in_num,
                    l2_empty ? 0 : 1, &bs_req, &bs_done, &bs_found);
#endif
            if (!l2_empty)
            {
#if (L3_STICKY_READY == true || L3_TERM_WAIT_ACK == true)
                // A frozen worker cannot consume newly visible L2 work.
                // Release the request before the early busy-path return.
                if (!lane_id && l3_term_req && l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req) != 0) {
#if (L3_TERM_WAIT_ACK == true && L3_PROGRESS_DIAG == true)
                    ++g_l3_progress.term_cancels;
                    ++g_l3_progress.term_invalid;
#endif
                    l3_atomic_store_release<cuda::thread_scope_system>(l3_term_state, static_cast<int>(L3_TERM_ACTIVE));
                    l3_atomic_store_release<cuda::thread_scope_device>(l3_term_req, 0);
                    __threadfence_system();
                }
#endif
                if (!lane_id) timeline_idle_write_reason(local_idle, 0, 3);
                __threadfence();
                continue;
            }

#if (L3_DEFER_BUSY_PROBE == true)
            // RX/cancellation and the L2 check above always run first. Busy is
            // only a reason to defer, never evidence that termination is safe.
            int probe_tx_busy = 0;
            if (!lane_id) probe_tx_busy = atomicAdd(&l3_busy, 0);
            probe_tx_busy = __shfl_sync(FULL_MASK, probe_tx_busy, 0);
            if (n_gpu > 1 && probe_policy.defer(probe_tx_busy != 0))
            {
                if (!lane_id)
                {
                    if (l3_term_req && l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req) != 0)
                    {
                        l3_atomic_store_release<cuda::thread_scope_system>(l3_term_state, static_cast<int>(L3_TERM_ACTIVE));
                        l3_atomic_store_release<cuda::thread_scope_device>(l3_term_req, 0);
                    }
                    timeline_idle_write_reason(local_idle, 0, 43);
                }
                __threadfence_system();
                __syncwarp();
                continue;
            }
#endif

#if (SEED_BARRIER == true)
            // 性能优化: 等待期（gpu0 phase3 等 gpu1 / gpu1 等 seed_ready）l2_empty 轮检查降频，
            //   每 4 轮做一次完整检查（小图 overhead 主要来自 warp0 每轮固定开销 × 大量轮次）。
            //   轻量检测: 降频轮只查等待解除条件（P2P 原子读 peer_idle / seed_ready），
            //   避免小图（等待轮次少）饿死；解除后才走完整检查。
            if (n_gpu > 1)
            {
                bool is_src_w = (src >= v_begin && src < v_end);
                int ph_w = (*(volatile int *)phase);
                bool wait_phase = is_src_w ? (ph_w == 3) : (*(volatile int *)seed_ready == 0);
                if (wait_phase)
                {
                    wait_round++;
                    if (wait_round % 4 != 0)
                    {
                        bool w_done = is_src_w
                            ? (peer_local_idle != NULL && l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) == 1)
                            : (*(volatile int *)seed_ready != 0);
                        if (!w_done) { __threadfence(); continue; }
                    }
                }
            }
#endif

#if (SEED_BARRIER == true)
            // ===== 解法0: phase/seed 门控（仅双卡）=====
            // 源卡（含 src）: phase 状态机（1=计算 2=灌值 3=done）
            // 接收卡: seed_ready 前禁 backstop 扫描 + 禁终止（防灌值窗口扫到部分种子）
            const bool sb_active = (n_gpu > 1);
            const bool is_src_card = sb_active && (src >= v_begin && src < v_end);
            const int ph = sb_active ? (*(volatile int *)phase) : 3;
            // bs_skip: 接收卡 seed_ready 前/注入完成前禁扫（防灌值窗口命中部分种子、
            //   防注入期 backstop 重标记致永不排空）；源卡 phase=3 也禁扫（收敛已确认）。
            //   正确性：termination 的 peer_idle 二次确认仍做协作 backstop 兜底。
#if (ASYNC_FB == true)
            // 方案 B: 源卡 phase3 仍可能有反向反馈改进落位 → 不禁 backstop（需兜底）；
            //   接收卡 inject 完成后正常。接收卡 seed_ready/inject 前仍禁。
            const bool bs_skip = sb_active && (is_src_card ? false
                : ((*(volatile int *)seed_ready == 0) || (*(volatile int *)seed_inject_done == 0)));
#else
            const bool bs_skip = sb_active && (is_src_card ? (ph == 3)
                : ((*(volatile int *)seed_ready == 0) || (*(volatile int *)seed_inject_done == 0)));
#endif
            const bool term_skip = sb_active && (is_src_card ? (ph == 1 || ph == 2)
                : ((*(volatile int *)seed_ready == 0) || (*(volatile int *)seed_inject_done == 0)));
            // 接收卡 warp0 首见 seed_ready 时刻（诊断注入延迟）
            if (sb_active && !is_src_card && *(volatile int *)seed_ready != 0)
            {
                if (!g_t_sr_w0) atomicExch(&g_t_sr_w0, TIMELINE_CLOCK());
            }
            // 接收卡不要求 mark_empty：其 remote_mark 是到源卡的候选（P2 下全无效，w17 已门控），
            //   残留不影响正确性；源卡必须要求（mark 是待灌最终值信号/终止前提）。
#if (ASYNC_FB == true)
            // 方案 B: 双方 mark 都是真实反向候选（接收卡 w17 已放行）→ 双方都要求 mark_empty。
            const bool need_mark = true;
#else
            const bool need_mark = sb_active ? is_src_card : true;
#endif
#endif

#if (SEED_BARRIER == true && ASYNC_FB == false)
            // P2/SEED_BARRIER 快路径：phase=3 表示源卡已经完成本地收敛、phase2
            // 全量发布、remote_mark 全扫确认和 seed_ready 屏障。源卡之后不再有
            // 合法的反向改进（该路径本来就依赖 P2），因此无需每 4 轮重复执行
            // dirty/mark/backstop/二次确认；只需置本卡 idle 并等待接收卡 idle。
            // 接收卡的 local_idle 仍由原有完整终止检查置位，因而不会跳过其最后
            // 一轮本地 work 或注入在途检查。
            if (sb_active && is_src_card && ph == 3)
            {
                if (!lane_id)
                {
                    __threadfence_system();
                    timeline_idle_write(local_idle, 1);
                    if (!g_t_idle) atomicExch(&g_t_idle, TIMELINE_CLOCK());
                    __threadfence_system();
                    bool peer_done = (peer_local_idle != NULL)
                                  && (l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) == 1);
                    peer_idle_s = peer_done ? 1 : 0;
                }
                __syncwarp();
                if (*(volatile int *)&peer_idle_s)
                {
                    if (!lane_id)
                    {
                        l3_atomic_store_release<cuda::thread_scope_block>(&manager_end, 1);
                        l3_atomic_store_release<cuda::thread_scope_device>(global_exit, 1);
                        atomicExch(&g_t_term, TIMELINE_CLOCK());
                    }
                }
                else
                {
                    __threadfence();
                }
                __syncwarp();
                continue;
            }
#endif

#if (SEED_BARRIER == true)
            // ===== 解法0: 接收卡一次性全量种子注入（免扫描版）=====
            // seed_ready 后、seed_inject_done 前: gpu0 灌值时已 P2P 写入 peer seed_list
            //   （本卡 seed_list）+ count，此处直接消费（免 419万 node_data 扫描，~23ms 省掉）。
            //   全部种子 write_through 入 L2（work 门控 first_pos=0 → 按距离正确落桶）。
            //   注入完成置 seed_inject_done=1，work 一次干净扫掠（SEED_EXP 近最优 work）。
            if (sb_active && !is_src_card && seed_inject_done != NULL
                && *(volatile int *)seed_ready != 0 && *(volatile int *)seed_inject_done == 0)
            {
                if (!g_t_sr_seen) atomicExch(&g_t_sr_seen, TIMELINE_CLOCK());
                unsigned t_inj0 = clock();
                atomicAdd(&g_inject_exec, 1ull);
                int cnt = *(volatile int *)seed_list_cnt;
                __syncwarp();
                // 性能优化: 注入批量增大——用 node_out 预留区（[2*node_size*WARP_NUM_PER_BLOCK, ...)，
                //   未使用）做大缓冲，每批 128 条（原 32/批 → 375 批 → 94 批），
                //   减少 write_through 批次数（gpu1 注入是等 gpu1 段的一部分）。
                NODE_TYPE *inj_buf = node_buf + 2 * node_size * WARP_NUM_PER_BLOCK;
                const int INJ_BIG = 128;   // 512 无稳定收益（中位 53.5ms 持平，偶发 44ms 系噪声），回退
                for (int s0 = 0; s0 < cnt; s0 += INJ_BIG)
                {
                    int n = mlq_min(INJ_BIG, cnt - s0);
                    for (int t = lane_id; t < n; t += WARP_SIZE)
                    {
                        NODE_TYPE seed = seed_list[s0 + t];
                        inj_buf[t] = seed;
#if (SEED_PHASE2_LIST_ONLY == true)
                        // work kernel 仍在 seed_inject_done gate 中；先写本卡权威距离，
                        // 解除 gate 后不会读到 DIST_MAX。
                        int local_id = seed.id - v_begin;
                        if (local_id >= 1 && local_id <= v_local)
                            node_data[local_id] = seed.get_data();
#endif
                    }
                    __syncwarp();
                    int wnum = n;
                    mlmq.write_through(inj_buf, wnum, 0, 0, lane_id, debug_time);
                    __syncwarp();
                }
                if (!lane_id)
                {
                    __threadfence();
                    atomicExch(seed_inject_done, 1);
                    atomicExch(&g_t_inject_done, TIMELINE_CLOCK());
                    atomicAdd(&g_inject_clks, (unsigned long long)(clock() - t_inj0));
                }
                __syncwarp();
            }
#endif


            // 4. 终止检测：L2 空 && dirty 全 0 && 兜底一致 && peer idle
#if (L3_LIVE_SNAPSHOT == true)
            l3_live_manager_lane_state(lane_id, 30, (int)l2_empty);
#endif
            // dirty 全 0 检查（lane 并行，原始全扫描保正确——hint 假阴会漏 dirty 致提前终止）
            // B 开销优化: dirty_empty 降频——默认轮用 dirty_hint 稀疏快速判断（13万 words →
            //   4096 words）；每 MARK_FULL_K 轮 + 源卡 phase2（seed_ready 前关键）全扫。
            //   终止正确性由二次确认（dirty 全扫已有）+ 注入 nohint 兜底保证。
            const int MARK_FULL_K = 2048;
            bool dirty_empty = true;
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
            bool async_rx_ready_empty = true;
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == false && \
     GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == false)
            {
#if (ASYNC_FB == true)
                bool need_full = (backstop_round % MARK_FULL_K == 0)
                              || (sb_active && is_src_card && ph == 2);
                if (!need_full && dirty_hint != NULL)
                {
                    int hwords = (dwords + 31) / 32;
                    for (int w = lane_id; w < hwords; w += 32)
                        if (l3_atomic_load_relaxed<cuda::thread_scope_system>(&dirty_hint[w]) != 0) { dirty_empty = false; break; }
                }
                else
                {
                    for (int w = lane_id; w < dwords; w += 32)
                        if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w]) != 0) { dirty_empty = false; break; }
                }
#else
                // SEED_BARRIER: dirty_hint 快速 + 关键轮全扫——ph==2（seed_ready 前）全扫、
                //   每 MARK_FULL_K 轮全扫；phase1→2 由 phase 状态机全扫确认；终止由二次确认
                //   （dirty 全扫已有）兜底。不再在 pre_ready 加全扫（每轮触发反成负优化）。
                const int MARK_FULL_K = 2048;
                // 平时 hint 快速；全扫仅每 MARK_FULL_K 轮（phase1→2/phase2→3 转换轮由状态机全扫确认）
                bool need_full = (backstop_round % MARK_FULL_K == 0);
                if (!need_full && dirty_hint != NULL)
                {
                    int hwords = (dwords + 31) / 32;
                    for (int w = lane_id; w < hwords; w += 32)
                        if (l3_atomic_load_relaxed<cuda::thread_scope_system>(&dirty_hint[w]) != 0) { dirty_empty = false; break; }
                }
                else
                {
                    for (int w = lane_id; w < dwords; w += 32)
                        if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w]) != 0) { dirty_empty = false; break; }
                }
#endif
            }
#elif (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
            // manager-owned receive frontier 使用独立 ready 位图；真实 dirty
            // 仍由注入 warp 消费。inbox ACK 只说明 manager 完成 atomicMin，
            // 不能说明 work warp 已经领取并处理这些顶点。
            for (int w = lane_id; w < dwords; w += WARP_SIZE)
                if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w]) != 0)
                    dirty_empty = false;
            // 普通终止轮只检查 ready_hint，允许 hint 假阳性；pre_ready
            // 路径会再做 ready bitmap 权威全扫。
            if (async_rx_ready_hint != NULL)
            {
                int ready_hwords = (dwords + 31) / 32;
                for (int w = lane_id; w < ready_hwords; w += WARP_SIZE)
                    if (*(volatile unsigned *)&async_rx_ready_hint[w] != 0)
                        async_rx_ready_empty = false;
            }
            else if (async_rx_ready_bitmap != NULL)
            {
                for (int w = lane_id; w < dwords; w += WARP_SIZE)
                    if (*(volatile unsigned *)&async_rx_ready_bitmap[w] != 0)
                        async_rx_ready_empty = false;
            }
#endif
            __syncwarp();

#if (L3_LIVE_SNAPSHOT == true)
            l3_live_manager_lane_state(lane_id, 31, (int)l2_empty);
#endif
            // 兜底扫描：node_data < last_processed -> 重新置脏位（防脏位丢失），lane 并行
            // 仅跨卡模式启用（n=1 无跨卡改进，脏位协议即完整正确性）
            // E1a 降频：每 BACKSTOP_K 轮全区间扫描一次（非扫描轮假定兜底一致，
            //   由 dirty 主路径保证；终止判定前强制补扫，见下方 pre_ready 分支）
            // SEED_BARRIER: 接收卡 seed_ready 前禁扫（node_data 仍是 DIST_MAX 本无害，
            //   但灌值窗口内扫描会命中部分种子 → 提前处理 → 重处理，须屏障后扫）
            bool backstop_ok = true;
            bool periodic_delegated = false;
#if (L3_ACK_SCAN == true)
            if(n_gpu > 1 && backstop_round % BACKSTOP_K == 0) {
                int token=0;
                if(!lane_id && l3_term_req) token=l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req);
                token=__shfl_sync(FULL_MASK,token,0);
                // A pending termination request already mandates a complete
                // local scan after all producer ACKs, before READY. Do not
                // insert a blocking single-warp scan while waiting for ACKs.
                // Cancellation returns to the active recovery path.
                periodic_delegated = token!=0;
#if (L3_PROGRESS_DIAG == true)
                if(!lane_id && periodic_delegated) ++g_l3_progress.periodic_delegated;
#endif
            }
#endif
#if (L3_BOUNDED_RECOVERY == true)
            bool periodic_check_complete = false;
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == false && \
     GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == false)
#if (SEED_BARRIER == true)
            if (bs_skip) atomicAdd(&g_bs_gate_skip, 1ull);
            if (n_gpu > 1 && !bs_skip && (backstop_round % BACKSTOP_K == 0))
#else
            if (n_gpu > 1 && (backstop_round % BACKSTOP_K == 0) && !periodic_delegated)
#endif
            {
                int scan_begin = 1, scan_end = v_local + 1;
#if (L3_PROGRESS_DIAG == true)
                const unsigned long long periodic_started = clock64();
#if (L3_ACK_SCAN == true)
                if(!lane_id) {
                    const int token=l3_term_req ? l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req) : 0;
                    if(!token) ++g_l3_progress.periodic_active;
                    else if(token==ack_scan_checked_token) ++g_l3_progress.periodic_certified;
                    else ++g_l3_progress.periodic_uncertified;
                }
#endif
#endif
#if (L3_BOUNDED_RECOVERY == true)
                int recovery_term_req = 0;
                if (!lane_id && l3_term_req != NULL)
                    recovery_term_req = l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req);
                recovery_term_req = __shfl_sync(FULL_MASK, recovery_term_req, 0);
                if (recovery_term_req == 0) {
                    const auto slice = recovery_cursor.take(v_local, RECOVERY_VERTEX_BUDGET);
                    scan_begin = slice.begin;
                    scan_end = slice.end;
                }
                // A full cycle assembled from old slices is NOT a certificate.
                // Frozen requests retain the original single-warp full scan.
                periodic_check_complete = scan_begin == 1 && scan_end == v_local + 1;
#endif
                if (backstop_scan_range(node_data, last_processed, dirty_bitmap, dirty_hint, scan_begin, scan_end, lane_id))
                {
                    dirty_empty = false;
                    backstop_ok = false;
                }
#if (L3_PROGRESS_DIAG == true)
                if (!lane_id) {
                    ++g_l3_progress.periodic_calls;
                    g_l3_progress.periodic_vertices += scan_end - scan_begin;
                    g_l3_progress.periodic_cycles += clock64() - periodic_started;
                }
#endif
            }
#endif
            __syncwarp();
#if (L3_LIVE_SNAPSHOT == true)
            l3_live_manager_lane_state(lane_id, 32, (int)l2_empty);
#endif
#if (MANAGE_PROFILE == true)
            unsigned p_t2 = clock();
#endif
            unsigned de_mask = __ballot_sync(FULL_MASK, dirty_empty);
            unsigned bs_mask = __ballot_sync(FULL_MASK, backstop_ok);
            bool all_dirty_empty = (de_mask == FULL_MASK);
            bool all_backstop_ok = (bs_mask == FULL_MASK);
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
            unsigned ar_mask = __ballot_sync(FULL_MASK, async_rx_ready_empty);
            bool all_async_rx_ready_empty = (ar_mask == FULL_MASK);
#else
            bool all_async_rx_ready_empty = true;
#endif

            // v3: remote_mark 全 0（本卡发送侧候选已全部被 L3 flush 消费）
#if (L3_LIVE_SNAPSHOT == true)
            l3_live_manager_lane_state(lane_id, 40, (int)l2_empty);
#endif
#if (MANAGE_PROFILE == true)
            unsigned t_m0 = clock();
#endif
            bool mark_empty = true;
#if (GLOBAL_ROUND_ASYNC == true)
            // bank pending/state 是异步候选协议的权威空闲条件；不能只检查
            // bank0，否则 active bank=1 时可能在仍有候选的情况下提前终止。
            mark_empty = async_candidate_quiet(cand_ctl, cand_bank0, cand_bank1);
#else
            if (remote_mark != NULL)
            {
                int mark_words = (l3_candidate_size(peer_v_local) + 31) / 32;
#if (BULK_ROUND == true)
                // 小/中分区直接检查权威 mark。BULK 的单层 L3 扫描会消费
                // remote_mark，但旧的 mark_hint/mark_hint2 可能保留 stale bit；
                // 继续依赖 summary 会把已经收敛的双卡拖进数十万次终止轮询。
                // 大图仍使用分层 hint，避免每轮扫描整个远程分区。
                if (mark_words <= 1024)
                {
                    mark_empty = l3_marks_empty_warp(remote_mark, mark_words, lane_id);
                }
                else
#endif
#if (SEED_BARRIER == true)
                // 接收卡 need_mark=false 时 mark 全扫纯浪费（结果不被使用）→ 跳过
                if (need_mark)
#endif
                {
                // B 开销优化: mark_empty 检查降频——默认轮用 mark_hint2 稀疏快速判断
                //   （13万 words → 128 words）；每 MARK_FULL_K 轮 + 源卡 phase2（seed_ready
                //   前正确性关键）强制全扫。终止正确性由二次确认（peer_idle 后 mark 全扫）
                //   兜底（见 term_ok 块），故快速路径的 hint 假阴不会导致提前终止。
                const int MARK_FULL_K = 2048;
#if (ASYNC_FB == true)
                bool need_full = (backstop_round % MARK_FULL_K == 0)
                              || (sb_active && is_src_card && ph == 2);
                if (!need_full && mark_hint2 != NULL)
                {
                    int mh2_words = (((mark_words + 31) / 32) + 31) / 32;
                    bool hint2_nonempty = false;
                    for (int w = lane_id; w < mh2_words; w += 32)
                        if (l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint2[w]) != 0) { hint2_nonempty = true; break; }
                    // mark_hint2 is only a sparse wake-up summary.  It may retain
                    // stale positive bits after the authoritative remote_mark word
                    // has been consumed.  A positive summary must therefore be
                    // validated against remote_mark before it can block termination.
                    unsigned hint2_mask = __ballot_sync(FULL_MASK, hint2_nonempty);
                    if (hint2_mask != 0)
                    {
                        mark_empty = true;
                        mark_empty = l3_marks_empty_warp(remote_mark, mark_words, lane_id);
                    }
                }
                else
                {
                    mark_empty = l3_marks_empty_warp(remote_mark, mark_words, lane_id);
                }
#else
                // SEED_BARRIER: hint2 快速 + 每 MARK_FULL_K 轮全扫；phase2 不再每轮全扫
                //   （warp0 phase2 开销 ~9ms，灌值 22ms 的一部分）；seed_ready 精确性由
                //   phase2→3 转换轮的全扫确认兜底（见 ph==2 状态机 dmm 检查）。
                bool need_full = (backstop_round % MARK_FULL_K == 0);
                if (!need_full && mark_hint2 != NULL)
                {
                    int mh2_words = (((mark_words + 31) / 32) + 31) / 32;
                    bool hint2_nonempty = false;
#if (L3_LIVE_SNAPSHOT == true)
                    l3_live_manager_lane_state(lane_id, 401, (int)l2_empty);
#endif
                    for (int w = lane_id; w < mh2_words; w += 32)
                        if (l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint2[w]) != 0) { hint2_nonempty = true; break; }
#if (L3_LIVE_SNAPSHOT == true)
                    l3_live_manager_lane_state(lane_id, 402, (int)l2_empty);
#endif
                    // mark_hint2 is only a sparse wake-up summary.  Validate a
                    // positive summary against the authoritative remote_mark so
                    // stale summary bits cannot keep the termination handshake
                    // cycling after the L3 candidates are already drained.
                    unsigned hint2_mask = __ballot_sync(FULL_MASK, hint2_nonempty);
#if (L3_LIVE_SNAPSHOT == true)
                    l3_live_manager_lane_state(lane_id, 403, (int)l2_empty);
#endif
                    if (hint2_mask != 0)
                    {
                        mark_empty = true;
#if (L3_LIVE_SNAPSHOT == true)
                        l3_live_manager_lane_state(lane_id, 404, (int)l2_empty);
#endif
                        mark_empty = l3_marks_empty_warp(remote_mark, mark_words, lane_id);
#if (L3_LIVE_SNAPSHOT == true)
                        l3_live_manager_lane_state(lane_id, 405, (int)l2_empty);
#endif
                    }
                }
                else
                {
#if (L3_LIVE_SNAPSHOT == true)
                    l3_live_manager_lane_state(lane_id, 410, (int)l2_empty);
#endif
                    mark_empty = l3_marks_empty_warp(remote_mark, mark_words, lane_id);
#if (L3_LIVE_SNAPSHOT == true)
                    l3_live_manager_lane_state(lane_id, 411, (int)l2_empty);
#endif
                }
#endif
                }
            }
#endif
            __syncwarp();
#if (L3_LIVE_SNAPSHOT == true)
            l3_live_manager_lane_state(lane_id, 41, (int)l2_empty);
#endif
#if (MANAGE_PROFILE == true)
            if (!lane_id) p_mark += (unsigned)clock() - t_m0;
#endif
            unsigned mk_mask = __ballot_sync(FULL_MASK, mark_empty);
            bool all_mark_empty = (mk_mask == FULL_MASK);
#if (GHOST_DEPTH > 0)
            // 方案 C: ghost 改进信号全 0（ghost 处理在途未完成不算收敛；仅源卡非 NULL）
            bool ghost_mark_empty = true;
            if (ghost_mark != NULL)
            {
                int ghost_words = (ghost_num + 31) / 32;
                for (int w = lane_id; w < ghost_words; w += 32)
                    if (l3_atomic_load_acquire<cuda::thread_scope_device>(&ghost_mark[w]) != 0) ghost_mark_empty = false;
            }
            __syncwarp();
            unsigned gm_mask = __ballot_sync(FULL_MASK, ghost_mark_empty);
            bool all_ghost_mark_empty = (gm_mask == FULL_MASK);
#else
            bool all_ghost_mark_empty = true;
#endif
            // v3: L3 flush 无在途批量（跨卡写在 fence+dirty 完成前不算 idle）。
            // phase2 并行发布改读发布 warp 计数，避免 phase2→3 越过在途发布。
            __threadfence_block();
            int l3_busy_now = 0;
            if (!lane_id)
                l3_busy_now = atomicAdd(&l3_busy, 0);
#if (SEED_BARRIER == true && SEED_PHASE2_PARALLEL == true)
            if (!lane_id && sb_active && is_src_card && ph == 2)
                l3_busy_now = atomicAdd(&seed_pub_busy, 0);
#endif
            l3_busy_now = __shfl_sync(FULL_MASK, l3_busy_now, 0);
            __threadfence_block();
            bool l3_ok = (l3_busy_now == 0);
#if (BULK_ROUND == true)
            bool bulk_inbox_empty = true;
            if (n_gpu > 1 && bulk_inbox_epoch != NULL)
                bulk_inbox_empty = !bulk_inbox_has_next(
                    bulk_inbox_epoch, bulk_inbox_state,
                    bulk_inbox_generation, bulk_rx_epoch);
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
            // active_slot 表示 manager 已把 READY 槽交给 work warp；即使
            // 当前 read_head 已覆盖 count，也必须等待该 slot 的 inflight
            // claim 完成并转为 DONE，不能仅依据“下一个 READY epoch”判空。
            int bulk_active_slot_now = -1;
            if (!lane_id && n_gpu > 1 && bulk_inbox_active_slot != NULL)
                bulk_active_slot_now = atomicAdd(bulk_inbox_active_slot, 0);
            bulk_active_slot_now = __shfl_sync(
                FULL_MASK, bulk_active_slot_now, 0);
            if (bulk_active_slot_now >= 0)
                bulk_inbox_empty = false;
#endif
            bool bulk_outbox_empty = true;
            if (n_gpu > 1 && peer_bulk_inbox_ack != NULL)
            {
                if (!lane_id)
                {
                    int published = atomicAdd(
                        (int *)&bulk_tx_epoch_published, 0);
                    if (published > 0)
                    {
                        int published_slot = bulk_inbox_slot(published);
                        bulk_outbox_empty =
                            (l3_atomic_load_acquire<cuda::thread_scope_system>(peer_bulk_inbox_ack + published_slot)
                             >= published);
                    }
                }
            }
            bulk_outbox_empty = __shfl_sync(
                FULL_MASK, bulk_outbox_empty, 0);
#else
            bool bulk_inbox_empty = true;
            bool bulk_outbox_empty = true;
#endif
#if (L3_RX_EXPRESS == true)
            // head advances only after global_wid==0 finishes
            // simple_process, so this one snapshot covers both published and
            // held express work for termination.
            bool rx_express_empty = true;
            if (!lane_id)
                rx_express_empty = l3_rx_express_empty(rx_express);
            rx_express_empty = __shfl_sync(
                FULL_MASK, rx_express_empty, 0);
#endif

            // inj_pending 与下方多个终止分支共享同一个 warp-uniform 快照。
            // producer 若在快照后开始工作，会由终止握手 token/ACK 再确认。
            int inj_pending_now = 0;
            if (!lane_id)
                inj_pending_now = atomicAdd(&inj_pending, 0);
            inj_pending_now = __shfl_sync(FULL_MASK, inj_pending_now, 0);

            // E1a: 终止预判前强制补扫 backstop（本轮非扫描轮且其余条件全就绪时，
            //   必须核对 last_processed==node_data 才允许终止，防降频漏兜底）
#if (SEED_BARRIER == true)
            bool pre_ready = l2_empty && all_dirty_empty && (!need_mark || all_mark_empty) && all_ghost_mark_empty && l3_ok
                           && bulk_inbox_empty && bulk_outbox_empty
#if (L3_RX_EXPRESS == true)
                           && rx_express_empty
#endif
                           ;
#else
            bool pre_ready = l2_empty && all_dirty_empty && all_mark_empty && all_ghost_mark_empty && l3_ok
                           && bulk_inbox_empty && bulk_outbox_empty
#if (L3_RX_EXPRESS == true)
                           && rx_express_empty
#endif
                           ;
#endif
            bool confirm_allowed = confirm_policy.allow(pre_ready);
            if (pre_ready) {
                int ca = confirm_allowed ? 1 : 0;
                ca = __shfl_sync(FULL_MASK, ca, 0);
                confirm_allowed = ca != 0;
            }
#if (L3_LIVE_SNAPSHOT == true)
            l3_live_manager_lane_state(lane_id, 45, (int)l2_empty);
#endif
#if (BULK_ROUND == true && L3_TERM_HANDSHAKE == true && \
     BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && \
     SEED_BARRIER == false)
            // 终止握手请求置位后，注入 warp 可能已经停在 token ACK 等待中；
            // manager 不能再发起 backstop_collaborative，否则会等待 bs_done，
            // 而注入 warp不会再响应新的 backstop 请求，形成协议内死锁。
            int l3_term_probe_req = 0;
            if (!lane_id && l3_term_req != NULL)
                l3_term_probe_req = l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req);
            l3_term_probe_req = __shfl_sync(
                FULL_MASK, l3_term_probe_req, 0);
            bool l3_term_probe_free = (l3_term_probe_req == 0);
#else
            bool l3_term_probe_free = true;
#endif
#if (SEED_BARRIER == true)
            if (pre_ready && l3_term_probe_free && n_gpu > 1 && !bs_skip && (backstop_round % BACKSTOP_K != 0))
#else
            if (pre_ready && confirm_allowed && l3_term_probe_free && n_gpu > 1
                && !L3_ACK_SCAN
                && ((backstop_round % BACKSTOP_K != 0)
#if (L3_BOUNDED_RECOVERY == true)
                    || !periodic_check_complete
#endif
                    ))
#endif
            {
#if (MANAGE_PROFILE == true)
                unsigned t_c0 = clock();
                if (!lane_id) p_pre_cnt++;
#endif
                // SEED_BARRIER: pre_ready 块内 mark 全扫（hint2 快速路径的终止精确性兜底）
#if (SEED_BARRIER == true)
                if (remote_mark != NULL && need_mark)
                {
                    int mkw = (peer_v_local + 31) / 32;
                    bool m_ok = true;
                    for (int w = lane_id; w < mkw; w += 32)
                        if (l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w])) { m_ok = false; break; }
                    __syncwarp();
                    unsigned mm = __ballot_sync(FULL_MASK, m_ok);
                    if (mm != FULL_MASK) all_mark_empty = false;
                }
#endif
                backstop_ok = true;
                if (backstop_collaborative(node_data, last_processed, dirty_bitmap, dirty_hint,
                                           &bs_req, &bs_done, &bs_found, v_local, lane_id
#if (L3_WORKER_RECOVERY == true)
                                           , bulk_work_warp_total / WARP_NUM_PER_BLOCK
#endif
#if (L3_RECOVERY_MODE > 0)
                                           , L3_RECOVERY_MODE == 2 ? mlmq.manage_warp_num() : 0, &bs_epoch
#endif
                                           ))
                {
                    dirty_empty = false;
                    backstop_ok = false;
                }
                __syncwarp();
#if (MANAGE_PROFILE == true)
                if (!lane_id) p_confirm += (unsigned)clock() - t_c0;
#endif
                de_mask = __ballot_sync(FULL_MASK, dirty_empty);
                bs_mask = __ballot_sync(FULL_MASK, backstop_ok);
                all_dirty_empty = (de_mask == FULL_MASK);
                all_backstop_ok = (bs_mask == FULL_MASK);
            }

#if (SEED_BARRIER == true)
            // ===== 解法0: 源卡 phase 状态机 =====
            // phase=1（计算期）: 本地收敛（l2_empty && dirty_empty && backstop_ok，
            //   不含 mark_empty——mark 是待灌的最终值信号）→ phase=2
            // phase=2（灌值期）: 全收敛 + mark 全空 + l3 空闲（w17 已把全部最终值写入
            //   peer node_data）→ __threadfence_system → P2P 置 peer seed_ready → phase=3
            // phase=3: 走下方正常终止流程
            if (sb_active && is_src_card)
            {
                if (ph == 1)
                {
                    // 本地收敛（不含 mark_empty——mark 是待灌最终值信号）→ phase=2。
                    // 注：本判定可提前（瞬时 l2_empty），但 seed_ready 需全收敛（phase2 条件）。
                    //   提前触发无害：w17 渐进灌值，晚到 mark 会被再 drain。
                    if (l2_empty && all_dirty_empty)
                    {
                        // phase1→2 全扫确认（hint 快速路径的精确性兜底；仅候选轮执行）
                        bool d_ok = true;
                        for (int w = lane_id; w < dwords; w += 32)
                            if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w])) { d_ok = false; break; }
                        __syncwarp();
                        unsigned dm2 = __ballot_sync(FULL_MASK, d_ok);
                        if (dm2 == FULL_MASK && !lane_id)
                        {
                            atomicExch(phase, 2);
                            atomicExch(&g_t_phase2, TIMELINE_CLOCK());
                        }
                    }
                }
                else if (ph == 2)
                {
                    // 并行发布允许 mark_hint 保留陈旧假阳；phase2 的精确性由下面
                    // 对 remote_mark 的全扫确认保证，不能把 all_mark_empty 作为前置门槛。
#if (SEED_PHASE2_PARALLEL == true)
                    if (l2_empty && all_dirty_empty && l3_ok
#else
                    if (l2_empty && all_dirty_empty && all_mark_empty && l3_ok
#endif
                        && all_ghost_mark_empty
                        && inj_pending_now == 0)
                    {
                        // phase2→3 全扫确认（dirty/mark hint 快速路径的精确性兜底；仅候选轮）
                        bool d_ok = true, m_ok = true;
                        for (int w = lane_id; w < dwords; w += 32)
                            if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w])) { d_ok = false; break; }
                        if (remote_mark != NULL && need_mark)
                        {
                            int mkw = (peer_v_local + 31) / 32;
                            for (int w = lane_id; w < mkw; w += 32)
                                if (l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w])) { m_ok = false; break; }
                        }
                        __syncwarp();
                        unsigned dmm = __ballot_sync(FULL_MASK, d_ok && m_ok);
                        if (dmm == FULL_MASK)
                        {
                            // 并行发布路径先把源卡本地紧凑列表复制到接收卡，
                            // 再通过 system fence 发布 seed_ready。
#if (SEED_PHASE2_PARALLEL == true)
                            int seed_cnt = atomicAdd(seed_list_cnt, 0);
                            if (seed_cnt < 0) seed_cnt = 0;
                            if (seed_cnt > peer_v_local) seed_cnt = peer_v_local;
                            if (peer_seed_list != NULL && peer_seed_list_cnt != NULL)
                            {
                                for (int s = lane_id; s < seed_cnt; s += WARP_SIZE)
                                    peer_seed_list[s] = seed_list[s];
                                __syncwarp();
                                __threadfence_system();
                                if (!lane_id)
                                    atomicExch(peer_seed_list_cnt, seed_cnt);
                                __syncwarp();
                            }
#endif
                            // 屏障: 先保证全部 seed/node_data 写对 peer 可见，再置 seed_ready
                            __threadfence_system();
                            if (!lane_id)
                            {
                                atomicExch(peer_seed_ready, 1);
                                __threadfence_system();
                                atomicExch(phase, 3);
                                atomicAdd(&g_seed_ready_fired, 1ull);
                                atomicExch(&g_t_seed_ready, TIMELINE_CLOCK());
                            }
                            __threadfence_system();
                            __syncwarp();
                        }
                    }
                }
                __threadfence_block();
            }
#endif

            // ASYNC_FB（方案 B）: 终止前 quiet 窗口——双方"真实跨卡改进"计数均稳定
            //   （无真实改进落位）连续 EFF_QUIET_K 轮才允许置 idle。任何一侧真实 flush
            //   都改变计数 → 重置快照+quiet 计数，等下一轮稳定。防止"反馈改进在途/
            //   刚落位而本地条件瞬时为空"导致的提前终止。
            bool eff_ok = true;
#if (SEED_BARRIER == true && ASYNC_FB == true)
            if (sb_active)
            {
                // B 开销优化: quiet 窗口 4→2 轮（终检 eff_final 仍兜底在途改进）
                const int EFF_QUIET_K = 2;
                if (!lane_id)
                {
                    unsigned long long my_eff = atomicAdd(remote_eff, 0ull);
                    unsigned long long p_eff = atomicAdd(peer_remote_eff, 0ull);
                    if (my_eff != s_eff_base || p_eff != s_peer_eff_base)
                    {
                        s_eff_base = my_eff;
                        s_peer_eff_base = p_eff;
                        s_eff_quiet = 0;
                        eff_ok = false;
                    }
                    else
                    {
                        s_eff_quiet++;
                        if (s_eff_quiet < EFF_QUIET_K) eff_ok = false;
                    }
                }
                __syncwarp();
                eff_ok = __shfl_sync(FULL_MASK, eff_ok, 0);
            }
#endif

#if (GLOBAL_ROUND_ASYNC == true)
            // WINDOWED_ASYNC/ASYNC 不在每轮 pack 前 quiesce，因此上面的
            // l2_empty 只代表 manager 的全局队列为空，不能代表 work warp
            // 私有 node_in/on_the_fly/local queue 已经排空。终止前发起一次
            // tokenized drain probe；work warp ACK 后才允许进入 peer-idle
            // 二次确认。若 probe 期间出现新候选/消息，撤销请求并恢复工作。
            bool async_term_candidate =
                (l2_empty && node_in_num == 0 && all_dirty_empty
                 && all_backstop_ok && all_mark_empty
                 && all_ghost_mark_empty && l3_ok
                 && bulk_inbox_empty && bulk_outbox_empty
                 && inj_pending_now == 0);
            int async_req_token = 0;
            int async_work_ack = 0;
            int async_peer_idle_1 = 1;
            if (!lane_id)
            {
                if (bulk_quiesce_req != NULL)
                    async_req_token = l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req);
                if (bulk_quiesce_ack != NULL)
                    async_work_ack = l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_ack);
                if (peer_local_idle != NULL)
                    async_peer_idle_1 = l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle);
            }
            async_req_token = __shfl_sync(FULL_MASK, async_req_token, 0);
            async_work_ack = __shfl_sync(FULL_MASK, async_work_ack, 0);
            async_peer_idle_1 = __shfl_sync(FULL_MASK, async_peer_idle_1, 0);
            __threadfence_system();
            int async_peer_idle_2 = 1;
            if (!lane_id && peer_local_idle != NULL)
                async_peer_idle_2 = l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle);
            async_peer_idle_2 = __shfl_sync(
                FULL_MASK, async_peer_idle_2, 0);
            if (n_gpu > 1 && bulk_quiesce_req != NULL
                && bulk_quiesce_ack != NULL)
            {
                if (async_req_token == 0 && async_term_candidate)
                {
                    if (!lane_id)
                    {
                        // 不再要求先看到 peer idle 才能发起 probe。双方都可
                        // 独立发布自己的排空意图；若对端随后发布最后一批
                        // inbox，manager 会撤销 probe、接收消息并恢复 work，
                        // 避免“双方都等待对端先 idle”的对称互等。
                        timeline_idle_write(local_idle, 1);
                        ++async_quiesce_token;
                        if (async_quiesce_token == 0)
                            ++async_quiesce_token;
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, async_quiesce_token);
                        timeline_idle_write_reason(local_idle, 0, 40);
#if (BULK_DIAG == true)
                        printf("ASYNC_TERM_REQ g%d token=%d peer=%d/%d ack=0 q=%d\\n",
                               v_begin, async_quiesce_token,
                               async_peer_idle_1, async_peer_idle_2,
                               (int)mlmq.get_global_queue_size());
#endif
                    }
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }

                if (async_req_token != 0 && !async_term_candidate)
                {
                    if (!lane_id)
                    {
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_req, 0);
                        l3_atomic_store_release<cuda::thread_scope_device>(bulk_quiesce_ack, 0);
                        timeline_idle_write_reason(local_idle, 0, 41);
#if (BULK_DIAG == true)
                        printf("ASYNC_TERM_CANCEL g%d token=%d peer=%d/%d q=%d dirty=%d mark=%d\\n",
                               v_begin, async_req_token,
                               async_peer_idle_1, async_peer_idle_2,
                               (int)mlmq.get_global_queue_size(),
                               (int)!all_dirty_empty, (int)!all_mark_empty);
#endif
                    }
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }

                if (async_req_token != 0
                    && async_work_ack < bulk_work_warp_total)
                {
                    __threadfence();
                    continue;
                }
            }
#endif

#if (BULK_ROUND == true && L3_TERM_HANDSHAKE == true && \
     BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && \
     SEED_BARRIER == false)
            // 默认 BULK 的独立终止握手。先用当前完整谓词发现“可能收敛”，
            // 再冻结所有 work/injection producer；只有所有 token ACK 和双方
            // READY 都成立时才写 global_exit。
            if (n_gpu > 1 && l3_term_req != NULL && l3_term_state != NULL
                && l3_term_ack_slots != NULL && peer_l3_term_state != NULL)
            {
#if (L3_LIVE_SNAPSHOT == true)
                l3_live_manager_lane_state(lane_id, 50, (int)l2_empty);
#endif
                bool term_local_candidate =
                    l2_empty && all_dirty_empty && all_backstop_ok
                    && all_mark_empty && all_ghost_mark_empty && l3_ok
                    && bulk_inbox_empty && bulk_outbox_empty
#if (L3_RX_EXPRESS == true)
                    && rx_express_empty
#endif
                    && inj_pending_now == 0
                    && node_in_num == 0
#if (L3_TILE_LOAN == true)
                    && !l3_tile_loan_any_active(l3_channel, lane_id);
#else
                    ;
#endif
                int term_state_now = L3_TERM_ACTIVE;
                int term_req_now = 0;
#if (L3_WAIT_DIAG == true)
                bool wait_exact=true, wait_drained=true, wait_recovered=false;
#endif
                if (!lane_id)
                {
                    term_state_now = l3_atomic_load_acquire<cuda::thread_scope_system>(l3_term_state);
                    term_req_now = l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req);
                }
                term_state_now = __shfl_sync(
                    FULL_MASK, term_state_now, 0);
                term_req_now = __shfl_sync(FULL_MASK, term_req_now, 0);
#if (L3_FEEDBACK_DRAIN == true)
                int drain_pending = 0;
                if (!lane_id)
                    drain_pending = l3_atomic_load_acquire<cuda::thread_scope_block>(
                        &l3_drain_requested);
                drain_pending = __shfl_sync(FULL_MASK, drain_pending, 0);
                // Let TX service the request before trying to freeze it again.
                term_local_candidate = term_local_candidate && !drain_pending;
#endif
#if (HANG_DIAG == true)
                if (!lane_id && term_state_now != L3_TERM_ACTIVE
                    && !diag_term_seen)
                {
                    printf("L3_TERM_STATE g%d req=%d state=%d candidate=%d peer=%d out=%d\n",
                           v_begin, term_req_now, term_state_now,
                           (int)term_local_candidate,
                           (peer_l3_term_state != NULL)
                               ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_l3_term_state) : -1,
                           (int)bulk_outbox_empty);
                    diag_term_seen = true;
                }
#endif

                if (term_state_now == L3_TERM_ACTIVE)
                {
                    if (term_local_candidate)
                    {
                        if (!lane_id)
                        {
                            ++l3_term_token;
#if (L3_PROGRESS_DIAG == true)
                            ++g_l3_progress.term_requests;
#endif
                            if (l3_term_token <= 0) l3_term_token = 1;
                            l3_atomic_store_release<cuda::thread_scope_system>(l3_term_state, static_cast<int>(L3_TERM_QUIESCING));
                            l3_atomic_store_release<cuda::thread_scope_device>(l3_term_req, l3_term_token);
#if (L3_EVENT_RING == true)
                            l3_event_push(L3_EVENT_TERM_REQUEST,
                                          l3_term_token,
                                          (int)term_local_candidate,
                                          (int)bulk_inbox_empty,
                                          (int)bulk_outbox_empty);
#endif
                            timeline_idle_write_reason(local_idle, 0, 40);
                        }
                        __threadfence_system();
                        __syncwarp();
                        continue;
                    }
                }
                else
                {
                    bool work_ack_ok = (term_req_now != 0);
                    if (work_ack_ok)
                    {
                        for (int i = lane_id; i < bulk_work_warp_total; i += WARP_SIZE)
                        {
                            if (l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_ack_slots + i) != term_req_now)
                            {
                                work_ack_ok = false;
                                break;
                            }
                        }
                    }
                    unsigned work_ack_mask = __ballot_sync(FULL_MASK, work_ack_ok);
                    bool all_work_ack = (work_ack_mask == FULL_MASK);

                    bool inject_ack_ok = (term_req_now != 0);
                    for (int i = lane_id; i < INJECT_WARP_NUM; i += WARP_SIZE)
                    {
                        if (l3_atomic_load_acquire<cuda::thread_scope_block>(l3_term_inject_ack + i) != term_req_now)
                        {
                            inject_ack_ok = false;
                            break;
                        }
                    }
                    unsigned inject_ack_mask = __ballot_sync(FULL_MASK, inject_ack_ok);
                    bool all_inject_ack = (inject_ack_mask == FULL_MASK);

                    int tx_ack = 0;
                    if (!lane_id) tx_ack = l3_atomic_load_acquire<cuda::thread_scope_block>(&l3_term_tx_ack);
                    tx_ack = __shfl_sync(FULL_MASK, tx_ack, 0);
                    bool all_tx_ack = term_req_now != 0 && tx_ack == term_req_now;

                    // Advisory hints can have a false negative after a racing
                    // clear. Once producers ACK, verify authoritative marks
                    // before READY/exit rather than trusting the earlier hint.
#if (L3_TEST_SKIP_FROZEN_MARK == false)
                    if (all_work_ack && all_inject_ack && all_tx_ack)
                    {
                        bool exact_mark_empty = l3_marks_empty_warp(remote_mark,
                            (peer_v_local + 31) / 32, lane_id);
#if (L3_WAIT_DIAG == true)
                        wait_exact=exact_mark_empty;
#endif
                        term_local_candidate = term_local_candidate && exact_mark_empty;
#if (L3_FEEDBACK_DRAIN == true)
                        if (!lane_id && !exact_mark_empty)
                            l3_atomic_store_release<cuda::thread_scope_block>(&l3_drain_requested, 1);
#endif
                        // Earlier busy/outbox snapshots may precede TX's last
                        // claim/publication. Read the published epoch AFTER its
                        // ACK; it cannot change until this request is cancelled.
                        int tx_drained = 1;
                        if (!lane_id) {
                            int published = l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published);
                            tx_drained = atomicAdd(&l3_busy, 0) == 0;
                            if (published > 0)
                                tx_drained = tx_drained && peer_bulk_inbox_ack != NULL
                                    && l3_atomic_load_acquire<cuda::thread_scope_system>(peer_bulk_inbox_ack + bulk_inbox_slot(published)) >= published;
                        }
                        tx_drained = __shfl_sync(FULL_MASK, tx_drained, 0);
#if (L3_WAIT_DIAG == true)
                        wait_drained=tx_drained;
#endif
                        term_local_candidate = term_local_candidate && tx_drained;
                    }
#endif

                    int peer_state_now = L3_TERM_ACTIVE;
                    if (!lane_id)
                        peer_state_now = l3_atomic_load_acquire<cuda::thread_scope_system>(peer_l3_term_state);
                    peer_state_now = __shfl_sync(FULL_MASK, peer_state_now, 0);

#if (L3_ACK_SCAN == true)
                    // Local-only frozen scan. The sole token writer (manager)
                    // stays here until every helper completes, so cancellation
                    // and request-storage reuse cannot race an unfinished scan.
                    // Peer ACTIVE does not prevent local recovery progress.
                    if (term_state_now == L3_TERM_QUIESCING && term_req_now &&
                        term_local_candidate && all_work_ack && all_inject_ack &&
                        all_tx_ack && ack_scan_checked_token != term_req_now) {
#if (L3_PROGRESS_DIAG == true)
                        const unsigned long long ack_scan_start = clock64();
#endif
                        bool found = l3_worker_recovery_request(dirty_bitmap,dirty_hint,
#if (L3_ACK_WIDE_SCAN == true)
                            bulk_work_warp_total,lane_id,true);
#else
                            bulk_work_warp_total / WARP_NUM_PER_BLOCK,lane_id);
#endif
#if (L3_PROGRESS_DIAG == true)
                        if(!lane_id) {
                            g_l3_progress.backstop_cycles += clock64()-ack_scan_start;
                            ++g_l3_progress.backstop_calls;
                            g_l3_progress.backstop_positive += found;
                        }
#endif
                        if(found) term_local_candidate = false;
                        else if(!lane_id) ack_scan_checked_token = term_req_now;
#if (L3_WAIT_DIAG == true)
                        wait_recovered=found;
#endif
                        __syncwarp();
                    }
#endif
#if (L3_TERM_WAIT_ACK == true)
                    // Pending ACK is not evidence of new work. Retain the
                    // current token while QUIESCING, but return to the outer
                    // manager loop so RX and the full activity predicate keep
                    // progressing. No READY or exit is allowed on this path.
                    if (term_state_now == L3_TERM_QUIESCING
                        && term_req_now != 0 && term_local_candidate
                        && (!all_work_ack || !all_inject_ack || !all_tx_ack))
                    {
#if (L3_PROGRESS_DIAG == true)
                        if (!lane_id) ++g_l3_progress.term_ack_waits;
#endif
#if (L3_WAIT_DIAG == true)
                        if(!lane_id) {
                            g_l3_wait_reason[14]+=!all_work_ack;
                            g_l3_wait_reason[15]+=!all_inject_ack;
                            g_l3_wait_reason[16]+=!all_tx_ack;
                        }
#endif
                        __threadfence_system();
                        __syncwarp();
                        continue;
                    }
#endif
                    // 任一完整谓词失效，或对端撤销握手，都必须回到 ACTIVE。
                    // 这样在途 inbox/新 dirty 到达时，冻结的 producer 会恢复。
                    if (term_req_now == 0 || !term_local_candidate
                        || !all_work_ack || !all_inject_ack || !all_tx_ack
#if (L3_STICKY_READY == false)
                        || (term_state_now == L3_TERM_READY
                            && peer_state_now == L3_TERM_ACTIVE)
#endif
                        )
                    {
                        if (!lane_id)
                        {
#if (L3_PROGRESS_DIAG == true)
                            ++g_l3_progress.term_cancels;
#if (L3_WAIT_DIAG == true)
                            g_l3_wait_reason[0]+=term_req_now==0;
                            g_l3_wait_reason[1]+=!l2_empty;
                            g_l3_wait_reason[2]+=!all_dirty_empty;
                            g_l3_wait_reason[3]+=!all_backstop_ok;
                            g_l3_wait_reason[4]+=!all_mark_empty;
                            g_l3_wait_reason[5]+=!all_ghost_mark_empty;
                            g_l3_wait_reason[6]+=!l3_ok;
                            g_l3_wait_reason[7]+=!bulk_inbox_empty;
                            g_l3_wait_reason[8]+=!bulk_outbox_empty;
                            g_l3_wait_reason[9]+=inj_pending_now!=0;
                            g_l3_wait_reason[10]+=node_in_num!=0;
                            g_l3_wait_reason[11]+=!wait_exact;
                            g_l3_wait_reason[12]+=!wait_drained;
                            g_l3_wait_reason[13]+=wait_recovered;
                            ++g_l3_wait_reason[17];
#endif
                            if (term_req_now == 0 || !term_local_candidate)
                                ++g_l3_progress.term_invalid;
                            else if (!all_work_ack || !all_inject_ack || !all_tx_ack)
                                ++g_l3_progress.term_ack_pending;
                            else
                                ++g_l3_progress.term_peer_cancel;
#endif
                            l3_atomic_store_release<cuda::thread_scope_system>(l3_term_state, static_cast<int>(L3_TERM_ACTIVE));
                            l3_atomic_store_release<cuda::thread_scope_device>(l3_term_req, 0);
#if (L3_EVENT_RING == true)
                            l3_event_push(L3_EVENT_TERM_CANCEL,
                                          term_req_now,
                                          (int)term_local_candidate,
                                          (int)all_work_ack,
                                          (int)all_inject_ack);
#endif
                            timeline_idle_write_reason(local_idle, 0, 41);
                        }
                        __threadfence_system();
                        __syncwarp();
                        continue;
                    }

                    if (term_state_now == L3_TERM_QUIESCING)
                    {
                        if (!lane_id)
                        {
                            l3_atomic_store_release<cuda::thread_scope_system>(l3_term_state, static_cast<int>(L3_TERM_READY));
#if (L3_EVENT_RING == true)
                            l3_event_push(L3_EVENT_TERM_READY,
                                          term_req_now,
                                          peer_state_now,
                                          (int)all_work_ack,
                                          (int)all_inject_ack);
#endif
                            timeline_idle_write(local_idle, 1);
                        }
                        __threadfence_system();
                        __syncwarp();
                        term_state_now = L3_TERM_READY;
                    }

                    if (term_state_now == L3_TERM_READY)
                    {
                        int peer_ready = L3_TERM_ACTIVE;
                        if (!lane_id)
                            peer_ready = l3_atomic_load_acquire<cuda::thread_scope_system>(peer_l3_term_state);
                        peer_ready = __shfl_sync(FULL_MASK, peer_ready, 0);
                        if (peer_ready == L3_TERM_READY)
                        {
                            if (!lane_id)
                            {
#if (L3_EVENT_RING == true)
                                l3_event_push(L3_EVENT_TERM_EXIT,
                                              term_req_now,
                                              peer_ready,
                                              bulk_rx_epoch,
                                              l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published));
#endif
#if (HANG_DIAG == true)
                                printf("L3_TERM_HANDSHAKE_EXIT g%d req=%d state=%d peerstate=%d\n",
                                       v_begin,
                                       (l3_term_req != NULL) ? l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req) : -1,
                                       (l3_term_state != NULL) ? l3_atomic_load_acquire<cuda::thread_scope_system>(l3_term_state) : -1,
                                       (peer_l3_term_state != NULL)
                                           ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_l3_term_state) : -1);
#endif
                                l3_atomic_store_release<cuda::thread_scope_block>(&manager_end, 1);
                                l3_atomic_store_release<cuda::thread_scope_device>(global_exit, 1);
                                atomicExch(&g_t_term, TIMELINE_CLOCK());
                            }
                            __syncwarp();
                            continue;
                        }
                    }
                    __threadfence_system();
                    __syncwarp();
                    continue;
                }
            }
#endif

            // E2: 终止条件含 inj_pending（注入在途清零才可判定终止）
#if (L3_LIVE_SNAPSHOT == true)
            l3_live_manager_lane_state(lane_id, 60, (int)l2_empty);
#endif
#if (SEED_BARRIER == true)
            if (eff_ok && !term_skip && l2_empty && all_dirty_empty && all_backstop_ok
                && (!need_mark || all_mark_empty) && all_ghost_mark_empty && l3_ok
                && inj_pending_now == 0)
#else
            if (l2_empty && all_dirty_empty && all_backstop_ok && all_mark_empty && all_ghost_mark_empty && l3_ok
                && bulk_inbox_empty && bulk_outbox_empty
#if (L3_RX_EXPRESS == true)
                && rx_express_empty
#endif
                && inj_pending_now == 0
                && node_in_num == 0
#if (L3_TILE_LOAN == true)
                && !l3_tile_loan_any_active(l3_channel, lane_id))
#else
                )
#endif
#endif
            {
                if (!lane_id)
                {
                    // 置位本卡空闲标志，读 peer 空闲（P2P），双方 idle -> 终止
                    __threadfence_system();
                    timeline_idle_write(local_idle, 1);
                    if (!g_t_idle) atomicExch(&g_t_idle, TIMELINE_CLOCK());
                    __threadfence_system();
                    // n=1 时 peer_local_idle 为 NULL，视为 peer 已空闲（单卡无需握手）
                    // 跨卡读用原子读（绕过 L2 缓存，V0.2 验证原子可见）
                    peer_idle_s = (peer_local_idle == NULL) || (l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) == 1);
                }
                __syncwarp();
                // E2 二次确认：peer 已 idle -> 其最终 flush（atomicMin+置 dirty）必已可见。
                // 用全 lane 并行重查本卡 dirty + backstop 全区间（含 re-mark），防"peer 最终
                // 改进恰在本卡 backstop 补扫之后、读 peer_idle 之前落位"的漏兜窗口。
                // 注意：此块必须在 if(!lane_id) 之外（含 __syncwarp/__ballot_sync，防收敛违规）。
                if (*(volatile int *)&peer_idle_s)
                {
                    if (!g_t_peer_idle_seen) atomicExch(&g_t_peer_idle_seen, TIMELINE_CLOCK());
                    // P0（方案2）: 单卡无跨卡改进（无 fire-and-forget 绕过 last_processed 的写），
                    //   dirty 协议恒空、last_processed 收敛由 l2_empty 保证，二次确认全扫纯冗余。
                    //   双卡仍需二次确认（防 peer 最终改进落位漏兜窗口）。
                    bool term_ok;
                    int confirm_inj_pending_now = 0;
                    if (!lane_id)
                        confirm_inj_pending_now = atomicAdd(&inj_pending, 0);
                    confirm_inj_pending_now = __shfl_sync(
                        FULL_MASK, confirm_inj_pending_now, 0);
                    if (n_gpu == 1)
                    {
                        term_ok = (confirm_inj_pending_now == 0);
                    }
                    else
                    {
                        bool c_ok = true;
                        atomicAdd(&g_confirm_exec, 1ull);
#if (MANAGE_PROFILE == true)
                        unsigned t_c2 = clock();
                        if (!lane_id) p_confirm_cnt++;
#endif
                        for (int w = lane_id; w < dwords; w += 32)
                            if (l3_atomic_load_acquire<cuda::thread_scope_system>(&dirty_bitmap[w])) c_ok = false;
#if (ASYNC_FB == true)
                        // B 开销优化: 二次确认加 mark 全扫（快速路径 hint 假阴的终止兜底）
                        if (remote_mark != NULL)
                        {
                            int mark_words = (peer_v_local + 31) / 32;
                            for (int w = lane_id; w < mark_words; w += 32)
                                if (l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w])) c_ok = false;
                        }
#endif
                        if (backstop_collaborative(node_data, last_processed, dirty_bitmap, dirty_hint,
                                                   &bs_req, &bs_done, &bs_found, v_local, lane_id
#if (L3_WORKER_RECOVERY == true)
                                                   , bulk_work_warp_total / WARP_NUM_PER_BLOCK
#endif
#if (L3_RECOVERY_MODE > 0)
                                                   , L3_RECOVERY_MODE == 2 ? mlmq.manage_warp_num() : 0, &bs_epoch
#endif
                                                   ))
                            c_ok = false;
                        __syncwarp();
#if (MANAGE_PROFILE == true)
                        if (!lane_id) p_confirm += (unsigned)clock() - t_c2;
#endif
                        unsigned ck_mask = __ballot_sync(FULL_MASK, c_ok);
                        bool all_c_ok = (ck_mask == FULL_MASK);
                        __threadfence_block();
                        term_ok = all_c_ok && (confirm_inj_pending_now == 0);
                    }
                    if (term_ok)
                    {
                        if (!lane_id)
                        {
#if (HANG_DIAG == true)
                            printf("L3_TERM_LEGACY_EXIT g%d req=%d state=%d peerstate=%d\n",
                                   v_begin,
                                   (l3_term_req != NULL) ? l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req) : -1,
                                   (l3_term_state != NULL) ? l3_atomic_load_acquire<cuda::thread_scope_system>(l3_term_state) : -1,
                                   (peer_l3_term_state != NULL)
                                       ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_l3_term_state) : -1);
#endif
#if (SEED_BARRIER == true && ASYNC_FB == true)
                            // 终检: peer 已 idle 后，双方有效改进计数仍等于 quiet 基准
                            //   （确认无"握手后刚落位"的真实跨卡改进在途）才终止。
                            bool eff_final = true;
                            if (sb_active)
                                eff_final = (atomicAdd(remote_eff, 0ull) == s_eff_base)
                                         && (atomicAdd(peer_remote_eff, 0ull) == s_peer_eff_base);
                            if (eff_final)
#endif
                            {
                                l3_atomic_store_release<cuda::thread_scope_block>(&manager_end, 1);
                                l3_atomic_store_release<cuda::thread_scope_device>(global_exit, 1);
                                atomicExch(&g_t_term, TIMELINE_CLOCK());
                            }
                        }
                    }
                }
            }
            else
            {
                // 非空闲状态：复位本卡空闲标志（可能之前已置位）
#if (TIMELINE64 == true)
                if (!all_mark_empty && remote_mark != NULL)
                {
                    unsigned long long mark_fail_sample = 0;
                    if (!lane_id)
                    {
                        mark_fail_sample = atomicAdd(&g_term_mark_sampled, 1ull);
                    }
                    mark_fail_sample = __shfl_sync(FULL_MASK, mark_fail_sample, 0);
                    if ((mark_fail_sample & 127ull) == 0)
                    {
                        bool actual_mark_empty = true;
                        int actual_mark_words = (peer_v_local + 31) / 32;
                        for (int w = lane_id; w < actual_mark_words; w += WARP_SIZE)
                            if (l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w]) != 0)
                                actual_mark_empty = false;
                        __syncwarp();
                        unsigned actual_mark_mask = __ballot_sync(FULL_MASK, actual_mark_empty);
                        if (!lane_id)
                        {
                            if (actual_mark_mask == FULL_MASK)
                                atomicAdd(&g_term_mark_stale, 1ull);
                            else
                                atomicAdd(&g_term_mark_actual, 1ull);
                        }
                    }
                }
#endif
                if (!lane_id)
                {
#if (TIMELINE64 == true)
                    unsigned term_fail_reason = 12;
                    if (!all_dirty_empty) term_fail_reason = 6;
                    else if (!all_backstop_ok) term_fail_reason = 5;
                    else if (!all_mark_empty) term_fail_reason = 7;
                    else if (!all_ghost_mark_empty) term_fail_reason = 8;
                    else if (!l3_ok) term_fail_reason = 9;
                    else if (!bulk_inbox_empty || !bulk_outbox_empty) term_fail_reason = 10;
                    else if (*(volatile int *)&inj_pending != 0) term_fail_reason = 11;
                    timeline_idle_write_reason(local_idle, 0, term_fail_reason);
#else
                    timeline_idle_write_reason(local_idle, 0, 5);
#endif
                }
            }
            __threadfence();
#if (MANAGE_PROFILE == true)
            if (!lane_id)
            {
                unsigned p_t3 = clock();
                p_inject += p_t1 - p_t0;
                p_backstop += p_t2 - p_t1;
                p_term += p_t3 - p_t2;
                p_total += p_t3 - p_t0;
                p_iter++;
            }
#endif
#if (HANG_DIAG == true)
            // 心跳（轻量：不扫位图，防寄存器膨胀挤爆 launch）
            if (n_gpu > 1 && !lane_id && (diag_iter % HANG_DIAG_K == 0))
            {
                int hd_ph = (phase != NULL) ? *(volatile int *)phase : -1;
                int hd_sr = (seed_ready != NULL) ? *(volatile int *)seed_ready : -1;
                int hd_sid = (seed_inject_done != NULL) ? *(volatile int *)seed_inject_done : -1;
                int hd_dirty = 0, hd_mark = 0;
                int hd_term_req = (l3_term_req != NULL)
                                ? l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req) : -1;
                int hd_term_state = (l3_term_state != NULL)
                                  ? l3_atomic_load_acquire<cuda::thread_scope_system>(l3_term_state) : -1;
                int hd_peer_term_state = (peer_l3_term_state != NULL)
                                       ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_l3_term_state) : -1;
                int hd_term_ack0 = (l3_term_ack_slots != NULL)
                                 ? l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_ack_slots) : -1;
                int hd_inject_ack0 = (INJECT_WARP_NUM > 0)
                                   ? l3_atomic_load_acquire<cuda::thread_scope_block>(l3_term_inject_ack) : -1;
                int hd_rx_slot = bulk_inbox_slot(bulk_rx_epoch + 1);
                int hd_tx = l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published);
                int hd_tx_slot = bulk_inbox_slot(hd_tx > 0 ? hd_tx : 1);
                int hd_tx_next = hd_tx + 1;
                int hd_tx_next_slot = bulk_inbox_slot(hd_tx_next);
                int hd_in_state = (bulk_inbox_state != NULL)
                                ? l3_atomic_load_acquire<cuda::thread_scope_system>(bulk_inbox_state + hd_rx_slot) : -1;
                int hd_in_gen = (bulk_inbox_generation != NULL)
                              ? l3_atomic_load_relaxed<cuda::thread_scope_system>(bulk_inbox_generation + hd_rx_slot) : -1;
                int hd_in_epoch = (bulk_inbox_epoch != NULL)
                                ? l3_atomic_load_relaxed<cuda::thread_scope_system>(bulk_inbox_epoch + hd_rx_slot) : -1;
                int hd_in_ack = (bulk_inbox_ack != NULL)
                              ? l3_atomic_load_acquire<cuda::thread_scope_system>(bulk_inbox_ack + hd_rx_slot) : -1;
                int hd_out_state = (peer_bulk_inbox_state != NULL)
                                 ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_bulk_inbox_state + hd_tx_slot) : -1;
                int hd_out_gen = (peer_bulk_inbox_generation != NULL)
                               ? l3_atomic_load_relaxed<cuda::thread_scope_system>(peer_bulk_inbox_generation + hd_tx_slot) : -1;
                int hd_out_ack = (peer_bulk_inbox_ack != NULL)
                               ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_bulk_inbox_ack + hd_tx_slot) : -1;
                int hd_next_state = (peer_bulk_inbox_state != NULL)
                                  ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_bulk_inbox_state + hd_tx_next_slot) : -1;
                int hd_next_gen = (peer_bulk_inbox_generation != NULL)
                                ? l3_atomic_load_relaxed<cuda::thread_scope_system>(peer_bulk_inbox_generation + hd_tx_next_slot) : -1;
                int hd_next_ack = (peer_bulk_inbox_ack != NULL)
                                ? l3_atomic_load_acquire<cuda::thread_scope_system>(peer_bulk_inbox_ack + hd_tx_next_slot) : -1;
                for (int h = lane_id; h < hwords; h += 32)
                    if (l3_atomic_load_relaxed<cuda::thread_scope_system>(&dirty_hint[h]) != 0) { hd_dirty = 1; break; }
                if (remote_mark != NULL && mark_hint != NULL)
                {
                    int mhw = ((peer_v_local + 31) / 32 + 31) / 32;
                    for (int h = lane_id; h < mhw; h += 32)
                        if (l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint[h]) != 0) { hd_mark = 1; break; }
                }
                printf("g%d w0 it=%u q=%d l3=%d idle=%d ph=%d sr=%d sid=%d dirty=%d mark=%d inj=%d rx=%d/%d/%d/%d out=%d s=%d/%d ack=%d next=%d/%d/%d term=%d/%d peerterm=%d ack0=%d iack0=%d\n",
                       v_begin, diag_iter, (int)mlmq.get_global_queue_size(), *(volatile int *)&l3_busy,
                       l3_atomic_load_acquire<cuda::thread_scope_system>(local_idle), hd_ph, hd_sr, hd_sid, hd_dirty, hd_mark,
                       *(volatile int *)&inj_pending, bulk_rx_epoch, hd_in_state,
                       hd_in_gen, hd_in_epoch, hd_tx, hd_out_state, hd_out_gen, hd_out_ack,
                       hd_next_state, hd_next_gen, hd_next_ack, hd_term_req,
                       hd_term_state, hd_peer_term_state, hd_term_ack0,
                       hd_inject_ack0);
            }
#endif
        }

        __syncwarp();
#if (MANAGE_PROFILE == true)
        if (!lane_id && mgmt_profile != NULL)
        {
            mgmt_profile[0] = p_inject;
            mgmt_profile[1] = p_backstop;
            mgmt_profile[2] = p_term;
            mgmt_profile[3] = p_total;
            mgmt_profile[4] = p_iter;
            mgmt_profile[9] = p_mark;
            mgmt_profile[10] = p_confirm;
            mgmt_profile[11] = p_pre_cnt;
            mgmt_profile[12] = p_confirm_cnt;
        }
#endif
#if (L3_WAIT_DIAG == true)
        manager_wait.save(256);
#endif
#if (L3_PROGRESS_DIAG == true)
        if (!lane_id) g_l3_progress.end = clock64();
#if (L3_RECOVERY_MODE > 0)
        if (!lane_id) {
            g_l3_progress.recovery_epochs = bs_epoch;
            g_l3_progress.recovery_helpers = INJECT_WARP_NUM
                + (L3_RECOVERY_MODE == 2 ? mlmq.manage_warp_num() : 0);
            g_l3_progress.recovery_done = bs_done;
            assert(bs_req == 0);
        }
#endif
#endif

#if (WORK_COUNT == true)
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            total_work += __shfl_down_sync(0xffffffff, total_work, offset);
            total_comp += __shfl_down_sync(0xffffffff, total_comp, offset);
        }
        if (!lane_id)
        {
            atomicAdd(global_work_count, total_work);
            atomicAdd(global_comp_count, total_comp);
        }
#endif
    }
    // l2 manage warp
    else if (local_wid <= mlmq.manage_warp_num())
    {
        int vec_id = local_wid - 1;
#if (L3_RECOVERY_MODE == 2)
        l3_recovery_cursor recovery_cursor;
#endif
#if (GLOBAL_ROUND_PARALLEL_APPLY == true)
        int apply_seen = 0;
#endif
#if (GLOBAL_ROUND_MULTI_PACK == true)
        int pack_seen = 0;
#endif
        while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
        {
#if (L3_RECOVERY_MODE == 2)
            if (l3_recovery_service(recovery_cursor, 1 + vec_id,
                    1 + mlmq.manage_warp_num() + INJECT_WARP_NUM, v_local, lane_id,
                    node_data, last_processed, dirty_bitmap, dirty_hint,
                    &bs_req, &bs_done, &bs_found)) continue;
#endif
#if (GLOBAL_ROUND_MULTI_PACK == true)
            int pack_req = l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_pack_req);
            if (pack_req > pack_seen)
            {
                bool packed = global_round_multi_pack_worker(
                    pack_req, vec_id, mlmq.manage_warp_num() + 1, lane_id,
                    peer_v_begin, peer_v_local, remote_cand, remote_mark,
                    mark_hint, mark_hint2, peer_node_data, bulk_send_list,
                    peer_bulk_inbox, peer_bulk_inbox_count,
                    peer_bulk_inbox_epoch, peer_bulk_inbox_ack,
                    peer_bulk_inbox_state, peer_bulk_inbox_generation,
                    &bulk_send_count, &bulk_pack_worker_done,
                    &bulk_pack_copy_start, &bulk_pack_copy_done,
                    &bulk_publish_done);
                if (packed)
                {
                    pack_seen = pack_req;
                    if (vec_id == 0 && !lane_id)
                        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published, pack_req);
                }
                continue;
            }
#endif
#if (GLOBAL_ROUND_PARALLEL_APPLY == true)
            int apply_req = atomicAdd((int *)&global_apply_req, 0);
            if (apply_req > apply_seen)
            {
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
                global_round_apply_dense_warp(
                    apply_req, vec_id, mlmq.manage_warp_num(),
                    bulk_dense_inbox, bulk_inbox_count, node_data, v_begin, v_local,
                    bulk_frontier, &bulk_frontier_append_base,
                    &bulk_frontier_append_count, &global_apply_done, lane_id);
#else
                global_round_apply_inbox_warp(
                    apply_req, vec_id, mlmq.manage_warp_num(),
                    bulk_inbox, bulk_inbox_count, node_data, peer_cache_feedback,
                    v_begin, v_local,
                    bulk_frontier, &bulk_frontier_append_base,
                    &bulk_frontier_append_count, &global_apply_done, lane_id);
#endif
                apply_seen = apply_req;
                continue;
            }
            if (apply_req != 0)
            {
                __threadfence();
                continue;
            }
#endif
            mlmq.l2_manager(vec_id, lane_id);
            __threadfence();
        }
    }
    // ===== v3 L3 flush warp（本地加速 + L3 批量传递层，design_v3 §5.3）=====
    // 消费本卡 remote_mark/remote_cand：mark 驱动稀疏扫描 → atomicExch 取出候选
    // → 批量 fire-and-forget 跨卡 atomicMin → 一次 fence → 批量置 peer dirty
    else if (local_wid == mlmq.manage_warp_num() + 1)
    {
#if (GLOBAL_ROUND_ASYNC == true)
        // GLOBAL_ROUND_ASYNC：L3 不再读取一个会被 work 同时清空的单 bank。
        // 每次发现 active bank 有候选就切换 bank；旧 bank 进入 FROZEN 后由
        // async_candidate_freeze 等待所有 work lease 结束，再复用已有的
        // bulk_pack_publish_warp 完成 sparse inbox 发布。active bank 上的 work
        // 在整个打包/等待 peer ACK 期间继续运行。
        if (n_gpu > 1)
        {
            int async_tx_epoch = 0;
            // 发布槽位暂时不可复用时，必须保留 frozen bank 和 epoch。
            // 不能像普通 BULK 批次那样直接丢弃：接收侧严格按
            // rx_epoch+1 领取，跳过一个 epoch 会让后续消息永远不可见。
            int async_frozen_bank = -1;
            int async_pending_epoch = 0;
            while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
            {
                if (async_frozen_bank < 0)
                {
                    int active_bank = 0;
                    int pending_now = 0;
                    if (!lane_id)
                    {
                        active_bank = atomicAdd(cand_ctl.active_bank, 0) & 1;
                        int *pending_ptr = (active_bank == 0)
                            ? cand_bank0.pending : cand_bank1.pending;
                        pending_now = (pending_ptr != NULL)
                            ? atomicAdd(pending_ptr, 0) : 0;
                    }
                    active_bank = __shfl_sync(FULL_MASK, active_bank, 0);
                    pending_now = __shfl_sync(FULL_MASK, pending_now, 0);
                    if (pending_now <= 0)
                    {
                        __threadfence();
                        continue;
                    }

                    if (!lane_id)
                        atomicExch(&l3_busy, 1);
                    __syncwarp();

                    int old_bank = -1;
                    int new_bank = -1;
                    int frozen = 0;
                    if (!lane_id)
                        frozen = async_candidate_freeze(cand_ctl, old_bank, new_bank) ? 1 : 0;
                    frozen = __shfl_sync(FULL_MASK, frozen, 0);
                    old_bank = __shfl_sync(FULL_MASK, old_bank, 0);
                    new_bank = __shfl_sync(FULL_MASK, new_bank, 0);
                    if (!frozen)
                    {
                        if (!lane_id)
                            atomicExch(&l3_busy, 0);
                        __syncwarp();
                        continue;
                    }

                    async_frozen_bank = old_bank;
                    async_pending_epoch = async_tx_epoch + 1;
                    if (!lane_id)
                        l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_publish_done, 0);
                    __threadfence_system();
                    __syncwarp();
                }

                async_candidate_bank frozen_bank =
                    (async_frozen_bank == 0) ? cand_bank0 : cand_bank1;
                bool published = bulk_pack_publish_warp(
                    async_pending_epoch, lane_id, peer_v_begin, peer_v_local,
                    frozen_bank.cand, frozen_bank.mark, frozen_bank.hint,
                    frozen_bank.hint2, peer_cache, peer_node_data,
                    peer_dirty_bitmap, peer_dirty_hint, bulk_send_list,
                    peer_bulk_inbox, peer_bulk_inbox_count, peer_bulk_inbox_epoch,
                    peer_bulk_inbox_ack, peer_bulk_inbox_state,
                    peer_bulk_inbox_generation, &bulk_send_count,
                    (volatile int *)&bulk_publish_done);

                if (!published)
                {
                    // bulk_inbox_try_acquire_write 是非阻塞的；槽位被上一条
                    // 消息占用时，frozen bank 尚未被消费，必须原样保留并在
                    // 下一轮重试同一个 epoch。active bank 继续承接本地 work。
                    __threadfence();
                    continue;
                }

                if (!lane_id)
                {
                    async_tx_epoch = async_pending_epoch;
                    l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published, async_tx_epoch);
                    timeline_idle_write_reason(local_idle, 0, 30);
                    // bulk_pack_publish_warp 已在 FROZEN bank 上完成 mark/cand
                    // 清空；发布成功后才允许复用该 bank。
                    __threadfence_system();
                    if (frozen_bank.pending != NULL)
                        atomicExch(frozen_bank.pending, 0);
                    atomicExch(cand_ctl.state + async_frozen_bank,
                               ASYNC_CAND_BANK_FREE);
                    atomicExch(&l3_busy, 0);
                    async_frozen_bank = -1;
                    async_pending_epoch = 0;
                }
                __threadfence_system();
                __syncwarp();
                // 上面的状态更新只由 lane0 执行；把本地控制状态广播给
                // 整个 warp，避免非 lane0 在下一轮重复发布已完成的 bank。
                async_frozen_bank = __shfl_sync(
                    FULL_MASK, async_frozen_bank, 0);
                async_pending_epoch = __shfl_sync(
                    FULL_MASK, async_pending_epoch, 0);
            }
        }
        else
#elif (GLOBAL_ROUND_MULTI_PACK == true)
        // GLOBAL_ROUND_MULTI_PACK：L2 manager warps 与 L3 warp 共同作为
        // pack workers；manager warp 0 在 helper 内完成唯一 publish。
        if (n_gpu > 1)
        {
            int handled_round = 0;
            while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
            {
                int req = l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_pack_req);
                if (req > handled_round)
                {
                    global_round_multi_pack_worker(
                        req, mlmq.manage_warp_num(), mlmq.manage_warp_num() + 1,
                        lane_id, peer_v_begin, peer_v_local,
                        remote_cand, remote_mark, mark_hint, mark_hint2,
                        peer_node_data, bulk_send_list, peer_bulk_inbox,
                        peer_bulk_inbox_count, peer_bulk_inbox_epoch,
                        peer_bulk_inbox_ack, peer_bulk_inbox_state,
                        peer_bulk_inbox_generation, &bulk_send_count,
                        &bulk_pack_worker_done, &bulk_pack_copy_start,
                        &bulk_pack_copy_done, &bulk_publish_done);
                    handled_round = req;
                }
                else
                {
                    __threadfence();
                }
            }
        }
        else
#elif (BULK_ROUND == true && GLOBAL_ROUND == true)
        // GLOBAL_ROUND：L3 warp 不再按 L3_BATCH 持续发送；只在 manager 已经
        // 冻结所有 work/injection 后，完整打包当前 round 的候选列表。
        if (n_gpu > 1)
        {
            int handled_round = 0;
            while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
            {
                int req = l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_pack_req);
                if (req > handled_round)
                {
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true)
                    bool published = global_round_publish_dense(
                        req, lane_id, peer_v_local,
                        remote_cand, remote_mark, mark_hint, mark_hint2,
                        peer_bulk_dense_inbox, peer_cache,
                        peer_bulk_inbox_count, peer_bulk_inbox_epoch,
                        peer_bulk_inbox_ack, peer_bulk_inbox_state,
                        peer_bulk_inbox_generation, &bulk_send_count,
                        &bulk_publish_done);
#else
                    bool published = bulk_pack_publish_warp(
                        req, lane_id, peer_v_begin, peer_v_local,
                        remote_cand, remote_mark, mark_hint, mark_hint2,
                        peer_cache, peer_node_data,
                        peer_dirty_bitmap, peer_dirty_hint,
                        bulk_send_list, peer_bulk_inbox, peer_bulk_inbox_count,
                        peer_bulk_inbox_epoch, peer_bulk_inbox_ack,
                        peer_bulk_inbox_state, peer_bulk_inbox_generation,
                        &bulk_send_count,
                        &bulk_publish_done);
#endif
                    if (published)
                    {
                        handled_round = req;
                        if (!lane_id)
                            l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published, req);
#if (GLOBAL_ROUND_DIAG == true)
                        if (!lane_id)
                            printf("GLOBAL_ROUND_PUBLISH g%d round=%d count=%d\\n",
                                   v_begin, req, atomicAdd(&bulk_send_count, 0));
#endif
                    }
                }
                else
                {
                    __threadfence();
                }
            }
        }
        else
#elif (BULK_ROUND == true && BULK_EPOCH == true)
        if (n_gpu > 1)
        {
            int handled_epoch = 0;
            while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
            {
                int req = l3_atomic_load_acquire<cuda::thread_scope_block>((int *)&bulk_pack_req);
                if (req > handled_epoch)
                {
                    bool published = bulk_pack_publish_warp(
                        req, lane_id, peer_v_begin, peer_v_local,
                        remote_cand, remote_mark, mark_hint, mark_hint2,
                        peer_cache, peer_node_data,
                        peer_dirty_bitmap, peer_dirty_hint,
                        bulk_send_list, peer_bulk_inbox, peer_bulk_inbox_count,
                        peer_bulk_inbox_epoch, peer_bulk_inbox_ack,
                        peer_bulk_inbox_state, peer_bulk_inbox_generation,
                        &bulk_send_count,
                        &bulk_publish_done);
                    if (published)
                    {
                        handled_epoch = req;
                        if (!lane_id)
                            l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published, req);
                    }
                }
                else
                {
                    __threadfence();
                }
            }
        }
        else
#endif
        {
            const int L3_BATCH = BULK_L3_BATCH;  // 每 lane 每轮收集上限
        __shared__ int l3_lidx[WARP_SIZE * L3_BATCH];
        __shared__ VALUE_TYPE l3_nd[WARP_SIZE * L3_BATCH];
        __shared__ int l3_cnt;
        // B 开销优化: 本轮真实改进 peer 的计数（自适应扫描间隔用）
        __shared__ int l3_eff;

        if (remote_cand == NULL)
        {
            // n=1 退化：无发送侧候选，空转等待终止（不阻塞 warp0 终止判定）
            while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0) { __threadfence(); }
        }
        else
        {
            const int l3_cap = WARP_SIZE * L3_BATCH;
            const int candidate_vertices = l3_candidate_size(peer_v_local);
            int mark_words = (candidate_vertices + 31) / 32;
            // E1b: mark 扫描自适应降频。扫完一轮发现无候选（cnt==0）→ 进入 quiet 态，
            //   quiet 态每 l3_k 轮才扫一次 mark（空转轮不扫），有候选立即退出 quiet。
            //   B 开销优化: l3_k 自适应——连续 8 轮候选全无效（P2 成立图接收卡 flush 全无效）
            //   提高到 256；出现真实改进立即恢复 32；quiet 空转至少 128。
            //   正确性：quiet 空转不影响 work 置 mark；延迟最多 l3_k 轮被处理，
            //   终止判定（warp0 all_mark_empty + l3_busy==0）保证不提前终止。
#if (ASYNC_FB == true)
            int l3_k = 32;
            int l3_invalid_streak = 0;
#else
#if (BULK_ROUND == true && SEED_BARRIER == false)
            // 通用 BULK 由 mark_signal 唤醒 quiet 状态；L3_K 只负责无事件时降频。
            const int L3_K = 32;
#else
            const int L3_K = 32;   // SEED_BARRIER: 固定扫描间隔（原始行为）
#endif
#endif
            int l3_round = 0;
            bool l3_quiet = false;
#if (L3_QUIET_DRAIN == true)
            bool quiet_drained = false; // lane0 only; one eager scan per quiet episode
#endif
#if (L3_DYNAMIC == true)
            l3_dynamic_state l3_dynamic = {
                L3_DYNAMIC_EAGER, 0, 0, 0, 0
            };
#endif
#if (BULK_ROUND == true && SEED_BARRIER == false)
            // g_bulk_mark_signal 由 work warp 异步递增；只能由 lane0 读取并广播。
            // 若各 lane 独立读取，signal 恰在读取期间变化时会出现部分 lane
            // continue、部分 lane 进入收集，最终在后续 __syncwarp() 处死锁。
            unsigned long long l3_signal_seen = 0;
            if (!lane_id)
                l3_signal_seen = atomicAdd(&g_bulk_mark_signal, 0ull);
            l3_signal_seen = __shfl_sync(FULL_MASK, l3_signal_seen, 0);
#endif
            // 方案2(bug修复): quiet 态周期性全扫 remote_mark 兜底。hint 保守清零有 ABA 竞态
            //   （work 置 mark_hint 时 bit 已 1 → or 幂等 → w17 CAS 用旧值误清 → mark_hint 假阴），
            //   残留 remote_mark 永不消费 → w0 all_mark_empty 永不成立 → rgg 卡死（SESSION 20）。
            //   故 quiet 态每 L3_K*L3_FULL 轮走一次单层全扫 remote_mark（不依赖 hint），清残留。
            // L3_FULL: quiet 态全扫兜底周期（固定 quiet 轮数，与 l3_k 解耦——l3_k 增大
            //   不得稀释 SESSION 20 的 hint ABA 兜底频率）
            const int L3_FULL = 256;
            int l3_full_round = 0;
#if (L3_WINDOW_MODE != 0)
            l3_window_state window;
            if (!lane_id) window.signal = l3_signal_seen;
#if (L3_EVENT_GATE == true)
            l3_event_gate event_gate;
            if (!lane_id) event_gate.last_full = clock64();
#endif
#endif
#if (MANAGE_PROFILE == true)
            // V0.5 profile: L3 mark 扫描 / 跨卡 flush 差分
            unsigned long long p_scan = 0, p_flush = 0, p_total = 0;
            unsigned p_iter = 0;
#endif
#if (BULK_ROUND == true)
            int bulk_tx_epoch = 0;
#if (L3_RETAIN_TX == true)
            // Warp-uniform pending journal length. l3_lidx/l3_nd stay immutable
            // until published; work continues producing the authoritative store.
            int retained_count = 0;
#endif
#if (L3_FAULT_INJECT_PUBLISH_RETRY == true)
            bool l3_fault_injected = false;
#endif
#endif
#if (L3_WAIT_DIAG == true)
            l3_wait_clock tx_wait(!lane_id);
#endif
            while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
            {
#if (L3_WAIT_DIAG == true)
                tx_wait.tick(0);
#endif
#if (L3_RETAIN_TX == true)
                if (retained_count > 0) {
#if (L3_WAIT_DIAG == true)
                    tx_wait.tick(1);
#endif
#if (L3_RX_FEEDBACK_MODE > 0)
                    l3_feedback_record observed = {};
#endif
                    bool sent = bulk_publish_l3_batch(
                        bulk_tx_epoch + 1, lane_id, peer_v_begin, peer_v_local,
                        retained_count, l3_lidx, l3_nd,
                        peer_bulk_inbox, peer_bulk_inbox_count,
                        peer_bulk_inbox_epoch, peer_bulk_inbox_ack,
                        peer_bulk_inbox_state, peer_bulk_inbox_generation
#if (L3_RX_FEEDBACK_MODE > 0)
                        , l3_channel.tx_feedback, &observed
#endif
                        );
                    if (sent) {
                        ++bulk_tx_epoch;
#if (L3_RX_FEEDBACK_MODE > 0)
                        if (!lane_id)
                            l3_apply_rx_feedback(window, observed, bulk_tx_epoch, v_begin, peer_v_begin);
#endif
                        retained_count = 0;
                        if (!lane_id) {
                            l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published, bulk_tx_epoch);
                            l3_busy = 0;
                        }
                        __syncwarp();
                        __threadfence_block();
                    }
                    // Nonblocking retry: the independent RX manager can run.
                    // On failure busy remains set and the journal is untouched.
                    continue;
                }
#endif
#if (BULK_ROUND == true && L3_TERM_HANDSHAKE == true && \
     BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && \
     SEED_BARRIER == false)
                // Safe point: any retained journal is published above; no new
                // mark can be claimed while this exact termination token lives.
                int tx_term_req = 0;
                if (!lane_id && l3_term_req) tx_term_req = l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req);
                tx_term_req = __shfl_sync(FULL_MASK, tx_term_req, 0);
                if (tx_term_req != 0) {
#if (L3_WAIT_DIAG == true)
                    tx_wait.tick(2);
#endif
                    __threadfence_block();
                    if (!lane_id) l3_atomic_store_release<cuda::thread_scope_block>(&l3_term_tx_ack, tx_term_req);
                    __syncwarp();
                    continue;
                }
#endif
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                {
                    atomicAdd(&g_l3_live_state.l3_iter, 1ull);
#if (BULK_ROUND == true)
                    l3_live_l3_state(1, bulk_tx_epoch, 0);
#else
                    l3_live_l3_state(1, 0, 0);
#endif
                }
#endif
#if (SEED_BARRIER == true && SEED_PHASE2_PARALLEL == true)
                // phase2 源卡由 L3 warp 与注入 warp 共同发布；旧 L3 路径在 phase2
                // 会空转，必须在其 gate 之前进入固定分片发布路径。
                if (n_gpu > 1 && src >= v_begin && src < v_end
                    && phase != NULL && *(volatile int *)phase == 2)
                {
                    seed_phase2_publish_warp(
                        0, 1 + INJECT_WARP_NUM, lane_id, m,
                        phase, &seed_pub_busy,
                        peer_v_begin, peer_v_local,
                        remote_cand, remote_mark, mark_hint, mark_hint2,
                        peer_cache, peer_node_data,
                        seed_list, seed_list_cnt);
                    continue;
                }
#endif
#if (SEED_BARRIER == true)
#if (ASYNC_FB == true)
                // 方案 B（异步反馈环）: 源卡 phase2（灌值）+ phase3（异步反馈）执行 flush；
                //   接收卡 seed_inject_done 后执行（此前无候选，禁扫免抢 L2）。
                if (n_gpu > 1)
                {
                    bool l3_gate;
                    if (src >= v_begin && src < v_end)
                    {
                        int ph_g = *(volatile int *)phase;
                        l3_gate = (ph_g != 2 && ph_g != 3);
                    }
                    else
                    {
                        l3_gate = (seed_inject_done != NULL && *(volatile int *)seed_inject_done == 0);
                    }
                    if (l3_gate)
                    {
                        __threadfence();
                        continue;
                    }
                }
#else
                // 解法0: 源卡仅灌值期（phase==2）执行 flush；计算期/seed_ready 后空转等 phase。
                //   接收卡也门控空转——P2 单调性（本实验图）下其跨卡候选对源卡全为无效
                //   flush（remote_effective=0），禁扫免抢 L2（gpu1 w17 空扫实测 ~72ms 拖慢
                //   work）。接收卡终止不要求 mark_empty（mark 是无用残留）。
                if (n_gpu > 1 && ((src >= v_begin && src < v_end)
                                      ? ((*(volatile int *)phase) != 2)
                                      : true))
                {
                    __threadfence();
                    continue;
                }
#endif
#endif
                // E1b: quiet 态降频空转（不扫 mark，仅检查 manager_end）
#if (L3_WINDOW_MODE != 0)
                int window_scan = 0;
                int feedback_scan = 0;
#if (L3_EVENT_GATE == true)
                int event_full = 0;
                unsigned long long event_snapshot = 0; // lane0, before extraction
#endif
#if (L3_FEEDBACK_DRAIN == true)
                if (!lane_id)
                    feedback_scan = l3_atomic_exchange_acq_rel<cuda::thread_scope_block>(
                        &l3_drain_requested, 0);
                feedback_scan = __shfl_sync(FULL_MASK, feedback_scan, 0);
#endif
                if (!lane_id) {
#if (L3_LOCAL_SETTLE == true)
                    const bool new_event = atomicAdd(&g_bulk_mark_signal, 0ull) != window.signal;
                    // Local L0/L1 work remains included in L2 outstanding until
                    // completion. This hint never replaces mark/ACK/term truth.
                    // No peer-idle shortcut: avoid eagerly exporting every
                    // transient boundary improvement while this owner works.
                    const bool local_empty = mlmq.get_global_queue_size() == 0;
                    unsigned long long limit = local_empty ? L3_WINDOW_MIN_CYCLES
                        : new_event ? L3_SETTLE_CYCLES : L3_WINDOW_MAX_CYCLES;
                    window_scan = window.allow(clock64(), false, false, limit);
#else
                    const auto decision_now = clock64();
                    const auto decision = window.decision(decision_now, L3_WINDOW_MODE == 2,
                                                          L3_WINDOW_MAX_CYCLES);
#if (L3_EVENT_GATE == true)
                    if (decision != l3_window_state::WAIT || feedback_scan) {
                        event_snapshot = atomicAdd(&g_bulk_mark_signal, 0ull);
                        event_full = event_gate.full_due(decision_now, L3_EVENT_FULL_CYCLES);
                        const bool has_hint = event_gate.work_hint(event_snapshot);
                        window_scan = event_full || (has_hint &&
                            (decision == l3_window_state::SCAN ||
                             mlmq.get_global_queue_size() == 0 ||
                             (peer_local_idle && l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) != 0)));
                        if (!window_scan && !feedback_scan && decision == l3_window_state::SCAN) {
                            window.deferred(decision_now);
#if (L3_DIAGNOSTICS == true)
                            ++g_l3_diagnostics.event_skips;
#endif
                        }
                    }
#else
                    window_scan = decision == l3_window_state::SCAN;
                    if (decision == l3_window_state::PROBE) {
                        const bool new_event = atomicAdd(&g_bulk_mark_signal, 0ull) != window.signal;
                        window_scan = new_event && (mlmq.get_global_queue_size() == 0
                            || (peer_local_idle && l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) != 0));
                    }
#endif
#endif
                }
                window_scan = __shfl_sync(FULL_MASK, window_scan, 0);
#if (L3_EVENT_GATE == true)
                event_full = __shfl_sync(FULL_MASK, event_full, 0);
#endif
                if (!window_scan && !feedback_scan) { __threadfence(); continue; }
#if (L3_DIAGNOSTICS == true)
                if (!lane_id) g_l3_diagnostics.feedback_drains += feedback_scan != 0;
#endif
                l3_full_round = (l3_full_round + 1) % L3_FULL;
#else
                if (l3_quiet)
                {
#if (BULK_ROUND == true && SEED_BARRIER == false)
                    unsigned long long signal_now = 0;
                    if (!lane_id)
                        signal_now = atomicAdd(&g_bulk_mark_signal, 0ull);
                    signal_now = __shfl_sync(FULL_MASK, signal_now, 0);
                    if (signal_now != l3_signal_seen)
                    {
                        // 新 mark 已产生：立即唤醒，不等待 L3_K 空轮。
                        unsigned long long signal_before = l3_signal_seen;
                        unsigned long long signal_delta = signal_now - signal_before;
                        l3_signal_seen = signal_now;
#if (L3_DYNAMIC == true)
                        bool peer_idle_now = false;
                        bool local_idle_now = false;
                        if (!lane_id)
                        {
                            peer_idle_now = (peer_local_idle != NULL)
                                          && (l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) != 0);
                            local_idle_now = (local_idle != NULL)
                                           && (l3_atomic_load_acquire<cuda::thread_scope_system>(local_idle) != 0);
                        }
                        peer_idle_now = __shfl_sync(FULL_MASK, peer_idle_now, 0);
                        local_idle_now = __shfl_sync(FULL_MASK, local_idle_now, 0);
                        bool scan_now = l3_dynamic_should_scan(
                            l3_dynamic, l3_cap, peer_idle_now,
                            local_idle_now, true, signal_delta);
                        l3_quiet = !scan_now;
                        l3_round = 0;
#if (L3_DYNAMIC == true)
                        // signal 只表示“有新事件”，动态策略仍可决定继续
                        // coalesce。若策略拒绝本轮扫描，必须在此退出；否则
                        // 后续代码仍会无条件进入 mark/candidate 收集路径。
                        if (!scan_now)
                        {
#if (L3_EVENT_RING == true)
                            if (!lane_id)
                                l3_event_push(L3_EVENT_SCAN_COALESCED,
                                              (int)signal_delta,
                                              (int)peer_idle_now,
                                              (int)local_idle_now, 1);
#endif
                            __threadfence();
                            continue;
                        }
#endif
#else
                        l3_quiet = false;
                        l3_round = 0;
#endif
                    }
                    else
#endif
                    {
#if (L3_DYNAMIC == true && BULK_ROUND == true && SEED_BARRIER == false)
                    bool peer_idle_now = false;
                    bool local_idle_now = false;
                    if (!lane_id)
                    {
                        peer_idle_now = (peer_local_idle != NULL)
                                      && (l3_atomic_load_acquire<cuda::thread_scope_system>(peer_local_idle) != 0);
                        local_idle_now = (local_idle != NULL)
                                       && (l3_atomic_load_acquire<cuda::thread_scope_system>(local_idle) != 0);
                    }
                    peer_idle_now = __shfl_sync(FULL_MASK, peer_idle_now, 0);
                    local_idle_now = __shfl_sync(FULL_MASK, local_idle_now, 0);
                    bool scan_now = l3_dynamic_should_scan(
                        l3_dynamic, l3_cap, peer_idle_now,
                        local_idle_now, false, 0);
                    if (!scan_now)
                    {
                        // 无新 signal 的 quiet 轮是轮询噪声，不记录；事件环只
                        // 保留新 signal 被策略合并的边界，避免覆盖 pack/rx/term。
                        __threadfence();
                        continue;
                    }
                    l3_round = 0;
                    l3_full_round++;
#else
                    l3_round++;
#if (ASYNC_FB == true)
                    if (l3_round % l3_k != 0)
#else
                    if (l3_round % L3_K != 0)
#endif
                    {
                        __threadfence();
                        continue;
                    }
                    l3_round = 0;
                    l3_full_round++;      // 全扫兜底周期 = 每 L3_FULL 次收集一次（与 l3_k 解耦）
#endif
                    }
                }
#endif
#if (MANAGE_PROFILE == true)
                unsigned p_t0 = clock();
#endif
                // 收集阶段开始：置在途标志（warp0 终止检测读，防"取走 mark 未 flush"误判）
#if (L3_DIAGNOSTICS == true)
                unsigned long long scan_start = clock64();
#endif
                if (!lane_id) { l3_cnt = 0; l3_eff = 0; l3_busy = 1; }
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                {
#if (BULK_ROUND == true)
                    l3_live_l3_state(2, bulk_tx_epoch, 0);
#else
                    l3_live_l3_state(2, 0, 0);
#endif
                }
#endif
                __syncwarp();
                __threadfence_block();
                // E4: 两级扫（外层扫 mark_hint 稀疏，命中块再扫 32 个 mark word）。
                //   mh_words <= 32 时两级扫退化为"每 lane 1 块、仅 mh_words 个 lane 活跃"，
                //   收集并行度坍缩（32→mh_words），小图（barth/del_n12）实测 3ms→24ms 退化。
                //   故 hybrid：mh_words <= 32 走单层 stride 扫（E3 原路径，32 lane 并行），
                //   >32 走两级扫（mark_hint 稀疏跳过大图 7 万 words 全扫）。两路径均保持
                //   不变量（终止直扫 mark + 连续扫描兜底），hint 缺陷仅延迟/假阳。
                int collected = 0;
                int mh_words = (mark_words + 31) / 32;
                // quiet 态周期性全扫兜底（清 hint ABA 假阴残留，见上方 L3_FULL 注释）
                bool full_scan = l3_quiet && (l3_full_round % L3_FULL == 0);
#if (L3_WINDOW_MODE != 0)
                full_scan = (l3_full_round == 0) || feedback_scan;
#if (L3_EVENT_GATE == true)
                full_scan = event_full || feedback_scan;
#endif
#endif
#if (L3_QUIET_DRAIN == true)
                // A scheduling hint only. The original mark CAS/Exch and
                // termination predicates still decide ownership/completion.
                int drain_full = 0;
                if (!lane_id) {
                    if (!l3_quiet) quiet_drained = false;
                    if (l3_quiet && !quiet_drained && mlmq.get_global_queue_size() == 0) {
                        drain_full = 1;
                        quiet_drained = true;
                    }
                }
                drain_full = __shfl_sync(FULL_MASK, drain_full, 0);
                full_scan = full_scan || drain_full;
#endif
#if (L3_COOPERATIVE_COLLECT == true)
                if (!full_scan && mh_words > 32) {
                    int packed = l3_collect_cooperative(remote_cand, remote_mark,
                        mark_hint, mark_hint2, peer_cache, candidate_vertices,
                        l3_cap, l3_lidx, l3_nd, lane_id);
                    if (!lane_id) l3_cnt = packed;
                } else
#endif
                if (full_scan || mh_words <= 32)
                {
                    // 单层扫 mark（E3 原路径）：32 lane stride 并行，无 hint
                    const int scan_words=l3_scan_word_count(mark_words);
                    for (int wi = lane_id; wi < scan_words && collected < L3_BATCH; wi += 32)
                    {
                        const int w=l3_scan_word(wi);
                        int mv = l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w]);
                        if (!mv) continue;
                        if (!l3_device_mark_claim(&remote_mark[w], (unsigned)mv)) continue;
                        unsigned done_bits = 0;
                        for (int b = 0; b < 32 && collected < L3_BATCH; b++)
                        {
                            if (!(mv & (1u << b))) continue;
                            int lidx = w * 32 + b + 1;    // 局部 1-based
                            if (lidx > candidate_vertices) continue;
                            int nd = atomicExch(&remote_cand[lidx], DIST_MAX); // 取出并重置
                            if (nd == DIST_MAX) continue; // 其他 lane 已取
#if (BULK_NO_CACHE == false)
                            atomicMin(&peer_cache[lidx], nd);
#endif
                            int slot = atomicAdd(&l3_cnt, 1);
                            if (slot < l3_cap)
                            {
                                l3_lidx[slot] = lidx;
                                l3_nd[slot] = nd;
                            }
                            else
                            {
                                atomicMin(&remote_cand[lidx], nd);
                                l3_device_mark_publish(&remote_mark[(lidx - 1) >> 5], 1u << ((lidx - 1) & 31));
                            }
                            done_bits |= (1u << b);
                            collected++;
                        }
                        if (collected >= L3_BATCH)
                        {
                            unsigned leftover = mv & ~done_bits;
                            if (leftover)
                                l3_device_mark_publish(&remote_mark[w], leftover);
                        }
                    }
                }
                else
                {
                int mh2_words = (mh_words + 31) / 32;
                if (mh2_words <= 32)
                {
                for (int h = lane_id; h < mh_words && collected < L3_BATCH; h += 32)
                {
                    unsigned hv = l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint[h]);
                    if (!hv) continue;
                    bool block_cleared = true;      // 块内所有 mark word 均已消费（或本就为空）
                    bool cap_hit = false;           // 容量触顶 → 跳过保守清零
                    int w0 = h << 5;                // 块 = [w0, w0+32)
                    for (int w = w0; w < w0 + 32 && !cap_hit; w++)
                    {
                        if (w >= mark_words) break; // 对齐上取整可能越界，clamp
                        int mv = l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w]);
                        if (!mv) continue;
                        if (!l3_device_mark_claim(&remote_mark[w], (unsigned)mv)) continue;
                        block_cleared = false;
                        // 已处理 bit 掩码（用于回填未处理 bit，防 collected 达上限丢失）
                        unsigned done_bits = 0;
                        for (int b = 0; b < 32 && collected < L3_BATCH; b++)
                        {
                            if (!(mv & (1u << b))) continue;
                            int lidx = w * 32 + b + 1;    // 局部 1-based
                            if (lidx > candidate_vertices) continue;
                            int nd = atomicExch(&remote_cand[lidx], DIST_MAX); // 取出并重置
                            if (nd == DIST_MAX) continue; // 其他 lane 已取
                            // 更新本卡 peer 镜像 = 已传候选值（work 再松弛同值不再视为改进，
                            //   防 L3 重置 remote_cand 后 work 无限写候选 -> 无限 flush 乒乓）
#if (BULK_NO_CACHE == false)
                            atomicMin(&peer_cache[lidx], nd);
#endif
                            int slot = atomicAdd(&l3_cnt, 1);
                            if (slot < l3_cap)
                            {
                                l3_lidx[slot] = lidx;
                                l3_nd[slot] = nd;
                            }
                            else
                            {
                                // 缓冲满：写回候选并重置 mark（下轮处理，不丢失）
                                atomicMin(&remote_cand[lidx], nd);
                                l3_device_mark_publish(&remote_mark[(lidx - 1) >> 5], 1u << ((lidx - 1) & 31));
                                // E4: 回填路径1 显式补 mark_hint（§18.3 补充点，消除对 block_cleared 语义依赖）
                                atomicOr(&mark_hint[(lidx - 1) >> 10], 1u << (((lidx - 1) >> 5) & 31));
                                // P1: 回填路径1 显式补 mark_hint2
                                atomicOr(&mark_hint2[(lidx - 1) >> 15], 1u << (((lidx - 1) >> 10) & 31));
                            }
                            done_bits |= (1u << b);       // 已处理（取出或写回）
                            collected++;
                        }
                        // collected 达上限提前退出：word 里未处理的置位 bit 回填 mark（防丢失）
                        if (collected >= L3_BATCH)
                        {
                            unsigned leftover = mv & ~done_bits;
                            if (leftover)
                            {
                                l3_device_mark_publish(&remote_mark[w], leftover);
                                // E4: 回填路径2 显式补 mark_hint
                                atomicOr(&mark_hint[w >> 5], 1u << (w & 31));
                                // P1: 回填路径2 显式补 mark_hint2
                                atomicOr(&mark_hint2[w >> 10], 1u << ((w >> 5) & 31));
                            }
                            cap_hit = true;
                        }
                    }
                    // E4: 保守清零（§18.3 不变量②）：块消费完且未触顶 → 复扫块全 0
                    //   才 CAS(mark_hint[h], hv, 0)（hv=块扫描前快照，并发置位→CAS 失败保留）
                    if (block_cleared && !cap_hit)
                    {
                        bool all_zero = true;
                        for (int w = w0; w < w0 + 32; w++)
                        {
                            if (w >= mark_words) break;
                            if (l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w]) != 0) { all_zero = false; break; }
                        }
                        if (all_zero)
                        {
                            atomicCAS(&mark_hint[h], hv, 0);

                            // P1 修复：两级路径原先只清 mark_hint，未清其对应的
                            // mark_hint2 bit，导致大分区终止检查长期看到 stale summary。
                            // 按 bit 做 CAS，并在 CAS 后复扫 mark_hint[h]；若生产者在
                            // 清理窗口内重新置位，立即恢复 mark_hint2，避免 false-negative。
                            if (mark_hint2 != NULL)
                            {
                                unsigned *h2_ptr = &mark_hint2[h >> 5];
                                unsigned h2_bit = 1u << (h & 31);
                                unsigned h2_old = atomicAdd(h2_ptr, 0u);
                                if (h2_old & h2_bit)
                                {
                                    if (l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint[h]) == 0)
                                    {
                                        unsigned h2_prev = atomicCAS(
                                            h2_ptr, h2_old, h2_old & ~h2_bit);
                                        if (h2_prev == h2_old
                                            && l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint[h]) != 0)
                                            atomicOr(h2_ptr, h2_bit);
                                    }
                                }
                            }
                        }
                    }
                }
                } // if (mh2_words <= 32): 两级扫
                else
                {
                // P1: 三级扫（大图 mh_words 巨大，mark_hint2 稀疏跳过大片空区）
                //   外层扫 mark_hint2（每 32 个 mark_hint word 一个 bit），命中块下钻扫
                //   mark_hint → remote_mark（复用两级逻辑）。mark_hint2 保守清零：
                //   复扫 mark_hint 块全 0 才 CAS 0（假阳保留到下轮，仅多扫，无正确性影响）。
                for (int h2 = lane_id; h2 < mh2_words && collected < L3_BATCH; h2 += 32)
                {
                    unsigned hv2 = l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint2[h2]);
                    if (!hv2) continue;
#if (WORK_COUNT == true)
                    atomicAdd(&g_h2_hit, 1ull);
#endif
                    bool cap_hit2 = false;
                    int h0 = h2 << 5;               // 块 = [h0, h0+32) mark_hint word
                    for (int h = h0; h < h0 + 32 && !cap_hit2 && collected < L3_BATCH; h++)
                    {
                        if (h >= mh_words) break;
                        unsigned hv = l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint[h]);
                        if (!hv) continue;
                        bool block_cleared = true;
                        bool cap_hit = false;
                        int w0 = h << 5;            // 块 = [w0, w0+32) mark word
                        for (int w = w0; w < w0 + 32 && !cap_hit; w++)
                        {
                            if (w >= mark_words) break;
                            int mv = l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w]);
#if (WORK_COUNT == true)
                            atomicAdd(&g_w_scan, 1ull);
#endif
                            if (!mv) continue;
#if (WORK_COUNT == true)
                            atomicAdd(&g_w_hit, 1ull);
#endif
                            if (!l3_device_mark_claim(&remote_mark[w], (unsigned)mv)) continue;
                            block_cleared = false;
                            unsigned done_bits = 0;
                            for (int b = 0; b < 32 && collected < L3_BATCH; b++)
                            {
                                if (!(mv & (1u << b))) continue;
                                int lidx = w * 32 + b + 1;
                                if (lidx > candidate_vertices) continue;
                                int nd = atomicExch(&remote_cand[lidx], DIST_MAX);
                                if (nd == DIST_MAX) continue;
#if (BULK_NO_CACHE == false)
                                atomicMin(&peer_cache[lidx], nd);
#endif
                                int slot = atomicAdd(&l3_cnt, 1);
                                if (slot < l3_cap)
                                {
                                    l3_lidx[slot] = lidx;
                                    l3_nd[slot] = nd;
                                }
                                else
                                {
                                    atomicMin(&remote_cand[lidx], nd);
                                    l3_device_mark_publish(&remote_mark[(lidx - 1) >> 5], 1u << ((lidx - 1) & 31));
                                    atomicOr(&mark_hint[(lidx - 1) >> 10], 1u << (((lidx - 1) >> 5) & 31));
                                    atomicOr(&mark_hint2[(lidx - 1) >> 15], 1u << (((lidx - 1) >> 10) & 31));
                                }
                                done_bits |= (1u << b);
                                collected++;
                            }
                            if (collected >= L3_BATCH)
                            {
                                unsigned leftover = mv & ~done_bits;
                                if (leftover)
                                {
                                    l3_device_mark_publish(&remote_mark[w], leftover);
                                    atomicOr(&mark_hint[w >> 5], 1u << (w & 31));
                                    atomicOr(&mark_hint2[w >> 10], 1u << ((w >> 5) & 31));
                                }
                                cap_hit = true;
                            }
                        }
                        if (block_cleared && !cap_hit)
                        {
                            bool all_zero = true;
                            for (int w = w0; w < w0 + 32; w++)
                            {
                                if (w >= mark_words) break;
                                if (l3_atomic_load_acquire<cuda::thread_scope_device>(&remote_mark[w]) != 0) { all_zero = false; break; }
                            }
                            if (all_zero)
                                atomicCAS(&mark_hint[h], hv, 0);
                        }
                        if (cap_hit) cap_hit2 = true;
                    }
                    // mark_hint2 保守清零：h2 块内所有 mark_hint word 均空才清
                    if (!cap_hit2)
                    {
                        bool all_zero2 = true;
                        for (int h = h0; h < h0 + 32; h++)
                        {
                            if (h >= mh_words) break;
                            if (l3_atomic_load_relaxed<cuda::thread_scope_device>(&mark_hint[h]) != 0) { all_zero2 = false; break; }
                        }
                        if (all_zero2)
                            atomicCAS(&mark_hint2[h2], hv2, 0);
                    }
                }
                } // else: 三级扫
                } // else: mh_words > 32
                __syncwarp();
                int cnt = l3_cnt;
                if (cnt > l3_cap) cnt = l3_cap;
#if (L3_DIAGNOSTICS == true)
                if (!lane_id) {
                    ++g_l3_diagnostics.scans;
                    g_l3_diagnostics.empty += cnt == 0;
                    g_l3_diagnostics.full += full_scan;
#if (L3_EVENT_GATE == true)
                    g_l3_diagnostics.event_full += event_full != 0;
#endif
                    g_l3_diagnostics.extracted += cnt;
                    g_l3_diagnostics.scan_cycles += clock64() - scan_start;
                    if (cnt) {
                        int minimum = DIST_MAX, maximum = 0;
                        for (int j = 0; j < cnt; ++j) {
                            minimum = min(minimum, l3_nd[j]);
                            maximum = max(maximum, l3_nd[j]);
                            if (j && l3_nd[j] < l3_nd[j - 1])
                                ++g_l3_diagnostics.adjacent_descents;
                        }
                        auto batch = g_l3_diagnostics.tx_batches++;
                        if (batch < 32)
                            g_l3_diagnostics.waves[batch] = {cnt, minimum, maximum};
                    }
                }
#endif
#if (L3_WINDOW_MODE != 0)
                if (!lane_id) {
#if (L3_DIAGNOSTICS == true)
                    auto old_budget = window.budget;
                    g_l3_diagnostics.budget_sum += L3_WINDOW_MODE == 2
                        ? window.budget : L3_WINDOW_MAX_CYCLES;
#endif
                    window.scanned(clock64(), atomicAdd(&g_bulk_mark_signal, 0ull),
                                   cnt, L3_WINDOW_MODE == 2, L3_WINDOW_MAX_CYCLES);
#if (L3_EVENT_GATE == true)
                    event_gate.scanned(clock64(), event_snapshot, cnt, full_scan);
#endif
#if (L3_DIAGNOSTICS == true)
                    g_l3_diagnostics.budget_changes += window.budget != old_budget;
#endif
                }
#endif
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                {
#if (BULK_ROUND == true)
                    l3_live_l3_state(3, bulk_tx_epoch, cnt);
#else
                    l3_live_l3_state(3, 0, cnt);
#endif
                }
#endif
#if (MANAGE_PROFILE == true)
                unsigned p_t1 = clock();
#endif

                if (cnt > 0)
                {
#if (L3_BATCH_ORDER == true)
                    l3_order_batch(l3_lidx, l3_nd, cnt, lane_id);
                    __syncwarp();
#endif
#if (L3_EVENT_RING == true)
                    if (!lane_id)
                    {
                        int event_epoch = 0;
#if (BULK_ROUND == true)
                        event_epoch = bulk_tx_epoch + 1;
#endif
                        l3_event_push(L3_EVENT_PACK_BEGIN,
                                      cnt, event_epoch, 0, 0);
                    }
#endif
#if (BULK_ROUND == true && SEED_BARRIER == false)
                    // 通用路径：把本轮聚合结果发布到对端 inbox。接收卡会在本地
                    // atomicMin 后置 dirty，再由现有注入 warp 写入本地 L2。
                    // 只有发布完成后才清 l3_busy，终止检查不会越过在途消息。
                    if (n_gpu > 1 && peer_bulk_inbox != NULL)
                    {
#if (L3_COMPACT_CANDIDATES == true)
                        // All candidate/cache/mark operations above use compact
                        // IDs. The retained batch below always uses wire IDs;
                        // retry at the top of the loop must not remap it again.
                        for(int i=lane_id;i<cnt;i+=WARP_SIZE)
                            l3_lidx[i]=l3_candidate_peer_index(l3_lidx[i]);
                        __syncwarp();
#endif
                        if (!lane_id)
                            l3_atomic_store_release<cuda::thread_scope_system>((int *)local_idle, 0);
                        __syncwarp();
                        int next_tx_epoch = bulk_tx_epoch + 1;
                        bool published = false;
#if (L3_RX_FEEDBACK_MODE > 0)
                        l3_feedback_record observed = {};
#endif
#if (L3_FAULT_INJECT_PUBLISH_RETRY == true)
                        if (!l3_fault_injected)
                        {
                            // 模拟一次瞬时 transport 失败：不消费候选，走与
                            // 槽位暂不可复用相同的 requeue 分支。
                            l3_fault_injected = true;
#if (L3_DIAGNOSTICS == true)
                            if (!lane_id) ++g_l3_diagnostics.injected_publish_retries;
#endif
                        }
                        else
#endif
                        {
                            published = bulk_publish_l3_batch(
                                next_tx_epoch, lane_id, peer_v_begin, peer_v_local,
                                cnt, l3_lidx, l3_nd,
                                peer_bulk_inbox, peer_bulk_inbox_count,
                                peer_bulk_inbox_epoch, peer_bulk_inbox_ack,
                                peer_bulk_inbox_state, peer_bulk_inbox_generation
#if (L3_RX_FEEDBACK_MODE > 0)
                                , l3_channel.tx_feedback, &observed
#endif
                                );
                        }
                        if (published)
                        {
                            bulk_tx_epoch = next_tx_epoch;
#if (L3_RX_FEEDBACK_MODE > 0)
                            if (!lane_id)
                                l3_apply_rx_feedback(window, observed, bulk_tx_epoch, v_begin, peer_v_begin);
#endif
                            if (!lane_id)
                            {
#if (L3_LIVE_SNAPSHOT == true)
                                l3_live_l3_state(4, bulk_tx_epoch, cnt);
#endif
                                l3_atomic_store_release<cuda::thread_scope_block>((int *)&bulk_tx_epoch_published, bulk_tx_epoch);
#if (L3_EVENT_RING == true)
                                l3_event_push(L3_EVENT_TX_PUBLISHED,
                                              bulk_tx_epoch, cnt, l3_eff,
                                              bulk_inbox_slot(bulk_tx_epoch));
#endif
                            }
                        }
                        else
                        {
#if (L3_LIVE_SNAPSHOT == true)
                            if (!lane_id)
                                l3_live_l3_state(5, bulk_tx_epoch, cnt);
#endif
                            // 槽位暂不可复用时，发布函数没有消费任何 inbox 数据；
                            // 把本批候选重新挂回发送侧协议，下一轮在接收侧推进后重试。
#if (L3_RETAIN_TX == true)
                            retained_count = cnt;
                            __syncwarp();
                            __threadfence_block();
                            // Keep busy=1 and skip the ordinary end-of-scan reset.
                            continue;
#else
                            bulk_requeue_l3_batch(
                                cnt, l3_lidx, l3_nd, remote_cand, remote_mark,
                                mark_hint, mark_hint2);
#endif
#if (L3_EVENT_RING == true)
                            if (!lane_id)
                                l3_event_push(L3_EVENT_TX_RETRY,
                                              next_tx_epoch, cnt, l3_eff,
                                              bulk_inbox_slot(next_tx_epoch));
#endif
                        }
                        __syncwarp();
                    }
                    else
#endif
                    {
#if (SEED_BARRIER == true)
#if (ASYNC_FB == true)
                    // 方案 B 双模式:
                    //   fill_mode（源卡 phase==2 灌值）: 只写 peer node_data + 追加 peer
                    //     seed_list（P2P 写 gpu1 侧收集缓冲，gpu1 注入免扫描直接消费）。
                    //     不置 dirty——seed_ready 屏障保证接收卡处理前看到全部最终值。
                    //   其余（源卡 phase3 反馈 / 接收卡 inject 后）: 生产异步路径——读返回值
                    //     判真改进 → peer dirty + remote_eff 计数（反向贡献异步回流）。
                    bool fill_mode = (n_gpu > 1 && (src >= v_begin && src < v_end)
                                      && (*(volatile int *)phase == 2));
                    if (fill_mode)
                    {
                        for (int i = lane_id; i < cnt; i += 32)
                        {
                            int lidx = l3_lidx[i];
                            l3_atomic_min_system(&peer_node_data[lidx], l3_nd[i]);
                            // B 开销优化: 对称预填充——灌值候选同时写本卡 peer 镜像，
                            //   使 gpu0 phase3 的 work 松弛 gpu1 顶点时被过滤（P2 成立图
                            //   消除 gpu0→gpu1 方向的无效跨卡候选）
#if (BULK_NO_CACHE == false)
                            atomicMin(&peer_cache[lidx], l3_nd[i]);
#endif
                            atomicAdd(&l3_eff, 1);   // B 开销优化: 灌值候选均有效（收敛后最终值）
                            if (peer_seed_list != NULL)
                            {
                                int slot = atomicAdd(peer_seed_list_cnt, 1);
                                peer_seed_list[slot] = node_struct(peer_v_begin + lidx, l3_nd[i]);
                            }
                        }
                        __syncwarp();
                        __threadfence_system();
                    }
                    else
                    {
                        for (int i = lane_id; i < cnt; i += 32)
                        {
                            int lidx = l3_lidx[i];
                            VALUE_TYPE old = l3_atomic_min_system(&peer_node_data[lidx], l3_nd[i]);
                            if (l3_nd[i] < old)
                            {
                                atomicAdd(&l3_eff, 1);   // B 开销优化: 真实改进计数（自适应）
                                atomicAdd(&g_remote_effective, 1ull);
                                if (remote_eff != NULL) atomicAdd(remote_eff, 1ull);
                                l3_system_mark_publish(&peer_dirty_bitmap[(lidx - 1) >> 5], 1u << ((lidx - 1) & 31));
                                l3_system_mark_publish(&peer_dirty_hint[(lidx - 1) >> 10], 1u << (((lidx - 1) >> 5) & 31));
                            }
                        }
                        __syncwarp();
                    }
#else
                    // 解法0 灌值期: 只写 peer node_data（最终值）+ 追加 peer seed_list
                    //   （P2P 写 gpu1 侧收集缓冲，gpu1 注入免 419万 扫描直接消费）。
                    //   seed_list 写+count 由 batch 后 threadfence_system + warp0 seed_ready
                    //   屏障保证对 peer 可见（warp0 见 l3_ok && mark_empty 才置 seed_ready）。
                    //   不置 dirty——由 seed_ready 屏障保证接收卡处理前看到全部最终值。
                    for (int i = lane_id; i < cnt; i += 32)
                    {
                        int lidx = l3_lidx[i];
                        // 性能优化: 灌值 node_data 普通 P2P 写（非 atomicMin）——灌值期 gpu1
                        //   work 门控（seed_inject_done 前不读），remote_cand 稳定（phase2 work 停），
                        //   每候选单值；普通写减少 L2 原子操作与缓存行污染（work_active 3.36→?ms）
                        peer_node_data[lidx] = l3_nd[i];
                        if (peer_seed_list != NULL)
                        {
                            int slot = atomicAdd(peer_seed_list_cnt, 1);
                            peer_seed_list[slot] = node_struct(peer_v_begin + lidx, l3_nd[i]);
                        }
                    }
                    __syncwarp();
                    __threadfence_system();
#endif
#else
                    // 批量跨卡原子：读返回值判断改进（原子完成即返回，NVLink 硬件流水化，
                    //   V0.3 吞吐 7.6e9/s，不串行等待）。仅真改进才置 peer dirty（防乒乓）。
                    // 发送侧已由 peer_cache 过滤（只发真改进候选），此处再过滤接收侧冗余。
                    // V0.4b 证明无 fence 也可（NVLink 原子序），读返回值本身即确认落位。
                    for (int i = lane_id; i < cnt; i += 32)
                    {
                        int lidx = l3_lidx[i];
                        VALUE_TYPE old = l3_atomic_min_system(&peer_node_data[lidx], l3_nd[i]);
                        if (l3_nd[i] < old)
                        {
                            atomicAdd(&g_remote_effective, 1ull);
                            l3_system_mark_publish(&peer_dirty_bitmap[(lidx - 1) >> 5], 1u << ((lidx - 1) & 31));
                            // E3: 同置 peer dirty_hint（每 32 个 dirty word 一个 hint bit）
                            //   dirty word = (lidx-1)>>5, hint word = word>>5 = (lidx-1)>>10, hint bit = word&31
                            l3_system_mark_publish(&peer_dirty_hint[(lidx - 1) >> 10], 1u << (((lidx - 1) >> 5) & 31));
                        }
                    }
                    __syncwarp();
#endif
                    }
                }
#if (MANAGE_PROFILE == true)
                if (!lane_id)
                {
                    unsigned p_t2 = clock();
                    p_scan += p_t1 - p_t0;
                    p_flush += p_t2 - p_t1;
                    p_total += p_t2 - p_t0;
                    p_iter++;
                }
#endif
                // 本轮收集+flush 完成：清在途
                if (!lane_id)
                {
                    l3_busy = 0;
#if (L3_LIVE_SNAPSHOT == true)
#if (BULK_ROUND == true)
                    l3_live_l3_state(6, bulk_tx_epoch, cnt);
#else
                    l3_live_l3_state(6, 0, cnt);
#endif
#endif
                }
                __syncwarp();
                __threadfence_block();
#if (L3_EFFECTIVE_FEEDBACK == true && L3_WINDOW_MODE == 2)
                if (!lane_id)
                    window.effective_feedback((unsigned)cnt,
                                              (unsigned)atomicAdd(&l3_eff, 0),
                                              L3_WINDOW_MAX_CYCLES);
#endif

                // E1b: 本轮无候选（cnt==0）→ 进入 quiet 态降频；有候选保持正常扫描
                // cnt 为 warp-uniform（l3_cnt 共享读后拷贝），直接整 warp 统一赋值
#if (L3_DYNAMIC == true)
                l3_dynamic_reset_after_scan(l3_dynamic, cnt);
#endif
                l3_quiet = (cnt == 0);
#if (BULK_ROUND == true && SEED_BARRIER == false)
                // 若 mark 在本轮收集期间产生，不能把 quiet 覆盖成 true；下一轮
                // 必须继续扫描，避免事件信号在扫描边界产生时被漏掉。
                unsigned long long signal_before = l3_signal_seen;
                unsigned long long signal_after = 0;
                if (!lane_id)
                    signal_after = atomicAdd(&g_bulk_mark_signal, 0ull);
                signal_after = __shfl_sync(FULL_MASK, signal_after, 0);
                l3_signal_seen = signal_after;
                if (signal_after != signal_before)
                    l3_quiet = false;
#endif
                // B 开销优化: 扫描间隔自适应（cnt>0 且全部无效→256；有真实改进→32；
                //   quiet 空转→至少 128）。l3_eff 在 flush 循环后 __syncwarp 已同步。
#if (ASYNC_FB == true)
                if (cnt > 0)
                {
                    if (l3_eff > 0)
                    {
                        l3_k = 32;
                        l3_invalid_streak = 0;
                    }
                    else
                    {
                        l3_invalid_streak++;
                        if (l3_invalid_streak >= 8) l3_k = 256;
                    }
                }
                else
                {
                    if (l3_k < 128) l3_k = 128;
                    l3_invalid_streak = 0;
                }
#endif
#if (MANAGE_PROFILE == true && HANG_DIAG == true)
                // E3 实时 profile：warp17 扫描/跨卡 flush 占比（大图 kernel 不终止，等不到结束打印）
                if (n_gpu > 1 && !lane_id && (p_iter % 100000 == 0) && p_iter > 0)
                    printf("w17 iters=%u cnt=%d scan=%.0f%% flush=%.0f%%\n",
                           p_iter, cnt, 100.0 * p_scan / p_total, 100.0 * p_flush / p_total);
#endif
                __syncwarp();
            }
#if (L3_WAIT_DIAG == true)
            tx_wait.save(257);
#endif
#if (MANAGE_PROFILE == true)
            if (!lane_id && mgmt_profile != NULL)
            {
                mgmt_profile[5] = p_scan;
                mgmt_profile[6] = p_flush;
                mgmt_profile[7] = p_total;
                mgmt_profile[8] = p_iter;
            }
#endif
        }
        __syncwarp();
        }
    }
    // ===== E2: 注入 warp（各扫一个 dirty 区间切片，独立收集 + write_through）=====
    // warp id: local_wid = manage_warp_num()+2 .. manage_warp_num()+1+INJECT_WARP_NUM
    // L2DQ write 为 MRMW 安全（consistent.txt §3：atomicAdd(write_reserve) 预留非重叠空间），
    // 多注入 warp 并发 write_through 无冲突。注入只写 L2，不 simple_process，
    // 无 v1 backpressure 环（v1 教训已规避）。终止由 warp0 全量检查兜底。
    else
    {
        extern __shared__ int s[];
        NODE_TYPE *node_buf = (NODE_TYPE*)(s + qshm_size / sizeof(int));
        int inj_id = local_wid - (mlmq.manage_warp_num() + 2);
#if (L3_RECOVERY_MODE > 0)
        l3_recovery_cursor recovery_cursor;
#endif
        // E2: node_in 按 warp 切片（串行 lane0 收集，容量 node_size）。
        //   warp0 用 [0,node_size)，注入 warp i 用 [(1+i)*node_size, ...)。
        //   布局上限 2*node_size*WARP_NUM_PER_BLOCK=1024（warp0 node_out 预留，未用）。
        NODE_TYPE *node_in = node_buf + (1 + inj_id) * node_size;

        int dwords = (v_local + 31) / 32;
        // E3: 注入区间切片按 hint-word（32 个 dirty word 一块）切分，保证块对齐。
#if (BULK_ROUND == true)
        // BULK_ROUND 中 warp0 不消费 dirty，全部切片由注入 warp 负责。
        int ncons = INJECT_WARP_NUM;
#else
        int ncons = 1 + INJECT_WARP_NUM;
#endif
        int hwords = (dwords + 31) / 32;
        int h_slice = hwords / ncons;
        int h_rem = hwords % ncons;
#if (BULK_ROUND == true)
        int cons_id = inj_id;
#else
        int cons_id = inj_id + 1;
#endif
        int hb_beg = cons_id * h_slice + (cons_id < h_rem ? cons_id : h_rem);
        int hb_end = hb_beg + h_slice + (cons_id < h_rem ? 1 : 0);
        int c_beg = hb_beg * 32;
        int c_end = hb_end * 32;

        int node_in_num = 0;
        unsigned debug_time[3] = {0, 0, 0};
#if (BULK_ROUND == true && L3_TERM_HANDSHAKE == true && \
     BULK_EPOCH == false && GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && \
     SEED_BARRIER == false)
        int l3_term_inject_seen = 0;
#endif
#if (GHOST_DEPTH > 0)
        int ghost_round_g = 0;   // 方案 C: ghost 处理节流计数（独立于 inj_quiet）
#endif
#if (SP_ASYNC_BF == true)
        // exp-sp_async: BF 化注入的松弛计数（relax_dst 的 total_work 汇聚）
        int bf_total_work = 0;
#endif
        // E2: 注入空转降频（本轮扫完 slice 无 dirty -> quiet，每 INJ_K 轮才扫一次；
        // 有候选下轮恢复。延迟最多 INJ_K 轮，由 backstop + 终止全量检查兜底正确性）
        // B 开销优化: 注入空扫降频 32→128（dirty_hint 稀疏扫 4096 words 的轮次开销；
        //   注入延迟由终止二次确认 + nohint 兜底保证正确性）
#if (ASYNC_FB == true)
        const int INJ_K = 128;
#else
        // SEED_BARRIER: 接收卡注入 warp 降频（注入完成后 dirty 少——P2 成立图无跨卡反馈，
        //   注入 warp 每 INJ_K 轮扫 dirty_hint 与 work 竞争 L2；SEED_EXP 对照无此竞争。
        //   正确性: dirty 延迟处理，终止二次确认 + nohint 兜底）
        const int INJ_K = (src >= v_begin && src < v_end) ? 32 : 1024;
#endif
        int inj_round = 0;
        bool inj_quiet = false;
        // SESSION 22: quiet 态周期性全扫 dirty_bitmap 兜底（清 dirty_hint ABA 假阴残留，
        //   与 w17 的 L3_FULL 同构）。每 INJ_K*INJ_FULL 轮走一次 nohint 全扫。
        const int INJ_FULL = 32;
        int inj_full_round = 0;
#if (GLOBAL_ROUND == true)
        bool global_round_inj_quiesced = false;
#endif

        while (l3_atomic_load_acquire<cuda::thread_scope_block>(&manager_end) == 0)
        {
#if (L3_LIVE_SNAPSHOT == true)
            if (!lane_id)
            {
                atomicAdd(&g_l3_live_state.inject_iter[inj_id], 1ull);
                l3_live_inject_state(inj_id, 1);
            }
#endif
#if (GLOBAL_ROUND_ASYNC == true)
            // inbox 已由 manager warp 直接 apply，异步路径不产生 dirty
            // 注入项。注入 warp 只保留 backstop 请求响应，避免六个 warp
            // 周期性扫描整张 dirty bitmap 与 work 竞争 L2。
            if (n_gpu > 1 && node_in_num == 0 && !(*(volatile int *)&bs_req))
            {
                __threadfence();
                continue;
            }
#endif
#if (GLOBAL_ROUND == true)
            // GLOBAL_ROUND 的安全点不仅要求 work warp ack，还要求每个注入 warp
            // 结束当前 dirty 收集/写入。否则 manager 可能在 inj_pending 仍为 0
            // 的窗口启动 full pack，而该 warp 随后才把旧 dirty 写入 L2。
            if (n_gpu > 1 && bulk_quiesce_req != NULL)
            {
                int gr_req = l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req);
                if (gr_req == 0)
                {
                    global_round_inj_quiesced = false;
                }
                else if (global_round_inj_quiesced)
                {
                    __threadfence();
                    continue;
                }
                else if (node_in_num == 0)
                {
                    if (!lane_id)
                        l3_atomic_fetch_add_acq_rel<cuda::thread_scope_block>(&bulk_inject_ack, 1);
                    __syncwarp();
                    global_round_inj_quiesced = true;
                    __threadfence();
                    continue;
                }
            }
#endif
#if (BULK_ROUND == true)
            // 默认 BULK 终止握手：注入 warp 也必须停止在一个明确安全点。
            // 每个 warp 写独立 token ACK；manager 不使用 bulk_inject_ack，避免
            // GLOBAL_ROUND 的共享计数与默认终止协议相互污染。
#if (L3_TERM_HANDSHAKE == true && BULK_EPOCH == false && \
     GLOBAL_ROUND == false && GLOBAL_ROUND_ASYNC == false && SEED_BARRIER == false)
            if (n_gpu > 1 && l3_term_req != NULL)
            {
                int term_req_now = l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req);
                if (term_req_now == 0)
                    l3_term_inject_seen = 0;
                bool term_inject_ready =
                    (node_in_num == 0 && !(*(volatile int *)&bs_req));
                unsigned term_inject_mask = __ballot_sync(
                    FULL_MASK, term_inject_ready);
                if (term_req_now != 0 && term_inject_mask == FULL_MASK)
                {
                    if (l3_term_inject_seen != term_req_now)
                    {
                        if (!lane_id)
                            l3_atomic_store_release<cuda::thread_scope_block>(l3_term_inject_ack + inj_id, term_req_now);
                        l3_term_inject_seen = term_req_now;
                    }
                    __syncwarp();
#if (L3_LIVE_SNAPSHOT == true)
                    if (!lane_id)
                        l3_live_inject_state(inj_id, 2);
#endif
                    while (l3_atomic_load_acquire<cuda::thread_scope_device>(l3_term_req) == term_req_now
                           && l3_atomic_load_acquire<cuda::thread_scope_device>(global_exit) == 0)
                    {
                        __threadfence();
                    }
                    __syncwarp();
                    continue;
                }
            }
#endif
            // work 静止请求期间，注入 warp 先完成当前 node_in；随后停止扫描 dirty，
            // 防止 manager 已开始打包时又产生本地注入/远程 mark 竞争。
            if (n_gpu > 1 && bulk_quiesce_req != NULL
                && l3_atomic_load_acquire<cuda::thread_scope_device>(bulk_quiesce_req) != 0 && node_in_num == 0
                // 终止二次确认可能同时发起 backstop_collaborative。
                // bs_req 的响应优先于 quiesce：否则注入 warp 会在这里空转，
                // 而 warp0 永远等不到 bs_done，形成异步终止互锁。
                && !(*(volatile int *)&bs_req))
            {
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                    l3_live_inject_state(inj_id, 3);
#endif
                __threadfence();
                continue;
            }
#endif
            // H1: backstop 协作请求响应（必须在 inj_quiet 分支之前，防漏响应）。
            //   响应后 continue，不污染 inj_round/inj_quiet。
#if (L3_RECOVERY_MODE > 0)
            const int recovery_extra = L3_RECOVERY_MODE == 2 ? mlmq.manage_warp_num() : 0;
            if (l3_recovery_service(recovery_cursor, 1 + recovery_extra + inj_id,
                    1 + recovery_extra + INJECT_WARP_NUM, v_local, lane_id,
                    node_data, last_processed, dirty_bitmap, dirty_hint,
                    &bs_req, &bs_done, &bs_found)) continue;
#else
            if (*(volatile int *)&bs_req)
            {
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                    l3_live_inject_state(inj_id, 4);
#endif
                const int n_warp = 1 + INJECT_WARP_NUM;
                int per = v_local / n_warp;
                int rem = v_local % n_warp;
                int wi = inj_id + 1;   // 段号
                int seg_beg = 1 + wi * per + (wi < rem ? wi : rem);
                int seg_end = seg_beg + per + (wi < rem ? 1 : 0);
                bool f = backstop_scan_range(node_data, last_processed, dirty_bitmap, dirty_hint,
                                             seg_beg, seg_end, lane_id);
                unsigned fm = __ballot_sync(FULL_MASK, f);
                if (fm && !lane_id) atomicOr(&bs_found, 1);
                if (!lane_id) atomicAdd(&bs_done, 1);
                __syncwarp();
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                    l3_live_inject_state(inj_id, 5);
#endif
                while (*(volatile int *)&bs_req) { __threadfence(); }   // 等本轮结束
                continue;
            }
#endif

#if (SEED_BARRIER == true)
#if (SEED_PHASE2_PARALLEL == true)
            // phase2 源卡的注入 warp 与 L3 warp 一起分片发布 remote_mark。
            if (n_gpu > 1 && (src >= v_begin && src < v_end)
                && phase != NULL && (*(volatile int *)phase == 2))
            {
                seed_phase2_publish_warp(
                    inj_id + 1, 1 + INJECT_WARP_NUM, lane_id, m,
                    phase, &seed_pub_busy,
                    peer_v_begin, peer_v_local,
                    remote_cand, remote_mark, mark_hint, mark_hint2,
                    peer_cache, peer_node_data,
                    seed_list, seed_list_cnt);
                continue;
            }
#else
            // 回退路径：源卡 phase2 由 L3 warp 单独灌值，注入 warp 空转。
            if (n_gpu > 1 && (src >= v_begin && src < v_end)
                && (*(volatile int *)phase == 2))
            {
                __threadfence();
                continue;
            }
#endif
#endif
#if (SEED_BARRIER == true)
            // 解法0: 接收卡注入完成前禁注入（seed_ready 前防处理部分种子；
            //   注入期间（seed_ready 后 inject_done 前）防与 warp0 争抢 dirty 收集）。
            //   源卡不受限（其 dirty 只来自自身 backstop，与灌值无关）。
            if (n_gpu > 1 && !(src >= v_begin && src < v_end)
                && seed_inject_done != NULL && (*(volatile int *)seed_inject_done == 0))
            {
                atomicAdd(&g_inj_gate_skip, 1ull);
                if (!g_t_sr_seen) atomicExch(&g_t_sr_seen, TIMELINE_CLOCK());
                __threadfence();
                continue;
            }
#endif

            if (inj_quiet)
            {
                inj_round++;
                if (inj_round % INJ_K != 0) { __threadfence(); continue; }
                inj_round = 0;
                inj_full_round++;
            }

#if (GHOST_DEPTH > 0)
            // 方案 C: ghost 改进处理（仅 inj_id==0 单 warp；ghost_mark 原子取走，超容量回填不丢失）。
            //   与注入 quiet 同节奏（每 INJ_K 轮），避免每轮空扫。ghost 出边松弛发现
            //   "穿到对端绕回本卡"的路径：返回边改进置 dirty（注入路径 write_through 处理），
            //   ghost 间边改进继续 BF 扩散（深度 GHOST_BF_DEPTH）。
            //   注: ghost 缓冲复用 rshm 区（与 SP_ASYNC_BF 的 bf_buf 互斥，不同时开启）。
            if (inj_id == 0 && ghost_row_start != NULL && node_in_num == 0)
            {
                ghost_round_g++;
                if (ghost_round_g % INJ_K == 0)
                {
                    // 快速空检查（lane 并行），空则跳过收集/BF
                    bool gmark_empty = true;
                    int ghost_words = (ghost_num + 31) / 32;
                    for (int w = lane_id; w < ghost_words; w += 32)
                        if (l3_atomic_load_acquire<cuda::thread_scope_device>(&ghost_mark[w])) gmark_empty = false;
                    __syncwarp();
                    unsigned gme = __ballot_sync(FULL_MASK, gmark_empty);
                    if (gme != FULL_MASK)
                    {
                        int *sbase_g = s + (qshm_size + nshm_size) / sizeof(int);
                        int *g_buf = sbase_g + inj_id * (2 * GHOST_BF_CAP + 1);
                        int *g_slot = g_buf + 2 * GHOST_BF_CAP;
                        if (!lane_id) *g_slot = 0;
                        __syncwarp();
                        for (int w = lane_id; w < ghost_words; w += 32)
                        {
                            unsigned val = l3_atomic_load_acquire<cuda::thread_scope_device>(&ghost_mark[w]);
                            if (!val) continue;
                            if (!l3_device_mark_claim(&ghost_mark[w], val)) continue;
                            for (int b = 0; b < 32; b++)
                            {
                                if (!(val & (1u << b))) continue;
                                int gi = w * 32 + b;
                                if (gi >= ghost_num) continue;
                                int slot = atomicAdd(g_slot, 1);
                                if (slot < GHOST_BF_CAP)
                                    g_buf[slot] = gi;
                                else
                                    l3_device_mark_publish(
                                        &ghost_mark[gi >> 5], 1u << (gi & 31));
                            }
                        }
                        __syncwarp();
                        int g_in_cnt = __shfl_sync(FULL_MASK, *g_slot, 0);
                        if (g_in_cnt > GHOST_BF_CAP) g_in_cnt = GHOST_BF_CAP;
                        int *cur_in = g_buf;
                        for (int d = 0; d < GHOST_BF_DEPTH && g_in_cnt > 0; d++)
                        {
                            int *next_out = (d & 1) ? g_buf : g_buf + GHOST_BF_CAP;
                            g_in_cnt = ghost_bf_relax(cur_in, g_in_cnt,
                                                      ghost_row_start, ghost_col, ghost_edge_data,
                                                      ghost_id_to_idx, peer_v_begin,
                                                      node_data, dirty_bitmap, dirty_hint,
                                                      v_begin, v_end, v_local,
                                                      ghost_node_data,
                                                      remote_cand, remote_mark, mark_hint, mark_hint2,
                                                      next_out, GHOST_BF_CAP, g_slot);
                            cur_in = next_out;
                        }
                        __syncwarp();
                    }
                }
            }
#endif

            // 扫描本 slice 脏位图取改进顶点（不消费 L2）。lane 并行收集
            if (node_in_num == 0 && n_gpu > 1)
            {
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                    l3_live_inject_state(inj_id, 6);
#endif
                bool nohint = inj_quiet && (inj_full_round % INJ_FULL == 0);
                if (nohint)
                    collect_dirty_slice_nohint(dirty_bitmap, dirty_hint, c_beg, c_end, v_local, v_begin,
                                               node_data, node_in, node_in_num, node_size);
                else
                    collect_dirty_slice(dirty_bitmap, dirty_hint, c_beg, c_end, v_local, v_begin,
                                        node_data, node_in, node_in_num, node_size);
                __syncwarp();
            }

            if (node_in_num > 0)
            {
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                    l3_live_inject_state(inj_id, 7);
#endif
#if (BULK_DIAG == true)
                if (!lane_id)
                {
                    unsigned long long p = atomicAdd(&g_bulk_inj_prints, 1ull);
                    if (p < 32)
                        printf("BULK_INJ g%d warp=%d n=%d q_before=%d\\n",
                               v_begin, inj_id, node_in_num, mlmq.get_global_queue_size());
                }
#endif
                // E2: 在途计数（write_through 前 +n / 后 -n），warp0 终止检查读，防提前终止
                if (!lane_id) atomicAdd(&inj_pending, node_in_num);
                __syncwarp();
#if (SP_ASYNC_BF == true && 1)
                // exp-sp_async: 注入顶点深度 BF 化——对注入顶点做多层 BF 扩散：
                //   每层 BF 松弛输入 buffer 顶点的本地出边，赢的邻居置 dirty（队列感知）
                //   + 写入下一层 buffer，逐层扩散直到无新赢者或达 BF_MAX_DEPTH。
                //   绕开 delta 桶序假设（SP Async: 本地 Dijkstra + 跨进程 BF 思想）。
                //   bf_buf 用 rshm 区（manage kernel 未使用，s+qshm+nshm 起，容量 4096
                //   NODE_TYPE）：per 注入 warp 专属 [2*BF_DEPTH_CAP NODE_TYPE + 1 int]。
                {
                int *sbase = s + (qshm_size + nshm_size) / sizeof(int);   // rshm 区基址
                NODE_TYPE *bf_buf = (NODE_TYPE *)sbase
                                    + inj_id * (2 * BF_DEPTH_CAP + 1);
                NODE_TYPE *bf_next = bf_buf;
                int *bf_slot = (int *)(bf_buf + 2 * BF_DEPTH_CAP);
                int bf_depth = 0;
                int cur_num = node_in_num;
                // 深度 BF 循环（第0层输入 = 原注入顶点 node_in）
                for (bf_depth = 0; bf_depth < BF_MAX_DEPTH; bf_depth++)
                {
                    NODE_TYPE *cur_in = (bf_depth == 0) ? node_in : (bf_depth & 1 ? bf_buf + BF_DEPTH_CAP : bf_buf);
                    NODE_TYPE *next_out = (bf_depth & 1) ? bf_buf : bf_buf + BF_DEPTH_CAP;
                    int out_cnt = inject_bf_relax(cur_in, cur_num, RowPtr, ColIdx, edge_data, node_data,
                                                  v_begin, v_end, remote_cand, remote_mark, mark_hint, mark_hint2,
                                                  peer_cache, peer_v_begin, dirty_bitmap, dirty_hint, v_local,
                                                  next_out, BF_DEPTH_CAP, bf_slot, bf_total_work);
                    __syncwarp();
                    if (out_cnt == 0) break;
                    cur_num = out_cnt;
                }
                (void)bf_depth;
                __syncwarp();
                }
#endif
                int write_num = node_in_num;
                mlmq.write_through(node_in, write_num, 0, 0, lane_id, debug_time);
                __syncwarp();
                if (!lane_id) atomicAdd(&inj_pending, -node_in_num);
                __syncwarp();
#if (BULK_DIAG == true)
                if (!lane_id)
                {
                    unsigned long long p = atomicAdd(&g_bulk_inj_prints, 0ull);
                    if (p < 32)
                        printf("BULK_INJ_DONE g%d warp=%d q_after=%d\\n",
                               v_begin, inj_id, mlmq.get_global_queue_size());
                }
#endif
                node_in_num = 0;
                inj_quiet = false;
#if (L3_LIVE_SNAPSHOT == true)
                if (!lane_id)
                    l3_live_inject_state(inj_id, 8);
#endif
            }
            else
            {
                // 本轮扫完 slice 无 dirty -> quiet 降频（有候选下轮恢复）
                inj_quiet = true;
            }
            __syncwarp();
        }
#if (SP_ASYNC_BF == true)
        // exp-sp_async: BF 化注入统计（每个注入 warp 各自打印）
        if (!lane_id)
            printf("vbeg%d inj%d bf_total_work=%d (BF 化注入松弛赢次数)\n",
                   v_begin, inj_id, bf_total_work);
#endif
    }

    __syncthreads();
}

// 每卡独立初始化：填充 gctx[gpu_id]，本卡 node_data 只含归属区间
int sssp_init(int gpu_id, int m_in, int nnz_in, int *RowPtr_in, int *ColIdx_in,
              VALUE_TYPE *edge_data_in, int v_begin_in, int v_end_in)
{
    cudaSetDevice(gpu_id);
    gctx[gpu_id].gpu_id = gpu_id;
    gctx[gpu_id].m = m_in;
    gctx[gpu_id].nnz = nnz_in;
    gctx[gpu_id].RowPtr = RowPtr_in;
    gctx[gpu_id].ColIdx = ColIdx_in;
    gctx[gpu_id].edge_data = edge_data_in;
    gctx[gpu_id].v_begin = v_begin_in;
    gctx[gpu_id].v_end = v_end_in;
    gctx[gpu_id].v_local = v_end_in - v_begin_in;

    VALUE_TYPE *nd = NULL;
    cudaMalloc(&nd, sizeof(VALUE_TYPE) * (gctx[gpu_id].v_local + 1));

    VALUE_TYPE init_max = DIST_MAX;
    VALUE_TYPE *node_data_h = new VALUE_TYPE[gctx[gpu_id].v_local + 1];
    for (int i = 0; i <= gctx[gpu_id].v_local; i++)
        node_data_h[i] = init_max;
    cudaMemcpy(nd, node_data_h, sizeof(VALUE_TYPE) * (gctx[gpu_id].v_local + 1), cudaMemcpyHostToDevice);
    delete[] node_data_h;

    gctx[gpu_id].node_data = nd;

    cudaMemcpyToSymbol(node_data_dev, &gctx[gpu_id].node_data, sizeof(VALUE_TYPE*));

    cudaMalloc(&gctx[gpu_id].global_exit, sizeof(int));
    cudaMemset(gctx[gpu_id].global_exit, 0, sizeof(int));

    // CP2b: L3 本地接收/终止资源由唯一 host owner 分配。
    gctx[gpu_id].initialize(gpu_id);
    gctx[gpu_id].allocate_local(gctx[gpu_id].v_local);

    // SEED_BARRIER: phase（源卡状态机，初值 1=计算）+ seed_ready（接收卡屏障，初值 0）
    cudaMalloc(&gctx[gpu_id].phase, sizeof(int));
    int ph_init = 1;
    cudaMemcpy(gctx[gpu_id].phase, &ph_init, sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&gctx[gpu_id].seed_ready, sizeof(int));
    cudaMemset(gctx[gpu_id].seed_ready, 0, sizeof(int));
    gctx[gpu_id].peer_seed_ready = NULL;
    cudaMalloc(&gctx[gpu_id].seed_inject_done, sizeof(int));
    cudaMemset(gctx[gpu_id].seed_inject_done, 0, sizeof(int));
    // phase2 并行发布的本地列表随后会复制给 peer；使用 m+1 容量覆盖不等分分区，
    // 并为每个 peer 顶点最多一个最终条目提供安全上限。
    cudaMalloc(&gctx[gpu_id].seed_list, sizeof(NODE_TYPE) * (gctx[gpu_id].m + 1));
    cudaMalloc(&gctx[gpu_id].seed_list_cnt, sizeof(int));
    cudaMemset(gctx[gpu_id].seed_list_cnt, 0, sizeof(int));
    gctx[gpu_id].peer_seed_list_cnt = NULL;
#if (GHOST_DEPTH > 0)
    gctx[gpu_id].ghost_num = 0;
    gctx[gpu_id].ghost_row_start = NULL;
    gctx[gpu_id].ghost_col = NULL;
    gctx[gpu_id].ghost_edge_data = NULL;
    gctx[gpu_id].ghost_node_data = NULL;
    gctx[gpu_id].ghost_id_to_idx = NULL;
    gctx[gpu_id].ghost_mark = NULL;
#endif

    cudaDeviceSynchronize();

    return 0;
}

#if (GHOST_DEPTH > 0)
// 方案 C: 源卡设置 ghost 子图（须在 sssp_setup_peers 之后调用，peer_v_local 已就绪）
int sssp_set_ghost(int gpu_id, int ghost_num, int *ghost_row_start_h, int *ghost_col_h,
                   VALUE_TYPE *ghost_edge_data_h, int *g_id_to_idx_h)
{
    cudaSetDevice(gpu_id);
    gctx[gpu_id].ghost_num = ghost_num;
    int *grs = NULL, *gc = NULL, *gid = NULL;
    VALUE_TYPE *ged = NULL, *gnd = NULL;
    unsigned *gm = NULL;
    if (ghost_num > 0)
    {
        cudaMalloc(&grs, sizeof(int) * (ghost_num + 1));
        cudaMemcpy(grs, ghost_row_start_h, sizeof(int) * (ghost_num + 1), cudaMemcpyHostToDevice);
        int nedges = ghost_row_start_h[ghost_num];
        cudaMalloc(&gc, sizeof(int) * nedges);
        cudaMemcpy(gc, ghost_col_h, sizeof(int) * nedges, cudaMemcpyHostToDevice);
        cudaMalloc(&ged, sizeof(VALUE_TYPE) * nedges);
        cudaMemcpy(ged, ghost_edge_data_h, sizeof(VALUE_TYPE) * nedges, cudaMemcpyHostToDevice);
        cudaMalloc(&gnd, sizeof(VALUE_TYPE) * ghost_num);
        VALUE_TYPE *gh_h = new VALUE_TYPE[ghost_num];
        for (int i = 0; i < ghost_num; i++) gh_h[i] = DIST_MAX;
        cudaMemcpy(gnd, gh_h, sizeof(VALUE_TYPE) * ghost_num, cudaMemcpyHostToDevice);
        delete[] gh_h;
        int peer_v_local = gctx[gpu_id].peer_v_local;
        cudaMalloc(&gid, sizeof(int) * peer_v_local);
        cudaMemcpy(gid, g_id_to_idx_h, sizeof(int) * peer_v_local, cudaMemcpyHostToDevice);
        cudaMalloc(&gm, sizeof(unsigned) * ((ghost_num + 31) / 32));
        cudaMemset(gm, 0, sizeof(unsigned) * ((ghost_num + 31) / 32));
    }
    gctx[gpu_id].ghost_row_start = grs;
    gctx[gpu_id].ghost_col = gc;
    gctx[gpu_id].ghost_edge_data = ged;
    gctx[gpu_id].ghost_node_data = gnd;
    gctx[gpu_id].ghost_id_to_idx = gid;
    gctx[gpu_id].ghost_mark = gm;
    cudaDeviceSynchronize();
    return 0;
}
#endif

// 设置每卡的 peer 指针（跨卡直访对方 node_data/dirty_bitmap/local_idle）
// 须在全部卡 init 完成后调用（n_gpu=1 时 peer 全空）
int sssp_setup_peers(int n_gpu)
{
    g_n_gpu = n_gpu;
    if (n_gpu == 1)
    {
        for (int i = 0; i < n_gpu; i++)
        {
#if (L3_BOUNDARY_INDEX == true)
            l3_boundary_release(i);
#endif
#if (L3_RECOVERY_DOMAIN_DIAG == true)
            l3_domain_release(i);
#endif
            gctx[i].disable();
            gctx[i].peer_seed_ready = NULL;
            gctx[i].peer_seed_list_cnt = NULL;
            gctx[i].peer_seed_list = NULL;
        }
        return 0;
    }

    // CP2b 仍使用唯一 peer。第一轮由各 owner 分配本地 candidate/RX 资源，
    // 并绑定不依赖对端 transport 分配顺序的数据面指针。
    for (int i = 0; i < n_gpu; i++)
    {
        int j = (i + 1) % n_gpu;
        gctx[i].allocate_peer(j, gctx[j].v_begin, gctx[j].v_local);
#if (L3_BOUNDARY_INDEX == true)
        l3_boundary_build(i,gctx[i].RowPtr,gctx[i].ColIdx,gctx[i].v_local,
                          gctx[j].v_begin,gctx[j].v_local);
#endif
        gctx[i].bind_peer_data(gctx[j].node_data, gctx[j].dirty_bitmap,
                              gctx[j].dirty_hint, gctx[j].local_idle);
#if (L3_TILE_LOAN == true)
        // P2a helper reads donor CSR through the same P2P/UVA mapping as the
        // existing peer distance transport; it never reads the peer queue.
        gctx[i].bind_peer_graph(gctx[j].RowPtr, gctx[j].ColIdx,
                                gctx[j].edge_data);
#endif
        // SEED_BARRIER: peer seed_ready（源卡 P2P 写，接收卡读自己的 seed_ready）
        gctx[i].peer_seed_ready = gctx[j].seed_ready;
        // SEED_BARRIER: peer seed_list_cnt（源卡灌值时 P2P atomicAdd，接收卡读）
        gctx[i].peer_seed_list_cnt = gctx[j].seed_list_cnt;
        gctx[i].peer_seed_list = gctx[j].seed_list;
    }

    // 第二轮绑定 receiver-owned transport，避免先看到尚未分配的 peer inbox。
    for (int i = 0; i < n_gpu; i++)
    {
        int j = (i + 1) % n_gpu;
        gctx[i].bind_peer_transport(gctx[j]);
#if (L3_RECOVERY_DOMAIN_DIAG == true)
        l3_domain_setup(i,gctx[i].v_local,l3_boundary_host_words[j]);
#endif
#if (BULK_DIAG == true)
        printf("BULK_HOST_PTR gpu%d inbox=%p ack=%p peer_gpu%d inbox=%p ack=%p\\n",
               i, (void *)gctx[i].bulk_inbox, (void *)gctx[i].bulk_inbox_ack,
               j, (void *)gctx[i].peer_bulk_inbox,
               (void *)gctx[i].peer_bulk_inbox_ack);
#endif
    }
    return 0;
}

// 多源数据并行模式：每张物理卡都持有完整图，但每次 kernel 运行都必须按
// n_gpu=1 的本地逻辑执行。不能调用 sssp_setup_peers(n_gpu)，否则 kernel 会
// 把另一张独立副本误认为跨卡 peer，重新进入 BULK/SEED 通信协议。
int sssp_setup_independent(int physical_gpu_count)
{
    if (physical_gpu_count < 1 || physical_gpu_count > MAX_GPU)
        return -1;

    // g_n_gpu 是 device kernel 使用的全局模式开关；物理卡数量由 host
    // worker 数量表示，不应写入这个逻辑值。
    g_n_gpu = 1;
    for (int i = 0; i < physical_gpu_count; i++)
    {
#if (L3_BOUNDARY_INDEX == true)
        l3_boundary_release(i);
#endif
#if (L3_RECOVERY_DOMAIN_DIAG == true)
        l3_domain_release(i);
#endif
        gctx[i].disable();
        gctx[i].peer_seed_ready = NULL;
        gctx[i].peer_seed_list_cnt = NULL;
        gctx[i].peer_seed_list = NULL;
    }
    return 0;
}

// SEED_EXP: 预置边界最终值种子（device 符号，work kernel 初始化时注入 L2）
// seed_ids 为全局 1-based 顶点 id；seed_dists 为对应最终距离（host 从 CPU 参考解提取）
#if (SEED_EXP == true)
int sssp_set_seeds(int gpu_id, int n_seeds, int *seed_ids, VALUE_TYPE *seed_dists)
{
    cudaSetDevice(gpu_id);
    int *ids_d = NULL;
    VALUE_TYPE *dists_d = NULL;
    cudaMalloc(&ids_d, sizeof(int) * (n_seeds > 0 ? n_seeds : 1));
    cudaMalloc(&dists_d, sizeof(VALUE_TYPE) * (n_seeds > 0 ? n_seeds : 1));
    if (n_seeds > 0)
    {
        cudaMemcpy(ids_d, seed_ids, sizeof(int) * n_seeds, cudaMemcpyHostToDevice);
        cudaMemcpy(dists_d, seed_dists, sizeof(VALUE_TYPE) * n_seeds, cudaMemcpyHostToDevice);
    }
    cudaMemcpyToSymbol(g_seed_ids, &ids_d, sizeof(ids_d));
    cudaMemcpyToSymbol(g_seed_dists, &dists_d, sizeof(dists_d));
    cudaMemcpyToSymbol(g_seed_num, &n_seeds, sizeof(n_seeds));
    cudaDeviceSynchronize();
    return 0;
}
#endif

// 单卡模式下用到的全局指针别名（旧接口兼容）
int sssp_init(int m_in, int nnz_in, int *RowPtr_in, int *ColIdx_in, VALUE_TYPE *edge_data_in)
{
    sssp_init(0, m_in, nnz_in, RowPtr_in, ColIdx_in, edge_data_in, 0, m_in);
    node_data = gctx[0].node_data;
    global_exit = gctx[0].global_exit;
    return 0;
}

int sssp_re_init(int gpu_id)
{
    VALUE_TYPE init_max;
    if (typeid(VALUE_TYPE) == typeid(int))
        init_max = INT_MAX;
    else if (typeid(VALUE_TYPE) == typeid(float))
        init_max = FLT_MAX;
    else if (typeid(VALUE_TYPE) == typeid(double))
        init_max = DBL_MAX;
    else
    {
        printf("VALUE TYPE error!\n");
        return -1;
    }

    cudaSetDevice(gpu_id);
#if (L3_CHAIN_PARTITION_DIAG == true)
    unsigned long long zero_chain_counts[8]={};
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_chain_partition_counts,zero_chain_counts,sizeof(zero_chain_counts))==cudaSuccess,"chain counters reset");
    l3_chain_partition_diag_reset(gpu_id);
#endif
#if (L3_REGION_DIAG == true)
    unsigned long long region_zero[4]={};
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_region_counts,region_zero,sizeof(region_zero))==cudaSuccess,"region counters reset");
#endif
    int v_local = gctx[gpu_id].v_local;
#if (L3_RECOVERY_DOMAIN_DIAG == true)
    l3_domain_reset();
#endif
    g_benchmark.require(l3_fill_values(gctx[gpu_id].node_data, size_t(v_local) + 1,
                        init_max) == cudaSuccess, "distance reset launch");

    cudaMemset(gctx[gpu_id].global_exit, 0, sizeof(int));
    gctx[gpu_id].reset_query(init_max);
#if (L3_RX_FEEDBACK_TRACE == true)
    unsigned feedback_trace_zero = 0;
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_feedback_trace_head,
                        &feedback_trace_zero, sizeof(feedback_trace_zero)) == cudaSuccess,
                        "feedback trace reset");
#endif
    // SEED_BARRIER: 重置 phase=1（计算期）+ seed_ready=0（屏障未触发）+ 注入完成=0
    int ph_init = 1;
    cudaMemcpy(gctx[gpu_id].phase, &ph_init, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(gctx[gpu_id].seed_ready, 0, sizeof(int));
    cudaMemset(gctx[gpu_id].seed_inject_done, 0, sizeof(int));
    cudaMemset(gctx[gpu_id].seed_list_cnt, 0, sizeof(int));
#if (L3_FAULT_INJECT_CLAIM_RETRY == true)
    unsigned int fault_claim_zero = 0;
    cudaMemcpyToSymbol(g_l3_fault_claim_retry, &fault_claim_zero,
                       sizeof(fault_claim_zero));
#endif
#if (L3_FAULT_INJECT_READY_DELAY == true)
    unsigned int fault_ready_zero = 0;
    cudaMemcpyToSymbol(g_l3_fault_ready_delay, &fault_ready_zero,
                       sizeof(fault_ready_zero));
#endif
#if (L3_FAULT_INJECT_ACK_DELAY == true)
    unsigned int fault_ack_zero = 0;
    int fault_ack_pending_zero = 0;
    cudaMemcpyToSymbol(g_l3_fault_ack_delay, &fault_ack_zero,
                       sizeof(fault_ack_zero));
    cudaMemcpyToSymbol(g_l3_fault_ack_pending, &fault_ack_pending_zero,
                       sizeof(fault_ack_pending_zero));
#endif
#if (L3_EVENT_RING == true)
    unsigned int l3_event_zero = 0;
    cudaMemcpyToSymbol(g_l3_event_head, &l3_event_zero,
                       sizeof(l3_event_zero));
    cudaMemcpyToSymbol(g_l3_event_overflow, &l3_event_zero,
                       sizeof(l3_event_zero));
#endif
#if (L3_LIVE_SNAPSHOT == true)
    l3_live_device_state l3_live_zero = {};
    cudaMemcpyToSymbol(g_l3_live_state, &l3_live_zero,
                       sizeof(l3_live_zero));
#endif
#if (BULK_ROUND == true)
#if (SEED_BARRIER == false)
    unsigned long long bulk_signal_zero = 0;
    cudaMemcpyToSymbol(g_bulk_mark_signal, &bulk_signal_zero, sizeof(bulk_signal_zero));
#if (L3_MAPPING_DIAG == true)
    l3_mapping_metrics mapping_zero = {};
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_mapping_metrics, &mapping_zero, sizeof(mapping_zero))
                        == cudaSuccess, "mapping reset");
#endif
#if (L3_DIAGNOSTICS == true)
    l3_diagnostics diag_zero = {};
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_diagnostics, &diag_zero, sizeof(diag_zero))
                        == cudaSuccess, "diagnostic reset");
#endif
#if (DQ_SPARSE_DIAG == true)
    std::vector<dq_sparse_metrics> sparse_zero(DQ_SPARSE_SLOTS);
    g_benchmark.require(cudaMemcpyToSymbol(g_dq_sparse, sparse_zero.data(),
                        sizeof(dq_sparse_metrics) * DQ_SPARSE_SLOTS) == cudaSuccess,
                        "sparse diagnostic reset");
#if (DQ_SPARSE_PHASE == true)
    std::vector<dq_phase_metrics> phase_zero(DQ_SPARSE_SLOTS);
    g_benchmark.require(cudaMemcpyToSymbol(g_dq_phase, phase_zero.data(),
                        sizeof(dq_phase_metrics) * DQ_SPARSE_SLOTS) == cudaSuccess,
                        "phase diagnostic reset");
#endif
#endif
#if (L3_WAIT_DIAG == true)
    std::vector<l3_wait_sample> wait_zero(258);
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_wait,wait_zero.data(),sizeof(l3_wait_sample)*258)==cudaSuccess,"wait diagnostic reset");
    unsigned long long reason_zero[18]={};
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_wait_reason,reason_zero,sizeof(reason_zero))==cudaSuccess,"wait reason reset");
#endif
#if (L3_WORK_DIAG == true)
    std::vector<l3_work_metrics> work_zero(L3_WORK_SLOTS);
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_work_metrics, work_zero.data(),
                        sizeof(l3_work_metrics) * L3_WORK_SLOTS) == cudaSuccess,
                        "work diagnostic reset");
#if (L3_SOURCE_EXPAND_DIAG == true)
    l3_source_expand_metrics source_expand_zero = {};
    g_benchmark.require(cudaMemset(gctx[gpu_id].source_expand_counts, 0,
                        sizeof(unsigned) * (v_local + 1)) == cudaSuccess,
                        "source expansion counters reset");
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_source_expand_metrics,
                        &source_expand_zero,
                        sizeof(source_expand_zero)) == cudaSuccess,
                        "source expansion diagnostic reset");
#endif
#endif
#if (L3_PROGRESS_DIAG == true)
    l3_progress_metrics progress_zero = {};
    g_benchmark.require(cudaMemcpyToSymbol(g_l3_progress, &progress_zero, sizeof(progress_zero))
                        == cudaSuccess, "progress reset");
#endif
    int *bulk_idle_ptr = gctx[gpu_id].local_idle;
    cudaMemcpyToSymbol(g_bulk_local_idle, &bulk_idle_ptr, sizeof(bulk_idle_ptr));
#endif
#endif
#if (GHOST_DEPTH > 0)
    // 方案 C: 重置 ghost 距离（DIST_MAX）+ 改进信号（0）
    if (gctx[gpu_id].ghost_node_data)
    {
        g_benchmark.require(l3_fill_values(gctx[gpu_id].ghost_node_data,
                            size_t(gctx[gpu_id].ghost_num), init_max) == cudaSuccess,
                            "ghost distance reset launch");
        cudaMemset(gctx[gpu_id].ghost_mark, 0, sizeof(unsigned) * ((gctx[gpu_id].ghost_num + 31) / 32));
    }
#endif

    g_benchmark.require(cudaDeviceSynchronize() == cudaSuccess, "query distance reset sync");

    return 0;
}

// 收集本卡归属区间的距离到 host 数组（全局 0-based 索引）
int sssp_cp_data(int gpu_id, VALUE_TYPE *node_data_h)
{
    cudaError_t status = cudaSetDevice(gpu_id);
    g_benchmark.require(status == cudaSuccess, "collection device");
    status = cudaMemcpy(node_data_h + gctx[gpu_id].v_begin, gctx[gpu_id].node_data + 1,
               sizeof(VALUE_TYPE) * gctx[gpu_id].v_local, cudaMemcpyDeviceToHost);
    g_benchmark.require(status == cudaSuccess, "distance collection");
    return status == cudaSuccess ? 0 : -1;
}

// CP2b: 显式释放本卡由 l3_host_context 拥有的资源。图 CSR、node_data、seed、
// ghost 与持久 MLMQ workspace 不属于 L3 owner，继续由原有宿主管理。
int sssp_release_l3(int gpu_id)
{
    if (gpu_id < 0 || gpu_id >= MAX_GPU)
        return -1;
#if (L3_REGION_RELAX == true)
    l3_region_release(gpu_id);
#endif
#if (L3_CHAIN_PARTITION == true)
    l3_chain_partition_release(gpu_id);
#endif
#if (L3_BOUNDARY_INDEX == true)
    l3_boundary_release(gpu_id);
#endif
#if (L3_RECOVERY_DOMAIN_DIAG == true)
    l3_domain_release(gpu_id);
#endif
    gctx[gpu_id].release();
    return 0;
}

#if (L3_L2_FINAL_COUNTS == true)
template <typename Q>
static void report_final_l2_counts(const Q &, int gpu)
{
    printf("L2_FINAL_UNSUPPORTED gpu=%d\n", gpu);
}
template <typename E>
static void report_final_l2_counts(const l2_delta_queue<E> &q, int gpu)
{
    std::vector<int> reads(q.bucketNum), writes(q.bucketNum);
    int completed = 0, guarded_writes = 0, overflow_detected = 0;
    g_benchmark.require(cudaMemcpy(reads.data(), q.bucket_read_done,
        sizeof(int)*q.bucketNum, cudaMemcpyDeviceToHost)==cudaSuccess, "L2 final reads");
    g_benchmark.require(cudaMemcpy(writes.data(), q.write_reserve,
        sizeof(int)*q.bucketNum, cudaMemcpyDeviceToHost)==cudaSuccess, "L2 final writes");
    g_benchmark.require(cudaMemcpy(&completed, q.read_done,
        sizeof(int), cudaMemcpyDeviceToHost)==cudaSuccess, "L2 final completion");
    g_benchmark.require(cudaMemcpy(&guarded_writes, q.debug_write_done,
        sizeof(int), cudaMemcpyDeviceToHost)==cudaSuccess, "L2 guarded writes");
    g_benchmark.require(cudaMemcpy(&overflow_detected, q.counter_overflow,
        sizeof(int), cudaMemcpyDeviceToHost)==cudaSuccess, "L2 overflow flag");
#if (DQ_COUNTER_OVERFLOW_GUARD == true)
    constexpr int overflow_guard = 1;
#else
    constexpr int overflow_guard = 0;
#endif
    g_benchmark.require(overflow_guard==1,
                        "L2 final evidence requires overflow guard");
    g_benchmark.require(overflow_detected==0,
                        "L2 cumulative counter overflow detected");
    long long read_total=0, write_total=0;
    const long long total_capacity =
        static_cast<long long>(q.bucketNum) * q.total_size;
    g_benchmark.require(total_capacity<=INT_MAX,
                        "L2 aggregate capacity exceeds counter range");
    int max_bucket_writes=0;
    for (int b=0; b<q.bucketNum; ++b) {
        g_benchmark.require(reads[b]>=0 && writes[b]>=0 && reads[b]==writes[b],
                            "L2 final per-bucket drain");
        g_benchmark.require(writes[b]<=q.total_size,
                            "L2 final per-bucket no-wrap bound");
        read_total+=reads[b]; write_total+=writes[b];
        max_bucket_writes=std::max(max_bucket_writes,writes[b]);
    }
    g_benchmark.require(write_total<=INT_MAX, "L2 final total counter range");
    g_benchmark.require(guarded_writes==write_total,
                        "L2 guarded write conservation");
    g_benchmark.require(completed==write_total, "L2 final completion conservation");
    printf("L2_FINAL gpu=%d buckets=%d reads=%lld writes=%lld completed=%d "
           "guarded_writes=%d max_bucket_writes=%d per_bucket_capacity=%d "
           "total_capacity=%lld counter_bits=%zu overflow_guard=%d "
           "overflow_detected=%d no_wrap=1\n",
           gpu,q.bucketNum,read_total,write_total,completed,guarded_writes,
           max_bucket_writes,q.total_size,total_capacity,8*sizeof(int),
           overflow_guard,overflow_detected);
}
#endif

template <typename QUEUE_TYPE>
void kernel_adaptive(int gpu_id, int src, mlmq_setup setup)
{
    cudaSetDevice(gpu_id);
    const int rx_express_enabled = g_rx_express_enabled;
#if (L3_RX_L2_PULL == true)
    const int rx_l2_pull_enabled = g_rx_l2_pull_enabled;
#endif
    int v_local = gctx[gpu_id].v_local;

    NODE_TYPE init_limits = node_struct(0, DIST_MAX);

    query_workspace<QUEUE_TYPE> &workspace = g_query_workspace<QUEUE_TYPE>[gpu_id];
    if (!workspace.ensure(gpu_id, v_local, init_limits, setup))
    {
        g_benchmark.require(false, "workspace allocation");
        return;
    }
    QUEUE_TYPE &mlmq = workspace.mlmq;

    work_count_type *global_work_count = nullptr;
    int *global_comp_count = nullptr;
    int *profile = nullptr;
    unsigned int *hist = nullptr;   // B1-2a
#if (WORK_COUNT == true)
    global_work_count = workspace.global_work_count;
    global_comp_count = workspace.global_comp_count;
    profile = workspace.profile;
    hist = workspace.hist;
#endif

#if (MANAGE_PROFILE == true)
    // V0.5 profile buffer（manage kernel 各环节 clock() 累积，13 项）
    unsigned long long *mgmt_profile = workspace.mgmt_profile;
#endif

    cudaDeviceSynchronize();

    int qshm_size = (mlmq.get_shm_size() + 1023) / 1024 * 1024;
#if (node_size < WARP_SIZE)
    int nshm_size = WARP_SIZE * WARP_NUM_PER_BLOCK * 4 * sizeof(NODE_TYPE);
#else
    int nshm_size = node_size * WARP_NUM_PER_BLOCK * 4 * sizeof(NODE_TYPE);
#endif
    int rshm_size = LARGEV * WARP_SIZE * WARP_NUM_PER_BLOCK * sizeof(int);

    int REPEAT_TIME = 1; // 步骤 C: 跨卡 re_init 同步留后续，先 REPEAT=1
    for (int repeat = 0; repeat < REPEAT_TIME; repeat++)
    {
        sssp_re_init(gpu_id);
        mlmq.reinit_host(GPU_MEMORY, init_limits, setup);
#if (WORK_COUNT == true)
        // workspace 持久化后，所有 query 级计数仍必须逐源清零；只复用
        // allocation，不改变原有诊断语义。
        cudaMemset(global_work_count, 0, sizeof(work_count_type));
        cudaMemset(global_comp_count, 0, sizeof(int));
        cudaMemset(profile, 0, 2 * sizeof(int));
        cudaMemset(hist, 0, (v_local + 1) * sizeof(unsigned int));
        cudaMemcpyToSymbol(g_hist, &hist, sizeof(hist));
#endif
#if (MANAGE_PROFILE == true)
    cudaMemset(mgmt_profile, 0, sizeof(unsigned long long) * 13);
#endif
#if (QUERY_WORKSPACE_DIAG == true)
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        if (free_bytes < workspace.min_free_bytes)
            workspace.min_free_bytes = free_bytes;
        if (free_bytes > workspace.max_free_bytes)
            workspace.max_free_bytes = free_bytes;
        workspace.reset_count++;
        printf("QUERY_REUSE gpu%d src=%d reset=%llu free=%.2fMiB min=%.2fMiB max=%.2fMiB\n",
               gpu_id, src, workspace.reset_count,
               free_bytes / (1024.0 * 1024.0),
               workspace.min_free_bytes / (1024.0 * 1024.0),
               workspace.max_free_bytes / (1024.0 * 1024.0));
#endif
        // B1-1: 清零本地/远程改进计数（device 全局符号）
        work_count_type zero_cnt = 0;
        cudaMemcpyToSymbol(g_local_work, &zero_cnt, sizeof(work_count_type));
        cudaMemcpyToSymbol(g_remote_work, &zero_cnt, sizeof(work_count_type));
        // E4: 清零 w17 下钻计数
        unsigned long long zero_u = 0;
        timeline_clock_type zero_t = 0;
        cudaMemcpyToSymbol(g_h2_hit, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_w_scan, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_w_hit, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_remote_effective, &zero_u, sizeof(zero_u));
#if (DQ_CLAMP_DIAG == true)
        cudaMemcpyToSymbol(g_dq_clamp, &zero_u, sizeof(zero_u));
#endif
#if (DQ_QUEUE_DIAG == true)
        dq_queue_diag_metrics zero_dq_queue = {};
        cudaMemcpyToSymbol(g_dq_queue_diag, &zero_dq_queue,
                           sizeof(zero_dq_queue));
#endif
        cudaMemcpyToSymbol(g_seed_ready_fired, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_inj_gate_skip, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_bs_gate_skip, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_t_phase2, &zero_t, sizeof(zero_t));
        cudaMemcpyToSymbol(g_t_seed_ready, &zero_t, sizeof(zero_t));
        cudaMemcpyToSymbol(g_t_sr_seen, &zero_t, sizeof(zero_t));
        cudaMemcpyToSymbol(g_t_inject_done, &zero_t, sizeof(zero_t));
        cudaMemcpyToSymbol(g_t_idle, &zero_t, sizeof(zero_t));
        cudaMemcpyToSymbol(g_t_term, &zero_t, sizeof(zero_t));
        cudaMemcpyToSymbol(g_t_peer_idle_seen, &zero_t, sizeof(zero_t));
        cudaMemcpyToSymbol(g_t_sr_w0, &zero_t, sizeof(zero_t));
#if (TIMELINE64 == true)
        cudaMemcpyToSymbol(g_t_idle_last_set, &zero_t, sizeof(zero_t));
        cudaMemcpyToSymbol(g_t_idle_last_reset, &zero_t, sizeof(zero_t));
        cudaMemcpyToSymbol(g_idle_set_count, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_idle_reset_count, &zero_u, sizeof(zero_u));
        unsigned long long zero_idle_reason[16] = {0};
        cudaMemcpyToSymbol(g_idle_reset_reason, zero_idle_reason, sizeof(zero_idle_reason));
        cudaMemcpyToSymbol(g_term_mark_actual, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_term_mark_stale, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_term_mark_sampled, &zero_u, sizeof(zero_u));
#endif
        cudaMemcpyToSymbol(g_confirm_exec, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_inject_exec, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_inject_clks, &zero_u, sizeof(zero_u));
        unsigned int zero_inject=0;
        g_benchmark.require(cudaMemcpyToSymbol(g_inject_scan_clks,&zero_inject,sizeof(zero_inject))==cudaSuccess,"inject scan counter reset");
        g_benchmark.require(cudaMemcpyToSymbol(g_inject_wt_clks,&zero_inject,sizeof(zero_inject))==cudaSuccess,"inject write counter reset");
#if (GLOBAL_ROUND_STATS == true)
        cudaMemcpyToSymbol(g_global_round_count, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_global_round_sent, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_global_round_recv, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_global_round_improved, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_global_round_frontier_append, &zero_u, sizeof(zero_u));
#endif
#if (GLOBAL_ROUND_PROFILE == true)
        cudaMemcpyToSymbol(g_gr_local_empty, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_quiesce, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_pack, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_inbox_wait, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_apply, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_release, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_local_empty_max, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_quiesce_max, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_pack_max, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_inbox_wait_max, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_apply_max, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_release_max, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_gr_profile_rounds, &zero_u, sizeof(zero_u));
#endif
#if (GLOBAL_ROUND_DIRECT_TRACE == true)
        int trace_min = 0x7fffffff;
        cudaMemcpyToSymbol(g_direct_relax_193, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_direct_relax_362, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_direct_cand_193, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_direct_cand_362, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_direct_tx_193, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_direct_tx_362, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_direct_inj_193, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_direct_inj_362, &zero_u, sizeof(zero_u));
        cudaMemcpyToSymbol(g_direct_relax_193_min, &trace_min, sizeof(trace_min));
        cudaMemcpyToSymbol(g_direct_relax_362_min, &trace_min, sizeof(trace_min));
        cudaMemcpyToSymbol(g_direct_tx_193_min, &trace_min, sizeof(trace_min));
        cudaMemcpyToSymbol(g_direct_tx_362_min, &trace_min, sizeof(trace_min));
        cudaMemcpyToSymbol(g_direct_inj_193_min, &trace_min, sizeof(trace_min));
        cudaMemcpyToSymbol(g_direct_inj_362_min, &trace_min, sizeof(trace_min));
#endif
        float elapsedTime = 0.0;

        int total_shm_size = (qshm_size + nshm_size + rshm_size + 1023) / 1024 * 1024;
        cudaFuncSetAttribute(work_block_kernel<QUEUE_TYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, total_shm_size);
        cudaFuncSetAttribute(manage_block_kernel<QUEUE_TYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, total_shm_size);

        struct cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, gpu_id);

        cudaDeviceSynchronize();

        if (!g_multi_source_quiet)
            printf("gpu%d shm_size %d qshm_size %d nshm_size %d rshm_size %d\n", gpu_id, total_shm_size, qshm_size, nshm_size, rshm_size);

        cudaEventRecord(workspace.start, 0);

        int work_block_num = bench_env_int("MLMQ_WORK_BLOCKS",
            dev_prop.multiProcessorCount - 1, 1, dev_prop.multiProcessorCount - 1);
#if (L0_SOURCE_SNAPSHOT == true)
        printf("L0_SOURCE_SNAPSHOT_CONFIG gpu=%d enabled=1\n", gpu_id);
#endif
#if (L0_DIRECT_SMALL == true)
        printf("L0_DIRECT_SMALL_CONFIG gpu=%d enabled=1\n", gpu_id);
#endif
#if (L3_LOCAL_YIELD_BATCHES > 0)
        printf("L3_LOCAL_YIELD_CONFIG gpu=%d budget=%d diag=%d\n",
               gpu_id, L3_LOCAL_YIELD_BATCHES, int(L3_LOCAL_YIELD_DIAG));
#endif
        if (g_benchmark.enabled)
            printf("L3_CONFIG gpu=%d work_blocks=%d delta=%d queue_type=%d window_mode=%d window_min=%llu window_max=%llu idle_backoff=%d worker_recovery=%d term_wait_ack=%d rx_priority_bootstrap=%d rx_express=%d rx_express_enabled=%d rx_express_slots=%d rx_express_batch=%d rx_l2_pull=%d rx_l2_pull_claim=%d rx_l2_pull_enabled=%d\n",
                   gpu_id, work_block_num, setup.s_l2_delta, int(setup.type),
                   int(L3_WINDOW_MODE),
                   static_cast<unsigned long long>(L3_WINDOW_MIN_CYCLES),
                   static_cast<unsigned long long>(L3_WINDOW_MAX_CYCLES),
                   int(L3_IDLE_BACKOFF),
                   int(L3_WORKER_RECOVERY), int(L3_TERM_WAIT_ACK),
                   int(L3_RX_PRIORITY_BOOTSTRAP), int(L3_RX_EXPRESS),
                   rx_express_enabled,
                   int(L3_RX_EXPRESS_SLOTS), int(L3_RX_EXPRESS_BATCH),
                   int(L3_RX_L2_PULL),
                   int(L3_RX_L2_PULL_SINGLE_CLAIM),
#if (L3_RX_L2_PULL == true)
                   rx_l2_pull_enabled
#else
                   0
#endif
                   );
#if (L3_TILE_LOAN == true)
    printf("L3_TILE_LOAN_CONFIG gpu=%d enabled=%d owner_commit=%d max_seeds=%d result_cap=%d poll_interval=%d read_retries=%d\n",
                   gpu_id, g_l3_tile_loan_enabled, int(L3_OWNER_COMMIT),
                   L3_TILE_LOAN_MAX_SEEDS,
                   L3_TILE_LOAN_RESULT_CAP, L3_TILE_LOAN_POLL_INTERVAL,
                   L3_TILE_LOAN_READ_RETRIES);
#endif

        // 注意：manage kernel 必须先 launch（v1 教训：work 先 launch 占满 GPU，manage 无法调度）
        // warp 分配: 0=终止+backstop+注入slice0, 1..manage_warp_num()=L2 manager,
        //   manage_warp_num()+1=L3 flush, 其后 INJECT_WARP_NUM 个=注入 warp
        int manage_thread_num = WARP_SIZE * (1 + mlmq.manage_warp_num() + 1 + INJECT_WARP_NUM);
        const int active_work_warps = work_block_num * WARP_NUM_PER_BLOCK;
        g_benchmark.require(active_work_warps <= gctx[gpu_id].l3_term_ack_capacity,
            "worker geometry ACK capacity");
        if (g_benchmark.enabled)
            printf("L3_WORKER_ACK gpu=%d active_slots=%d capacity=%d work_blocks=%d warps_per_block=%d\n",
                   gpu_id, active_work_warps, gctx[gpu_id].l3_term_ack_capacity,
                   work_block_num, WARP_NUM_PER_BLOCK);
#if (MLMQ_WORKER_THREADS == 384 || MLMQ_WORKER_THREADS == 320)
        int resident_blocks = 0;
        g_benchmark.require(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &resident_blocks, work_block_kernel<QUEUE_TYPE>,
            ALIGN_THREAD_PER_BLOCK + WARP_SIZE, total_shm_size) == cudaSuccess,
            "worker geometry occupancy query");
        g_benchmark.require(resident_blocks == 1, "worker geometry requires one resident block per SM");
        printf("L3_WORKER_GEOMETRY gpu=%d threads=%d warps=%d blocks=%d resident=%d ack_slots=%d shm=%d\n",
            gpu_id, ALIGN_THREAD_PER_BLOCK + WARP_SIZE, WARP_NUM_PER_BLOCK,
            work_block_num, resident_blocks, gctx[gpu_id].l3_term_ack_capacity, total_shm_size);
#endif
        l3_channel_view l3_channel = gctx[gpu_id].make_channel_view();

        if (g_benchmark.enabled) {
            g_benchmark.require(cudaGetLastError() == cudaSuccess, "pre-launch CUDA error");
            g_benchmark.require(cudaDeviceSynchronize() == cudaSuccess, "pre-launch sync");
            g_benchmark.begin();
        }

#if (L3_TIMING_DIAG == true)
        auto &host_timing = g_l3_host_timing[gpu_id];
        host_timing = {};
        host_timing.barrier_return = mlmq_bench_ms();
        host_timing.manage_start = workspace.timing_manage_start;
        host_timing.manage_end = workspace.timing_manage_end;
        host_timing.work_start = workspace.timing_work_start;
        host_timing.work_end = workspace.timing_work_end;
        g_benchmark.require(cudaEventRecord(host_timing.manage_start, workspace.manage_stream) == cudaSuccess, "timing manager start record");
        host_timing.manage_before = mlmq_bench_ms();
#endif

        manage_block_kernel<QUEUE_TYPE><<<1, manage_thread_num, total_shm_size, workspace.manage_stream>>>(gctx[gpu_id].m, gctx[gpu_id].nnz,
            gctx[gpu_id].RowPtr, gctx[gpu_id].ColIdx, gctx[gpu_id].edge_data, src, mlmq, setup,
            qshm_size, nshm_size, rshm_size, gctx[gpu_id].global_exit,
            gctx[gpu_id].v_begin, gctx[gpu_id].v_end, gctx[gpu_id].v_local,
            gctx[gpu_id].node_data, gctx[gpu_id].dirty_bitmap, gctx[gpu_id].dirty_hint, gctx[gpu_id].last_processed,
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
            gctx[gpu_id].async_rx_ready_bitmap, gctx[gpu_id].async_rx_ready_hint,
#endif
            gctx[gpu_id].local_idle, l3_channel,
            gctx[gpu_id].phase, gctx[gpu_id].seed_ready, gctx[gpu_id].peer_seed_ready, gctx[gpu_id].seed_inject_done,
            gctx[gpu_id].seed_list, gctx[gpu_id].seed_list_cnt, gctx[gpu_id].peer_seed_list, gctx[gpu_id].peer_seed_list_cnt,
#if (GLOBAL_ROUND_ASYNC == true)
            gctx[gpu_id].cand_bank[0], gctx[gpu_id].cand_bank[1], gctx[gpu_id].cand_ctl,
#endif
#if (GHOST_DEPTH > 0)
            gctx[gpu_id].ghost_num, gctx[gpu_id].ghost_row_start, gctx[gpu_id].ghost_col, gctx[gpu_id].ghost_edge_data,
            gctx[gpu_id].ghost_node_data, gctx[gpu_id].ghost_id_to_idx, gctx[gpu_id].ghost_mark,
#endif
            global_work_count, global_comp_count, g_n_gpu,
#if (L3_RX_EXPRESS == true)
            rx_express_enabled,
#endif
#if (BULK_ROUND == true)
            active_work_warps,
#endif
#if (MANAGE_PROFILE == true)
            mgmt_profile);
#else
            NULL);
#endif

#if (L3_TIMING_DIAG == true)
        host_timing.manage_after = mlmq_bench_ms();
#endif
        cudaError_t m_err = cudaGetLastError();
        if (m_err != cudaSuccess)
            fprintf(stderr, "gpu%d manage launch error: %s (%d)\n", gpu_id, cudaGetErrorString(m_err), (int)m_err);
        g_benchmark.require(m_err == cudaSuccess, "manager launch");

#if (L3_TIMING_DIAG == true)
        g_benchmark.require(cudaEventRecord(host_timing.manage_end, workspace.manage_stream) == cudaSuccess, "timing manager end record");
        g_benchmark.require(cudaEventRecord(host_timing.work_start, workspace.work_stream) == cudaSuccess, "timing work start record");
        host_timing.work_before = mlmq_bench_ms();
#endif
        work_block_kernel<QUEUE_TYPE><<<work_block_num, ALIGN_THREAD_PER_BLOCK + WARP_SIZE, total_shm_size, workspace.work_stream>>>(gctx[gpu_id].m,
            gctx[gpu_id].nnz, gctx[gpu_id].RowPtr, gctx[gpu_id].ColIdx, gctx[gpu_id].edge_data, gctx[gpu_id].node_data,
            src, mlmq, setup, qshm_size, nshm_size, gctx[gpu_id].global_exit, global_work_count, global_comp_count, profile,
            gctx[gpu_id].v_begin, gctx[gpu_id].v_end, gctx[gpu_id].v_local,
            gctx[gpu_id].remote_cand, gctx[gpu_id].remote_mark, gctx[gpu_id].mark_hint, gctx[gpu_id].mark_hint2, gctx[gpu_id].peer_cache, gctx[gpu_id].peer_v_begin, gctx[gpu_id].peer_v_local,
#if (GLOBAL_ROUND_ASYNC == true)
            gctx[gpu_id].cand_bank[0], gctx[gpu_id].cand_bank[1], gctx[gpu_id].cand_ctl,
#endif
#if (GHOST_DEPTH > 0)
            gctx[gpu_id].ghost_id_to_idx, gctx[gpu_id].ghost_node_data, gctx[gpu_id].ghost_mark,
#endif
            gctx[gpu_id].last_processed
#if (L3_WORK_DIAG == true)
            , gctx[gpu_id].source_expand_counts
#endif
            , g_n_gpu
#if (L3_TILE_LOAN == true)
            , l3_channel, g_l3_tile_loan_enabled
#endif
#if (L3_RX_EXPRESS == true)
            , gctx[gpu_id].rx_express, rx_express_enabled
#endif
#if (L3_RX_L2_PULL == true)
            , gctx[gpu_id].rx_commit_seq, rx_l2_pull_enabled
#if (L3_RX_L2_PULL_SINGLE_CLAIM == true)
            , gctx[gpu_id].rx_pull_claim_seq
#endif
#if (L3_RX_L2_PULL_DIAG == true)
            , gctx[gpu_id].rx_l2_pull_stats
#endif
#endif
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true)
            , gctx[gpu_id].async_rx_ready_bitmap,
            gctx[gpu_id].async_rx_ready_hint
#endif
#if (BULK_ROUND == true)
            , gctx[gpu_id].bulk_quiesce_req, gctx[gpu_id].bulk_quiesce_ack,
            gctx[gpu_id].l3_term_req, gctx[gpu_id].l3_term_ack_slots
#if (BULK_FRONTIER_ENABLED == true)
            , gctx[gpu_id].bulk_frontier, gctx[gpu_id].bulk_frontier_head, gctx[gpu_id].bulk_frontier_tail
#endif
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
            , gctx[gpu_id].bulk_inbox, gctx[gpu_id].bulk_inbox_count,
            gctx[gpu_id].bulk_inbox_epoch, gctx[gpu_id].bulk_inbox_state,
            gctx[gpu_id].bulk_inbox_generation, gctx[gpu_id].bulk_inbox_read_head,
            gctx[gpu_id].bulk_inbox_inflight, gctx[gpu_id].bulk_inbox_active_slot
#endif
#if (SEED_BARRIER == true)
            , gctx[gpu_id].seed_inject_done
#endif
            );

#if (L3_TIMING_DIAG == true)
        host_timing.work_after = mlmq_bench_ms();
#endif
        cudaError_t la_err = cudaGetLastError();
        if (la_err != cudaSuccess)
            fprintf(stderr, "gpu%d kernel launch error: %s (%d)\n", gpu_id, cudaGetErrorString(la_err), (int)la_err);
        g_benchmark.require(la_err == cudaSuccess, "worker launch");

#if (L3_LIVE_SNAPSHOT == true)
        std::atomic<bool> l3_live_snapshot_stop(false);
        std::thread l3_live_snapshot_thread;
        bool l3_live_snapshot_started = false;
        // An explicit diagnostic build must remain observable in BENCH mode,
        // which suppresses normal per-query output via g_multi_source_quiet.
        if (!g_multi_source_quiet || g_benchmark.enabled)
        {
            l3_live_snapshot_thread = std::thread(
                [&l3_live_snapshot_stop, gpu_id]()
                {
                    l3_live_snapshot_watchdog(
                        gpu_id, gctx[gpu_id], l3_live_snapshot_stop);
                });
            l3_live_snapshot_started = true;
        }
#endif
#if (L3_TIMING_DIAG == true)
        g_benchmark.require(cudaEventRecord(host_timing.work_end, workspace.work_stream) == cudaSuccess, "timing work end record");
        host_timing.sync_before = mlmq_bench_ms();
#endif
        cudaError_t solve_status = cudaDeviceSynchronize();
#if (L3_TIMING_DIAG == true)
        host_timing.sync_after = mlmq_bench_ms();
#endif
        g_benchmark.finish();
#if (L3_TIMING_DIAG == true)
        host_timing.finish_after = mlmq_bench_ms();
#endif
        if(solve_status!=cudaSuccess)
            fprintf(stderr,"gpu%d solve sync error: %s (%d)\n",gpu_id,cudaGetErrorString(solve_status),int(solve_status));
        g_benchmark.require(solve_status == cudaSuccess, "solve sync");
#if (L3_L2_FINAL_COUNTS == true)
        // Read existing cumulative counters only after synchronize + finish.
        // No instrumentation is added to worker or manager kernels.
        report_final_l2_counts(mlmq.q2, gpu_id);
#endif
#if (L3_RX_EXPRESS == true && L3_RX_EXPRESS_DIAG == true)
        {
            unsigned long long express_head = 0, express_tail = 0;
            unsigned long long express_stats[L3_RX_EXPRESS_STAT_COUNT] = {};
            g_benchmark.require(
                cudaMemcpy(&express_head, gctx[gpu_id].rx_express.head,
                           sizeof(express_head), cudaMemcpyDeviceToHost)
                    == cudaSuccess,
                "express head read");
            g_benchmark.require(
                cudaMemcpy(&express_tail, gctx[gpu_id].rx_express.tail,
                           sizeof(express_tail), cudaMemcpyDeviceToHost)
                    == cudaSuccess,
                "express tail read");
            g_benchmark.require(
                cudaMemcpy(express_stats, gctx[gpu_id].rx_express.stats,
                           sizeof(express_stats), cudaMemcpyDeviceToHost)
                    == cudaSuccess,
                "express stats read");
            printf("L3_RX_EXPRESS gpu=%d enqueue_records=%llu dequeue_records=%llu "
                   "complete_records=%llu fallback_records=%llu "
                   "enqueue_batches=%llu dequeue_batches=%llu normal_batches=%llu "
                   "high_water=%llu head=%llu tail=%llu held=%llu\n",
                   gpu_id,
                   express_stats[L3_RX_EXPRESS_ENQUEUE_RECORDS],
                   express_stats[L3_RX_EXPRESS_DEQUEUE_RECORDS],
                   express_stats[L3_RX_EXPRESS_COMPLETE_RECORDS],
                   express_stats[L3_RX_EXPRESS_FALLBACK_RECORDS],
                   express_stats[L3_RX_EXPRESS_ENQUEUE_BATCHES],
                   express_stats[L3_RX_EXPRESS_DEQUEUE_BATCHES],
                   express_stats[L3_RX_EXPRESS_NORMAL_BATCHES],
                   express_stats[L3_RX_EXPRESS_HIGH_WATER],
                   express_head, express_tail, express_tail - express_head);
        }
#endif
#if (L3_RX_L2_PULL == true && L3_RX_L2_PULL_DIAG == true)
        {
            unsigned rx_commit_seq = 0;
            unsigned long long pull_stats[L3_RX_L2_PULL_STATS_COUNT] = {};
            g_benchmark.require(
                cudaMemcpy(&rx_commit_seq, gctx[gpu_id].rx_commit_seq,
                           sizeof(rx_commit_seq), cudaMemcpyDeviceToHost)
                    == cudaSuccess,
                "rx l2 pull sequence read");
            g_benchmark.require(
                cudaMemcpy(pull_stats, gctx[gpu_id].rx_l2_pull_stats,
                           sizeof(pull_stats), cudaMemcpyDeviceToHost)
                    == cudaSuccess,
                "rx l2 pull stats read");
            printf("L3_RX_L2_PULL gpu=%d rx_events=%llu hint_seen=%llu "
                   "pull_attempt=%llu pull_records=%llu empty_pull=%llu "
                   "ordinary_after_pull=%llu old_inflight=%llu l2_available=%llu "
                   "old_inflight_with_record=%llu l2_available_with_record=%llu "
                   "claim_lost=%llu seq=%u\n",
                   gpu_id,
                   pull_stats[L3_RX_L2_PULL_RX_EVENTS],
                   pull_stats[L3_RX_L2_PULL_HINT_SEEN],
                   pull_stats[L3_RX_L2_PULL_PULL_ATTEMPT],
                   pull_stats[L3_RX_L2_PULL_PULL_RECORDS],
                   pull_stats[L3_RX_L2_PULL_EMPTY_PULL],
                   pull_stats[L3_RX_L2_PULL_ORDINARY_AFTER_PULL],
                   pull_stats[L3_RX_L2_PULL_OLD_INFLIGHT],
                   pull_stats[L3_RX_L2_PULL_L2_AVAILABLE],
                   pull_stats[L3_RX_L2_PULL_OLD_INFLIGHT_WITH_RECORD],
                   pull_stats[L3_RX_L2_PULL_L2_AVAILABLE_WITH_RECORD],
                   pull_stats[L3_RX_L2_PULL_CLAIM_LOST],
                   rx_commit_seq);
        }
#endif
#if (L3_ADMISSION_BUDGET == true)
        int advice_report[4]={};
        g_benchmark.require(cudaMemcpyFromSymbol(advice_report,g_l3_admission,sizeof(advice_report))==cudaSuccess,"admission report");
        printf("L3_ADMISSION gpu=%d deferred_batches=%d\n",gpu_id,advice_report[3]);
#endif
#if (DQ_CLAMP_DIAG == true)
        // Explicit diagnostic request only. Paired timing runs leave it unset.
        const char *dq_report = getenv("MLMQ_DQ_CLAMP_REPORT");
        if (dq_report && dq_report[0] == '1' && dq_report[1] == '\0') {
            unsigned long long dq_clamp = 0;
            g_benchmark.require(cudaMemcpyFromSymbol(&dq_clamp, g_dq_clamp, sizeof(dq_clamp))
                                == cudaSuccess, "dq clamp read");
            printf("DQ_CLAMP gpu=%d count=%llu\n", gpu_id, dq_clamp);
        }
#endif
#if (L3_CHAIN_PARTITION_DIAG == true)
        unsigned long long chain_counts[8]={};
        g_benchmark.require(cudaMemcpyFromSymbol(chain_counts,g_l3_chain_partition_counts,sizeof(chain_counts))==cudaSuccess,"chain counters read");
        g_benchmark.require(chain_counts[0]==chain_counts[5]+chain_counts[6],"chain closure accounting");
        printf("L3_CHAIN_PARTITION gpu=%d closures=%llu materialized=%llu tails=%llu rx_sources=%llu route_reads=%llu unique_segments=%llu repeated_closures=%llu max_segment_visits=%llu\n",gpu_id,chain_counts[0],chain_counts[1],chain_counts[2],chain_counts[3],chain_counts[4],chain_counts[5],chain_counts[6],chain_counts[7]);
#endif
#if (L3_REGION_DIAG == true)
        unsigned long long region_counts[4]={};
        g_benchmark.require(cudaMemcpyFromSymbol(region_counts,g_l3_region_counts,sizeof(region_counts))==cudaSuccess,"region counters read");
        printf("L3_REGION gpu=%d regions=%llu rounds=%llu materialized=%llu external=%llu\n",
               gpu_id,region_counts[0],region_counts[1],region_counts[2],region_counts[3]);
#endif
#if (L3_PROGRESS_DIAG == true)
        l3_progress_metrics progress = {};
        g_benchmark.require(cudaMemcpyFromSymbol(&progress, g_l3_progress, sizeof(progress))
                            == cudaSuccess, "progress read");
#if (L3_IDLE_TOKEN_PROBE == true)
        printf("L3_IDLE_PROBE gpu=%d busy_skips=%llu\n", gpu_id, progress.idle_probe_skips);
#endif
#if (L3_ACK_SCAN == true)
        printf("L3_PERIODIC_PHASE gpu=%d active=%llu uncertified=%llu certified=%llu\n",
               gpu_id,progress.periodic_active,progress.periodic_uncertified,progress.periodic_certified);
        printf("L3_PERIODIC_DELEGATED gpu=%d count=%llu\n",gpu_id,progress.periodic_delegated);
#if (L3_ACK_SCAN_FAULT == true)
        printf("L3_ACK_FAULT gpu=%d dropped=%llu delayed=%llu recovered_scans=%llu\n",
               gpu_id,progress.dropped_notification,progress.recovery_delays,progress.backstop_positive);
#endif
#endif
        printf("L3_PROGRESS gpu=%d span=%llu first_rx=%llu last_rx=%llu first_winner=%llu last_commit=%llu "
               "rx_cycles=%llu batches=%llu items=%llu winners=%llu backstop_cycles=%llu backstop_calls=%llu "
               "recovery_mode=%d recovery_epochs=%llu recovery_helpers=%llu recovery_done=%llu recovery_delays=%llu "
               "periodic_calls=%llu periodic_vertices=%llu periodic_cycles=%llu "
               "backstop_positive=%llu term_requests=%llu term_cancels=%llu "
               "term_invalid=%llu term_ack_pending=%llu term_peer_cancel=%llu term_inbox_cancel=%llu term_ack_waits=%llu\n",
               gpu_id, progress.end - progress.start,
               progress.batches ? progress.first_receive - progress.start : 0ull,
               progress.batches ? progress.last_receive - progress.start : 0ull,
               progress.winners ? progress.first_winner - progress.start : 0ull,
               progress.winners ? progress.last_commit - progress.start : 0ull,
               progress.receive_cycles, progress.batches, progress.items, progress.winners,
               progress.backstop_cycles, progress.backstop_calls, L3_RECOVERY_MODE,
               progress.recovery_epochs, progress.recovery_helpers, progress.recovery_done, progress.recovery_delays,
               progress.periodic_calls, progress.periodic_vertices, progress.periodic_cycles,
               progress.backstop_positive, progress.term_requests, progress.term_cancels,
               progress.term_invalid, progress.term_ack_pending, progress.term_peer_cancel,
               progress.term_inbox_cancel, progress.term_ack_waits);
#if (L3_RX_LAG_DIAG == true)
        printf("L3_RX_LAG gpu=%d winners=%llu seen_snapshot=%llu below_base=%llu far_ahead=%llu max_lag_buckets=%llu\n",
               gpu_id,progress.winners,progress.rx_seen_snapshot,progress.rx_below_base,
               progress.rx_far_ahead,progress.rx_max_lag_buckets);
#endif
#endif
#if (DQ_SPARSE_DIAG == true)
        std::vector<dq_sparse_metrics> slots(DQ_SPARSE_SLOTS);
        g_benchmark.require(cudaMemcpyFromSymbol(slots.data(), g_dq_sparse,
                            sizeof(dq_sparse_metrics) * DQ_SPARSE_SLOTS) == cudaSuccess,
                            "sparse diagnostic read");
        dq_sparse_metrics total = {};
#if (DQ_SPARSE_PHASE == true)
        std::vector<dq_phase_metrics> phases(DQ_SPARSE_SLOTS);
        g_benchmark.require(cudaMemcpyFromSymbol(phases.data(), g_dq_phase,
                            sizeof(dq_phase_metrics) * DQ_SPARSE_SLOTS) == cudaSuccess,
                            "phase diagnostic read");
        for (int b = 0; b < DQ_SPARSE_SLOTS; ++b) {
            const auto &p = phases[b]; const auto &d = slots[b];
            if (!p.start) continue;
            g_benchmark.require(p.end >= p.start &&
                ((!p.work_calls && !p.first && !p.last) ||
                 (p.work_calls && p.start <= p.first && p.first <= p.last && p.last <= p.end)),
                "same-warp phase order");
            g_benchmark.require(p.first_unfinished <= p.first_empty &&
                p.first_empty <= p.last_empty && p.last_empty <= d.empty &&
                p.first_unfinished <= p.last_unfinished && p.last_unfinished <= d.unfinished,
                "phase prefix order");
            printf("DQ_PHASE gpu=%d block=%d start=%llu first=%llu last=%llu end=%llu work_calls=%llu "
                   "first_empty=%llu first_unfinished=%llu last_empty=%llu last_unfinished=%llu empty=%llu unfinished=%llu\n",
                   gpu_id, b, p.start, p.first, p.last, p.end, p.work_calls,
                   p.first_empty, p.first_unfinished, p.last_empty, p.last_unfinished, d.empty, d.unfinished);
        }
#endif
        int readers = 0;
        for (const auto &d : slots) {
            readers += d.calls != 0;
            g_benchmark.require(d.samples == d.calls / DQ_SPARSE_PERIOD &&
                                d.empty + d.success == d.samples &&
                                d.records >= d.success,
                                "sparse sample accounting");
#define SPARSE_ADD(field) total.field += d.field
            SPARSE_ADD(calls); SPARSE_ADD(samples); SPARSE_ADD(empty);
            SPARSE_ADD(success); SPARSE_ADD(records);
#define SPARSE_EVENT(field) \
            g_benchmark.require(d.field <= d.empty, "sparse event bound"); SPARSE_ADD(field)
            SPARSE_EVENT(unfinished); SPARSE_EVENT(local_lag); SPARSE_EVENT(own_ready);
            SPARSE_EVENT(unpublished_wait); SPARSE_EVENT(speculative_wait);
            SPARSE_EVENT(unreserved_inside); SPARSE_EVENT(unreserved_outside);
#undef SPARSE_EVENT
#undef SPARSE_ADD
        }
        printf("DQ_SPARSE gpu=%d readers=%d period=1024 calls=%llu samples=%llu empty=%llu success=%llu records=%llu "
               "unfinished=%llu local_lag=%llu own_ready=%llu unpublished_wait=%llu speculative_wait=%llu "
               "unreserved_inside=%llu unreserved_outside=%llu\n",
               gpu_id, readers, total.calls, total.samples, total.empty, total.success, total.records,
               total.unfinished, total.local_lag, total.own_ready, total.unpublished_wait,
               total.speculative_wait, total.unreserved_inside, total.unreserved_outside);
#endif
#if (DQ_QUEUE_DIAG == true)
        dq_queue_diag_metrics dq_queue = {};
        g_benchmark.require(cudaMemcpyFromSymbol(&dq_queue, g_dq_queue_diag,
                            sizeof(dq_queue)) == cudaSuccess,
                            "q2 supply diagnostic read");
        printf("DQ_QUEUE gpu=%d read_calls=%llu read_success=%llu "
               "read_empty=%llu records=%llu empty_no_work=%llu "
               "empty_unpublished=%llu empty_published=%llu "
               "manager_calls=%llu manager_pending=%llu\n",
               gpu_id, dq_queue.read_calls, dq_queue.read_success,
               dq_queue.read_empty, dq_queue.records,
               dq_queue.empty_no_work, dq_queue.empty_unpublished,
               dq_queue.empty_published, dq_queue.manager_calls,
               dq_queue.manager_pending);
#endif
#if (L3_WAIT_DIAG == true)
        std::vector<l3_wait_sample> wait_rows(258);
        g_benchmark.require(cudaMemcpyFromSymbol(wait_rows.data(),g_l3_wait,sizeof(l3_wait_sample)*258)==cudaSuccess,"wait diagnostic read");
        unsigned long long reasons[18];
        g_benchmark.require(cudaMemcpyFromSymbol(reasons,g_l3_wait_reason,sizeof(reasons))==cudaSuccess,"wait reason read");
        for(int r=0;r<18;++r) printf("L3_WAIT_REASON gpu=%d reason=%d count=%llu\n",gpu_id,r,reasons[r]);
        for(int slot=0;slot<258;++slot) if(wait_rows[slot].span) {
            const auto &w=wait_rows[slot];
            unsigned long long sum=0;
            for(int p=0;p<6;++p)sum+=w.cycles[p];
            g_benchmark.require(sum==w.span,"wait phase conservation");
            printf("L3_WAIT gpu=%d slot=%d span=%llu last_work=%llu c0=%llu c1=%llu c2=%llu c3=%llu c4=%llu c5=%llu\n",gpu_id,slot,w.span,w.last_work,w.cycles[0],w.cycles[1],w.cycles[2],w.cycles[3],w.cycles[4],w.cycles[5]);
        }
#endif
#if (L3_RECOVERY_DOMAIN_DIAG == true)
        l3_domain_report(gpu_id);
#endif
#if (L3_WORK_DIAG == true)
        std::vector<l3_work_metrics> work_metrics(L3_WORK_SLOTS);
        g_benchmark.require(cudaMemcpyFromSymbol(work_metrics.data(), g_l3_work_metrics,
                            sizeof(l3_work_metrics) * L3_WORK_SLOTS) == cudaSuccess,
                            "work diagnostic read");
        unsigned long long popped = 0, expanded = 0, edges = 0, calls = 0;
        unsigned long long empty_reads = 0, idle_iters = 0, busy_iters = 0;
        unsigned long long owner_destination_edges = 0;
        unsigned long long cross_owner_destination_edges = 0;
        int participants = 0;
        std::vector<double> active_fraction, first_fraction;
        for (const auto &w : work_metrics) {
            if (!w.span) continue;
            ++participants;
            g_benchmark.require(w.expanded <= w.popped && w.active <= w.span
                                && w.first <= w.last && w.last <= w.span,
                                "work diagnostic invariants");
            popped += w.popped; expanded += w.expanded; edges += w.edges; calls += w.calls;
            owner_destination_edges += w.owner_destination_edges;
            cross_owner_destination_edges += w.cross_owner_destination_edges;
            empty_reads += w.empty_reads; idle_iters += w.idle_iters; busy_iters += w.busy_iters;
            active_fraction.push_back(double(w.active) / w.span);
            // No-call warps are counted separately, not interpreted as starting at zero.
            if (w.calls) first_fraction.push_back(double(w.first) / w.span);
        }
        std::sort(active_fraction.begin(), active_fraction.end());
        std::sort(first_fraction.begin(), first_fraction.end());
        auto quantile = [](const std::vector<double> &v, double p) {
            return v.empty() ? -1.0 : v[(size_t)(p * (v.size() - 1))];
        };
#if (L3_WORK_COUNT_ONLY == false)
        g_benchmark.require(owner_destination_edges + cross_owner_destination_edges == edges,
                            "work destination accounting");
#endif
#if (L3_SOURCE_EXPAND_DIAG == true)
        l3_source_expand_metrics source_expand = {};
        g_benchmark.require(cudaMemcpyFromSymbol(&source_expand,
                            g_l3_source_expand_metrics,
                            sizeof(source_expand)) == cudaSuccess,
                            "source expansion diagnostic read");
        g_benchmark.require(source_expand.unique_sources +
                            source_expand.repeated_sources == expanded,
                            "source expansion accounting");
#endif
#if (L3_WORK_COUNT_ONLY == true)
        printf("L3_WORK gpu=%d mode=count_only warps=%d called_warps=%zu popped=%llu expanded=%llu edges=%llu calls=%llu empty_reads=%llu idle_iters=%llu busy_iters=%llu\n",
               gpu_id, participants, first_fraction.size(), popped, expanded, edges, calls,
               empty_reads, idle_iters, busy_iters);
#else
        printf("L3_WORK gpu=%d warps=%d called_warps=%zu popped=%llu expanded=%llu edges=%llu calls=%llu "
               "owner_dst_edges=%llu cross_owner_dst_edges=%llu "
               "empty_reads=%llu idle_iters=%llu busy_iters=%llu "
               "active_p10=%.6f active_p50=%.6f active_p90=%.6f first_p10=%.6f first_p50=%.6f first_p90=%.6f\n",
               gpu_id, participants, first_fraction.size(), popped, expanded, edges, calls,
               owner_destination_edges, cross_owner_destination_edges,
               empty_reads, idle_iters, busy_iters,
               quantile(active_fraction, .1), quantile(active_fraction, .5), quantile(active_fraction, .9),
               quantile(first_fraction, .1), quantile(first_fraction, .5), quantile(first_fraction, .9));
#endif
#if (L3_SOURCE_EXPAND_DIAG == true)
        printf("L3_SOURCE_EXPAND gpu=%d unique=%llu repeated=%llu repeated_edges=%llu\n",
               gpu_id, source_expand.unique_sources, source_expand.repeated_sources,
               source_expand.repeated_edges);
#else
        printf("L3_SOURCE_EXPAND_DISABLED gpu=%d warp_private_only=1\n", gpu_id);
#endif
#endif
#if (L3_RX_FEEDBACK_TRACE == true)
        unsigned trace_count = 0;
        g_benchmark.require(cudaMemcpyFromSymbol(&trace_count, g_l3_feedback_trace_head,
                            sizeof(trace_count)) == cudaSuccess, "feedback trace size");
        // Overflow invalidates evidence instead of silently dropping events.
        if (trace_count > L3_FEEDBACK_TRACE_CAPACITY) {
            fprintf(stderr, "L3 feedback trace overflow: %u\n", trace_count);
            exit(2);
        }
        std::vector<l3_feedback_trace_entry> feedback_trace(trace_count);
        if (trace_count)
            g_benchmark.require(cudaMemcpyFromSymbol(feedback_trace.data(), g_l3_feedback_trace,
                                trace_count * sizeof(l3_feedback_trace_entry)) == cudaSuccess,
                                "feedback trace read");
        for (const auto &entry : feedback_trace) {
            const auto &r = entry.record;
            if (!entry.kind)
                printf("L3_RX_RESULT owner=%d epoch=%d count=%d improved=%d\n",
                       entry.owner, r.epoch, r.count, r.improved);
            else
                printf("L3_TX_RESULT owner=%d receiver=%d tx_epoch=%d epoch=%d count=%d improved=%d before=%llu after=%llu mode=%d\n",
                       entry.owner, entry.receiver, entry.tx_epoch, r.epoch, r.count, r.improved,
                       entry.before, entry.after, entry.mode);
        }
#endif
#if (MANAGE_PROFILE == true)
        if (g_benchmark.enabled) {
            unsigned long long mp[13] = {};
            g_benchmark.require(cudaMemcpy(mp, mgmt_profile, sizeof(mp), cudaMemcpyDeviceToHost)
                                == cudaSuccess, "manager profile read");
            printf("MANAGER_DIAG gpu=%d inject_cycles=%llu backstop_cycles=%llu "
                   "term_cycles=%llu total_cycles=%llu iterations=%llu scan_cycles=%llu "
                   "flush_cycles=%llu tx_total_cycles=%llu scans=%llu mark_cycles=%llu "
                   "confirm_cycles=%llu pre_count=%llu confirm_count=%llu\n",
                   gpu_id, mp[0], mp[1], mp[2], mp[3], mp[4], mp[5], mp[6],
                   mp[7], mp[8], mp[9], mp[10], mp[11], mp[12]);
        }
#endif
#if (L3_MAPPING_DIAG == true)
        l3_mapping_metrics mapping = {};
        g_benchmark.require(cudaMemcpyFromSymbol(&mapping, g_l3_mapping_metrics, sizeof(mapping))
                            == cudaSuccess, "mapping read");
        printf("L3_MAPPING gpu=%d adaptive=%d blocks=%llu eligible=%llu chosen=%llu "
               "static_passes=%llu selected_passes=%llu emitted=%llu block_cycles=%llu "
               "team_cycles=%llu lane_cycles=%llu\n",
               gpu_id, int(L3_ADAPTIVE_WORDS), mapping.blocks, mapping.eligible, mapping.chosen,
               mapping.static_passes, mapping.selected_passes, mapping.emitted,
               mapping.block_cycles, mapping.team_cycles, mapping.lane_cycles);
        for (int bin = 0; bin <= 32; ++bin)
            printf("L3_MAPPING_HIST gpu=%d bin=%d k=%llu m=%llu\n",
                   gpu_id, bin, mapping.k_hist[bin], mapping.m_hist[bin]);
#endif
#if (L3_DIAGNOSTICS == true)
        l3_diagnostics diag = {};
        unsigned long long signal = 0;
        work_count_type local_work = 0, remote_work = 0;
        g_benchmark.require(cudaMemcpyFromSymbol(&diag, g_l3_diagnostics, sizeof(diag))
                            == cudaSuccess, "diagnostic read");
        g_benchmark.require(cudaMemcpyFromSymbol(&signal, g_bulk_mark_signal, sizeof(signal))
                            == cudaSuccess, "signal read");
#if (WORK_COUNT == true)
        g_benchmark.require(cudaMemcpyFromSymbol(&local_work, g_local_work, sizeof(local_work))
                            == cudaSuccess, "local work read");
        g_benchmark.require(cudaMemcpyFromSymbol(&remote_work, g_remote_work, sizeof(remote_work))
                            == cudaSuccess, "remote work read");
#endif
        printf("L3_DIAG gpu=%d mode=%d scans=%llu empty=%llu full=%llu extracted=%llu "
               "scan_cycles=%llu rx_batches=%llu received=%llu improved=%llu "
               "signal=%llu local_work=%llu remote_work=%llu budget_sum=%llu budget_changes=%llu "
               "edges=%llu vertices=%llu tx_batches=%llu adjacent_descents=%llu feedback_drains=%llu "
               "remote_attempts=%llu cache_filtered=%llu candidate_updates=%llu "
               "feedback_batches=%llu feedback_count=%llu feedback_improved=%llu feedback_changes=%llu "
               "event_skips=%llu event_full=%llu\n",
               gpu_id, L3_WINDOW_MODE, diag.scans, diag.empty, diag.full, diag.extracted,
               diag.scan_cycles, diag.rx_batches, diag.received, diag.improved, signal,
               local_work, remote_work, diag.budget_sum, diag.budget_changes,
               diag.edges, diag.vertices, diag.tx_batches, diag.adjacent_descents,
               diag.feedback_drains, diag.remote_attempts, diag.cache_filtered,
               diag.candidate_updates, diag.feedback_batches, diag.feedback_count,
               diag.feedback_improved, diag.feedback_changes, diag.event_skips, diag.event_full);
#if (L3_FAULT_INJECT_PUBLISH_RETRY == true && L3_FAULT_INJECT_CLAIM_RETRY == true && \
     L3_FAULT_INJECT_READY_DELAY == true && L3_FAULT_INJECT_ACK_DELAY == true)
        // BENCH quiet mode suppresses the legacy human-readable fault prints.
        // Keep exercised-fault evidence with diagnostics, outside that guard.
        unsigned fault_claim = 0, fault_ready = 0, fault_ack = 0;
        int fault_pending = -1;
        g_benchmark.require(cudaMemcpyFromSymbol(&fault_claim, g_l3_fault_claim_retry,
                            sizeof(fault_claim)) == cudaSuccess, "fault claim read");
        g_benchmark.require(cudaMemcpyFromSymbol(&fault_ready, g_l3_fault_ready_delay,
                            sizeof(fault_ready)) == cudaSuccess, "fault ready read");
        g_benchmark.require(cudaMemcpyFromSymbol(&fault_ack, g_l3_fault_ack_delay,
                            sizeof(fault_ack)) == cudaSuccess, "fault ack read");
        g_benchmark.require(cudaMemcpyFromSymbol(&fault_pending, g_l3_fault_ack_pending,
                            sizeof(fault_pending)) == cudaSuccess, "fault pending read");
        printf("L3_FAULT_AUDIT gpu=%d publish=%llu claim=%u ready=%u ack=%u pending=%d\n",
               gpu_id, diag.injected_publish_retries, fault_claim, fault_ready, fault_ack, fault_pending);
#endif
        for (unsigned j = 0; j < 32 && j < diag.tx_batches; ++j)
            printf("L3_WAVE gpu=%d batch=%u count=%d min=%d max=%d\n", gpu_id, j,
                   diag.waves[j].count, diag.waves[j].minimum, diag.waves[j].maximum);
#endif

#if (L3_LIVE_SNAPSHOT == true)
        if (l3_live_snapshot_started)
        {
            l3_live_snapshot_stop.store(true, std::memory_order_release);
            l3_live_snapshot_thread.join();
        }
#endif
#if (L3_EVENT_RING == true)
        if (!g_multi_source_quiet)
            l3_event_dump_host(gpu_id);
#endif
        if (!g_multi_source_quiet)
        {
            cudaEventRecord(workspace.stop, 0);
            cudaEventSynchronize(workspace.stop);

            cudaEventElapsedTime(&elapsedTime, workspace.start, workspace.stop);

            printf("gpu%d Elapse time: %.2f ms\n", gpu_id, elapsedTime);

#if (L3_FAULT_INJECT_CLAIM_RETRY == true)
            unsigned int fault_claim_count = 0;
            cudaMemcpyFromSymbol(&fault_claim_count, g_l3_fault_claim_retry,
                                 sizeof(fault_claim_count));
            printf("gpu%d L3_FAULT_CLAIM_RETRY injected=%u (首次 READY claim 有界失败，后续轮询重试)\n",
                   gpu_id, fault_claim_count);
#endif
#if (L3_FAULT_INJECT_READY_DELAY == true)
            unsigned int fault_ready_count = 0;
            cudaMemcpyFromSymbol(&fault_ready_count, g_l3_fault_ready_delay,
                                 sizeof(fault_ready_count));
            printf("gpu%d L3_FAULT_READY_DELAY injected=%u (首次 READY 可见性有界延迟，后续轮询重试)\n",
                   gpu_id, fault_ready_count);
#endif
#if (L3_FAULT_INJECT_ACK_DELAY == true)
            unsigned int fault_ack_count = 0;
            int fault_ack_pending = 0;
            cudaMemcpyFromSymbol(&fault_ack_count, g_l3_fault_ack_delay,
                                 sizeof(fault_ack_count));
            cudaMemcpyFromSymbol(&fault_ack_pending, g_l3_fault_ack_pending,
                                 sizeof(fault_ack_pending));
            printf("gpu%d L3_FAULT_ACK_DELAY injected=%u pending_epoch=%d (首次 ACK 有界延迟，manager 下一轮重发)\n",
                   gpu_id, fault_ack_count, fault_ack_pending);
#endif

#if (MANAGE_PROFILE == true)
        {
            unsigned long long mp[13] = {0};
            cudaMemcpy(mp, mgmt_profile, sizeof(unsigned long long) * 13, cudaMemcpyDeviceToHost);
            double freq = FRE;
            printf("gpu%d manage profile (cycles, ms):\n", gpu_id);
            printf("  warp0  inject=%.4f backstop=%.4f term=%.4f total=%.4f iter=%llu\n",
                   mp[0] / freq, mp[1] / freq, mp[2] / freq, mp[3] / freq, mp[4]);
            if (mp[3] > 0)
                printf("    warp0 ratio: inject=%.2f%% backstop=%.2f%% term=%.2f%%\n",
                       100.0 * mp[0] / mp[3], 100.0 * mp[1] / mp[3], 100.0 * mp[2] / mp[3]);
            printf("    warp0 sub: mark=%.4f confirm=%.4f pre_cnt=%llu confirm_cnt=%llu\n",
                   mp[9] / freq, mp[10] / freq, mp[11], mp[12]);
            printf("  warp17 scan=%.4f flush=%.4f total=%.4f iter=%llu\n",
                   mp[5] / freq, mp[6] / freq, mp[7] / freq, mp[8]);
            if (mp[7] > 0)
                printf("    warp17 ratio: scan=%.2f%% flush=%.2f%%\n",
                       100.0 * mp[5] / mp[7], 100.0 * mp[6] / mp[7]);
        }
#endif

#if (WORK_COUNT == true)
        work_count_type global_work_count_host = 0;
        cudaMemcpy(&global_work_count_host, global_work_count, sizeof(work_count_type), cudaMemcpyDeviceToHost);
        work_count_type local_work_host = 0, remote_work_host = 0;
        cudaMemcpyFromSymbol(&local_work_host, g_local_work, sizeof(work_count_type));
        cudaMemcpyFromSymbol(&remote_work_host, g_remote_work, sizeof(work_count_type));
        printf("gpu%d Total work count %llu (local=%llu remote=%llu)\n",
               gpu_id, global_work_count_host, local_work_host, remote_work_host);
        // E4: w17 三级扫下钻统计
        unsigned long long h2_hit = 0, w_scan = 0, w_hit = 0, remote_eff = 0;
        cudaMemcpyFromSymbol(&h2_hit, g_h2_hit, sizeof(h2_hit));
        cudaMemcpyFromSymbol(&w_scan, g_w_scan, sizeof(w_scan));
        cudaMemcpyFromSymbol(&w_hit, g_w_hit, sizeof(w_hit));
        cudaMemcpyFromSymbol(&remote_eff, g_remote_effective, sizeof(remote_eff));
        printf("gpu%d w17scan: h2_hit=%llu w_scan=%llu w_hit=%llu (false=%.1f%%)\n",
               gpu_id, h2_hit, w_scan, w_hit, w_scan ? 100.0 * (w_scan - w_hit) / w_scan : 0.0);
        printf("gpu%d remote_effective=%llu (flush 真正改进 peer 的次数)\n", gpu_id, remote_eff);
#if (GLOBAL_ROUND_STATS == true)
        unsigned long long global_round_count = 0;
        unsigned long long global_round_sent = 0;
        unsigned long long global_round_recv = 0;
        unsigned long long global_round_improved = 0;
        unsigned long long global_round_frontier_append = 0;
        cudaMemcpyFromSymbol(&global_round_count, g_global_round_count,
                             sizeof(global_round_count));
        cudaMemcpyFromSymbol(&global_round_sent, g_global_round_sent,
                             sizeof(global_round_sent));
        cudaMemcpyFromSymbol(&global_round_recv, g_global_round_recv,
                             sizeof(global_round_recv));
        cudaMemcpyFromSymbol(&global_round_improved, g_global_round_improved,
                             sizeof(global_round_improved));
        cudaMemcpyFromSymbol(&global_round_frontier_append,
                             g_global_round_frontier_append,
                             sizeof(global_round_frontier_append));
        printf("gpu%d GLOBAL_ROUND_STATS: rounds=%llu sent=%llu recv=%llu improved=%llu frontier_append=%llu\n",
               gpu_id, global_round_count, global_round_sent, global_round_recv,
               global_round_improved, global_round_frontier_append);
#endif
#if (GLOBAL_ROUND_DIRECT_TRACE == true)
        unsigned long long dr193 = 0, dr362 = 0, dc193 = 0, dc362 = 0;
        unsigned long long dt193 = 0, dt362 = 0, di193 = 0, di362 = 0;
        int drm193 = 0, drm362 = 0, dtm193 = 0, dtm362 = 0;
        int dim193 = 0, dim362 = 0;
        cudaMemcpyFromSymbol(&dr193, g_direct_relax_193, sizeof(dr193));
        cudaMemcpyFromSymbol(&dr362, g_direct_relax_362, sizeof(dr362));
        cudaMemcpyFromSymbol(&dc193, g_direct_cand_193, sizeof(dc193));
        cudaMemcpyFromSymbol(&dc362, g_direct_cand_362, sizeof(dc362));
        cudaMemcpyFromSymbol(&dt193, g_direct_tx_193, sizeof(dt193));
        cudaMemcpyFromSymbol(&dt362, g_direct_tx_362, sizeof(dt362));
        cudaMemcpyFromSymbol(&di193, g_direct_inj_193, sizeof(di193));
        cudaMemcpyFromSymbol(&di362, g_direct_inj_362, sizeof(di362));
        cudaMemcpyFromSymbol(&drm193, g_direct_relax_193_min, sizeof(drm193));
        cudaMemcpyFromSymbol(&drm362, g_direct_relax_362_min, sizeof(drm362));
        cudaMemcpyFromSymbol(&dtm193, g_direct_tx_193_min, sizeof(dtm193));
        cudaMemcpyFromSymbol(&dtm362, g_direct_tx_362_min, sizeof(dtm362));
        cudaMemcpyFromSymbol(&dim193, g_direct_inj_193_min, sizeof(dim193));
        cudaMemcpyFromSymbol(&dim362, g_direct_inj_362_min, sizeof(dim362));
        printf("gpu%d DIRECT_TRACE: relax193=%llu/%d cand193=%llu tx193=%llu/%d inj193=%llu/%d; relax362=%llu/%d cand362=%llu tx362=%llu/%d inj362=%llu/%d\n",
               gpu_id, dr193, drm193, dc193, dt193, dtm193, di193, dim193,
               dr362, drm362, dc362, dt362, dtm362, di362, dim362);
#endif
        unsigned long long seed_fired = 0;
        cudaMemcpyFromSymbol(&seed_fired, g_seed_ready_fired, sizeof(seed_fired));
        printf("gpu%d seed_ready_fired=%llu (SEED_BARRIER 屏障触发次数)\n", gpu_id, seed_fired);
        unsigned long long inj_gate = 0, bs_gate = 0;
        cudaMemcpyFromSymbol(&inj_gate, g_inj_gate_skip, sizeof(inj_gate));
        cudaMemcpyFromSymbol(&bs_gate, g_bs_gate_skip, sizeof(bs_gate));
        printf("gpu%d gate_skip: inj=%llu bs=%llu (SEED_BARRIER 接收卡 gate 跳过)\n", gpu_id, inj_gate, bs_gate);
        unsigned long long t_ph2 = 0, t_sr = 0, t_inj = 0, t_term = 0, t_seen = 0, t_idle = 0, t_peer = 0, t_srw0 = 0;
        unsigned long long confirm_exec = 0;
        cudaMemcpyFromSymbol(&t_ph2, g_t_phase2, sizeof(timeline_clock_type));
        cudaMemcpyFromSymbol(&t_sr, g_t_seed_ready, sizeof(timeline_clock_type));
        cudaMemcpyFromSymbol(&t_seen, g_t_sr_seen, sizeof(timeline_clock_type));
        cudaMemcpyFromSymbol(&t_inj, g_t_inject_done, sizeof(timeline_clock_type));
        cudaMemcpyFromSymbol(&t_idle, g_t_idle, sizeof(timeline_clock_type));
        cudaMemcpyFromSymbol(&t_peer, g_t_peer_idle_seen, sizeof(timeline_clock_type));
        cudaMemcpyFromSymbol(&t_srw0, g_t_sr_w0, sizeof(timeline_clock_type));
        cudaMemcpyFromSymbol(&t_term, g_t_term, sizeof(timeline_clock_type));
#if (TIMELINE64 == true)
        unsigned long long t_idle_last_set = 0, t_idle_last_reset = 0;
        unsigned long long idle_set_count = 0, idle_reset_count = 0;
        unsigned long long idle_reset_reason[16] = {0};
        unsigned long long term_mark_actual = 0, term_mark_stale = 0;
        unsigned long long term_mark_sampled = 0;
        cudaMemcpyFromSymbol(&t_idle_last_set, g_t_idle_last_set, sizeof(timeline_clock_type));
        cudaMemcpyFromSymbol(&t_idle_last_reset, g_t_idle_last_reset, sizeof(timeline_clock_type));
        cudaMemcpyFromSymbol(&idle_set_count, g_idle_set_count, sizeof(idle_set_count));
        cudaMemcpyFromSymbol(&idle_reset_count, g_idle_reset_count, sizeof(idle_reset_count));
        cudaMemcpyFromSymbol(idle_reset_reason, g_idle_reset_reason, sizeof(idle_reset_reason));
        cudaMemcpyFromSymbol(&term_mark_actual, g_term_mark_actual, sizeof(term_mark_actual));
        cudaMemcpyFromSymbol(&term_mark_stale, g_term_mark_stale, sizeof(term_mark_stale));
        cudaMemcpyFromSymbol(&term_mark_sampled, g_term_mark_sampled, sizeof(term_mark_sampled));
#endif
        cudaMemcpyFromSymbol(&confirm_exec, g_confirm_exec, sizeof(confirm_exec));
        double freq = FRE;
        unsigned long long inj_exec = 0;
        cudaMemcpyFromSymbol(&inj_exec, g_inject_exec, sizeof(inj_exec));
        printf("gpu%d inject exec=%llu (SEED_BARRIER 一次性注入执行次数)\n", gpu_id, inj_exec);
        printf("gpu%d timeline cycles: phase2=%llu seed_ready=%llu sr_seen=%llu sr_w0=%llu inject_done=%llu idle=%llu peer_seen=%llu term=%llu confirm=%llu\n",
               gpu_id, t_ph2, t_sr, t_seen, t_srw0, t_inj, t_idle, t_peer, t_term, confirm_exec);
        printf("gpu%d timeline ms: phase2=%.2f seed_ready=%.2f sr_seen=%.2f sr_w0=%.2f inject_done=%.2f idle=%.2f peer_seen=%.2f term=%.2f\n",
               gpu_id, (double)t_ph2 / freq, (double)t_sr / freq,
               (double)t_seen / freq, (double)t_srw0 / freq,
               (double)t_inj / freq, (double)t_idle / freq,
               (double)t_peer / freq, (double)t_term / freq);
        // 阶段差分：p1=启动→phase2, p2=phase2→seed_ready,
        //   p3=seed_ready→idle, p4=idle→term
        double phase1_ms = t_ph2 ? (double)t_ph2 / freq : 0.0;
        double phase2_ms = (t_sr > t_ph2) ? (double)(t_sr - t_ph2) / freq : 0.0;
        double phase3_ms = (t_sr != 0 && t_idle > t_sr)
                         ? (double)(t_idle - t_sr) / freq : 0.0;
        double phase4_ms = (t_idle != 0 && t_term > t_idle)
                         ? (double)(t_term - t_idle) / freq : 0.0;
        printf("gpu%d phases(ms): p1=%.2f p2=%.2f p3=%.2f p4=%.2f\n", gpu_id,
               phase1_ms, phase2_ms, phase3_ms, phase4_ms);
#if (TIMELINE64 == true)
        double term_after_last_set_ms = (t_term > t_idle_last_set)
                                      ? (double)(t_term - t_idle_last_set) / freq : 0.0;
        double last_reset_to_idle_ms = (t_idle_last_set > t_idle_last_reset && t_idle_last_reset != 0)
                                     ? (double)(t_idle_last_set - t_idle_last_reset) / freq : 0.0;
        printf("gpu%d idle transitions: set=%llu reset=%llu last_reset_to_set=%.2f ms term_after_last_set=%.2f ms\n",
               gpu_id, idle_set_count, idle_reset_count,
               last_reset_to_idle_ms, term_after_last_set_ms);
        printf("gpu%d idle reset reasons: inbox=%llu round=%llu l2=%llu dirty=%llu term_backstop=%llu term_dirty=%llu term_mark=%llu term_ghost=%llu term_l3=%llu term_bulk=%llu term_inj=%llu other=%llu\n",
               gpu_id, idle_reset_reason[1], idle_reset_reason[2], idle_reset_reason[3],
               idle_reset_reason[4], idle_reset_reason[5], idle_reset_reason[6],
               idle_reset_reason[7], idle_reset_reason[8], idle_reset_reason[9],
               idle_reset_reason[10], idle_reset_reason[11], idle_reset_reason[12]);
        printf("gpu%d term mark detail: sampled_failures=%llu actual_remote_mark=%llu stale_hint2=%llu\n",
               gpu_id, term_mark_sampled, term_mark_actual, term_mark_stale);
#endif
#if (WORK_SEG_PROFILE == true)
        unsigned long long wk_clks = 0;
        cudaMemcpyFromSymbol(&wk_clks, g_wk_active_clks, sizeof(wk_clks));
        printf("gpu%d work_active(ms): %.2f\n", gpu_id, (double)wk_clks / freq);
#endif
#if (DQ_CLAMP_DIAG == true)
        unsigned long long dq_clamp = 0;
        cudaMemcpyFromSymbol(&dq_clamp, g_dq_clamp, sizeof(dq_clamp));
        printf("gpu%d dq_clamp=%llu (delta 桶倒退距离 clamp 次数)\n", gpu_id, dq_clamp);
#endif
        // B1-2a: per-vertex 改进直方图统计
        if (hist)
        {
            unsigned int *hist_h = (unsigned int *)malloc((v_local + 1) * sizeof(unsigned int));
            cudaMemcpy(hist_h, hist, (v_local + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            unsigned int cnt[8] = {0};  // 0,1,2,3,4-5,6-9,10-19,20+
            for (int i = 0; i < v_local; i++)
            {
                unsigned int h = hist_h[i];
                if (h == 0) cnt[0]++;
                else if (h == 1) cnt[1]++;
                else if (h == 2) cnt[2]++;
                else if (h == 3) cnt[3]++;
                else if (h <= 5) cnt[4]++;
                else if (h <= 9) cnt[5]++;
                else if (h <= 19) cnt[6]++;
                else cnt[7]++;
            }
            printf("gpu%d hist: 0=%u 1=%u 2=%u 3=%u 4-5=%u 6-9=%u 10-19=%u 20+=%u\n",
                   gpu_id, cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], cnt[6], cnt[7]);
            // top-10 高频顶点（全局 id + 次数）
            struct { unsigned int c; int idx; } top[10];
            for (int t = 0; t < 10; t++) { top[t].c = 0; top[t].idx = -1; }
            for (int i = 0; i < v_local; i++)
            {
                unsigned int h = hist_h[i];
                for (int t = 0; t < 10; t++)
                    if (h > top[t].c)
                    {
                        for (int s = 9; s > t; s--) { top[s] = top[s - 1]; }
                        top[t].c = h; top[t].idx = i;
                        break;
                    }
            }
            printf("gpu%d top10:", gpu_id);
            for (int t = 0; t < 10 && top[t].idx >= 0; t++)
                printf(" v%d(%u)", gctx[gpu_id].v_begin + top[t].idx, top[t].c);
            printf("\n");
            free(hist_h);
        }
#endif

#if (GLOBAL_ROUND_PROFILE == true)
        unsigned long long gr_rounds = 0;
        unsigned long long gr_local_empty = 0, gr_quiesce = 0;
        unsigned long long gr_pack = 0, gr_inbox_wait = 0;
        unsigned long long gr_apply = 0, gr_release = 0;
        unsigned long long gr_local_empty_max = 0, gr_quiesce_max = 0;
        unsigned long long gr_pack_max = 0, gr_inbox_wait_max = 0;
        unsigned long long gr_apply_max = 0, gr_release_max = 0;
        cudaMemcpyFromSymbol(&gr_rounds, g_gr_profile_rounds, sizeof(gr_rounds));
        cudaMemcpyFromSymbol(&gr_local_empty, g_gr_local_empty, sizeof(gr_local_empty));
        cudaMemcpyFromSymbol(&gr_quiesce, g_gr_quiesce, sizeof(gr_quiesce));
        cudaMemcpyFromSymbol(&gr_pack, g_gr_pack, sizeof(gr_pack));
        cudaMemcpyFromSymbol(&gr_inbox_wait, g_gr_inbox_wait, sizeof(gr_inbox_wait));
        cudaMemcpyFromSymbol(&gr_apply, g_gr_apply, sizeof(gr_apply));
        cudaMemcpyFromSymbol(&gr_release, g_gr_release, sizeof(gr_release));
        cudaMemcpyFromSymbol(&gr_local_empty_max, g_gr_local_empty_max,
                             sizeof(gr_local_empty_max));
        cudaMemcpyFromSymbol(&gr_quiesce_max, g_gr_quiesce_max, sizeof(gr_quiesce_max));
        cudaMemcpyFromSymbol(&gr_pack_max, g_gr_pack_max, sizeof(gr_pack_max));
        cudaMemcpyFromSymbol(&gr_inbox_wait_max, g_gr_inbox_wait_max,
                             sizeof(gr_inbox_wait_max));
        cudaMemcpyFromSymbol(&gr_apply_max, g_gr_apply_max, sizeof(gr_apply_max));
        cudaMemcpyFromSymbol(&gr_release_max, g_gr_release_max, sizeof(gr_release_max));
        double gr_freq = FRE;
        printf("gpu%d GLOBAL_ROUND_PROFILE rounds=%llu cycles: local_empty=%llu quiesce=%llu pack=%llu inbox_wait=%llu apply=%llu release=%llu\n",
               gpu_id, gr_rounds, gr_local_empty, gr_quiesce, gr_pack,
               gr_inbox_wait, gr_apply, gr_release);
        printf("gpu%d GLOBAL_ROUND_PROFILE ms: local_empty=%.3f quiesce=%.3f pack=%.3f inbox_wait=%.3f apply=%.3f release=%.3f\n",
               gpu_id, gr_local_empty / gr_freq, gr_quiesce / gr_freq,
               gr_pack / gr_freq, gr_inbox_wait / gr_freq,
               gr_apply / gr_freq, gr_release / gr_freq);
        printf("gpu%d GLOBAL_ROUND_PROFILE max_ms: local_empty=%.3f quiesce=%.3f pack=%.3f inbox_wait=%.3f apply=%.3f release=%.3f\n",
               gpu_id, gr_local_empty_max / gr_freq, gr_quiesce_max / gr_freq,
               gr_pack_max / gr_freq, gr_inbox_wait_max / gr_freq,
               gr_apply_max / gr_freq, gr_release_max / gr_freq);
#endif

#if (WORK_CLOCK == true)
        int global_comp_count_host = 0;
        int profile_host[2];
        cudaMemcpy(&global_comp_count_host, global_comp_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(profile_host, profile, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        printf("gpu%d Total comp count %d\n", gpu_id, global_comp_count_host);
        float profile_total = profile_host[1];
        printf("gpu%d total time %.4f read proportion %.4f process proportion %.4f\n", 
        gpu_id, profile_total / FRE, profile_host[0] / profile_total, (profile_total - profile_host[0]) / profile_total);
#endif
        }
    }

}

int sssp_run_adaptive(int gpu_id, int src, graph_info info)
{

    mlmq_setup setup;

    setup.init_setup();
    // 队列自适应: 按图特征覆盖（main.cu 设置 g_queue_override，如计算主导图用 L1V_L2DQ）
    if (g_queue_override >= 0) setup.type = (mlmq_type)g_queue_override;
#if (L3_DIRECT_RX == true)
    if (setup.type != L1V_L2DQ && setup.type != L1SLF_L2DQ) {
        fprintf(stderr, "L3_DIRECT_RX supports only L1V/L1SLF + L2DQ\n");
        exit(2);
    }
#endif
    //setup.init_setup_adaptive(info);
    // E2 实验: 命令行 -d 覆盖 delta（扫描 delta 对 work/wall 的影响）
    if (g_delta_override > 0)
        setup.s_l2_delta = g_delta_override;

    switch (setup.type)
    {
        case L1N_L2DQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(gpu_id, src, setup);
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1V_L2DQ:
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(gpu_id, src, setup);
            kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1NF_L2DQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1V_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1N_L2V:
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1NF_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_near_far_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1FQ_L2DQ:
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(gpu_id, src, setup);
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue_new<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1V_L2PQ:
            //kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_bgpq_queue<NODE_TYPE>>>(gpu_id, src, setup);
            kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_bgpq_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1NF_L2PQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_near_far_queue<NODE_TYPE>, l2_bgpq_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1FQ_L2PQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_bgpq_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1V_L2MPQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_multi_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1NF_L2MPQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_near_far_queue<NODE_TYPE>, l2_multi_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1FQ_L2MPQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_multi_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1HQ_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_hop_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1HQ_L2DQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_hop_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1FQ_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_filter_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1N_L2MV:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_multi_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1N_L2BV:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_none_queue<NODE_TYPE>, l2_batch_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1V_L2BV:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_vector_queue<NODE_TYPE>, l2_batch_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1SLF_L2DQ:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_SLF_queue<NODE_TYPE>, l2_delta_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        case L1SLF_L2V:
            kernel_adaptive<ml_queue<NODE_TYPE, l1_SLF_queue<NODE_TYPE>, l2_vector_queue<NODE_TYPE>>>(gpu_id, src, setup);
            break;
        default:
            printf("MLMQ type not implemented!\n");
    }

    return 0;
}
