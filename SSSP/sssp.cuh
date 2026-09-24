#pragma once

#include "csr_graph.h"
#include "../core/include/ml_queue.cuh"

// The queue host API takes int bytes: 2 GiB itself is not representable.
#define GPU_MEMORY INT_MAX

// Exact static domain of remote marks; runtime candidates remain dynamic.
#ifndef L3_BOUNDARY_INDEX
#define L3_BOUNDARY_INDEX false
#endif
#ifndef L3_RECOVERY_DOMAIN_DIAG
#define L3_RECOVERY_DOMAIN_DIAG false
#endif
#ifndef L3_COMPACT_CANDIDATES
#define L3_COMPACT_CANDIDATES false
#endif
#ifndef L3_CHAIN_SHORTCUTS
#define L3_CHAIN_SHORTCUTS false
#endif
#ifndef L3_REGION_RELAX
#define L3_REGION_RELAX false
#endif
#ifndef L3_REGION_DIAG
#define L3_REGION_DIAG false
#endif
#ifndef L3_CHAIN_PARTITION
#define L3_CHAIN_PARTITION false
#endif
#ifndef L3_CHAIN_PARTITION_DEGREE_GATE
#define L3_CHAIN_PARTITION_DEGREE_GATE false
#endif
#ifndef L3_CHAIN_PARTITION_DIAG
#define L3_CHAIN_PARTITION_DIAG false
#endif
#if (L3_CHAIN_PARTITION_DIAG == true && L3_CHAIN_PARTITION == false)
#error "chain partition diagnostics require L3_CHAIN_PARTITION=true"
#endif
#if (L3_CHAIN_PARTITION == true)
constexpr int L3_CHAIN_PARTITION_TAG = 1 << 30;
#endif
#ifndef L3_IDLE_TOKEN_PROBE
#define L3_IDLE_TOKEN_PROBE false
#endif
// L3_IDLE_BACKOFF: dual-GPU work warps briefly sleep after consecutive
// fully-idle reads instead of spin-reading the delta queue counters.
// Single-GPU (n_gpu==1) always keeps the original non-backoff path.
#ifndef L3_IDLE_BACKOFF
#define L3_IDLE_BACKOFF false
#endif
#ifndef L3_TERM_ONLY_WORKER
#define L3_TERM_ONLY_WORKER false
#endif
#if (L3_CHAIN_SHORTCUTS == true && (!defined(TYPE_INT) || EDGE_BALANCED_PARTITION == true || GHOST_DEPTH > 0 || SEED_EXP == true))
#error "owner-local chain shortcuts require integer weights and fixed vertex ownership"
#endif
#if (L3_COMPACT_CANDIDATES == true && (L3_BOUNDARY_INDEX == false || L3_RETAIN_TX == false || L3_DIRECT_RX == false || L3_WINDOW_MODE == 0))
#error "compact candidate domains require indexed, retained, direct-RX window BULK"
#endif

// calculate total work count
#define ANALYSIS false

#ifndef WORK_COUNT
#define WORK_COUNT true
#endif
#define WORK_CLOCK ANALYSIS
#define PROFILE_COUNT ANALYSIS

// V0.5 manage kernel profiling (阶段 E: 量化 manage 各环节占比)
#ifndef MANAGE_PROFILE
#define MANAGE_PROFILE false
#endif

// query workspace 复用诊断：仅在显式诊断构建中逐 source 记录 cudaMemGetInfo，
// 默认关闭，避免改变多源性能测量路径。
#ifndef QUERY_WORKSPACE_DIAG
#define QUERY_WORKSPACE_DIAG false
#endif
#ifndef L3_L2_FINAL_COUNTS
#define L3_L2_FINAL_COUNTS false
#endif
#ifndef L3_TIMING_DIAG
#define L3_TIMING_DIAG false
#endif
#ifndef L3_BOUNDED_RECOVERY
#define L3_BOUNDED_RECOVERY false
#endif
#ifndef L3_STICKY_READY
#define L3_STICKY_READY false
#endif
#ifndef L3_WORK_DIAG
#define L3_WORK_DIAG false
#endif
#ifndef L3_WORK_COUNT_ONLY
#define L3_WORK_COUNT_ONLY false
#endif
#ifndef L3_SOURCE_EXPAND_DIAG
// Preserve historical --work diagnostics; --work-lite explicitly disables
// per-source/global atomics while retaining warp-private work counters.
#define L3_SOURCE_EXPAND_DIAG L3_WORK_DIAG
#endif
#if (L3_SOURCE_EXPAND_DIAG == true && L3_WORK_DIAG == false)
#error "source expansion diagnostics require L3_WORK_DIAG"
#endif
#ifndef L3_WORKER_RECOVERY
#define L3_WORKER_RECOVERY false
#endif
#ifndef L3_TERM_WAIT_ACK
#define L3_TERM_WAIT_ACK false
#endif
#ifndef L3_RX_PRIORITY_BOOTSTRAP
#define L3_RX_PRIORITY_BOOTSTRAP false
#endif
#ifndef L3_RX_LAG_DIAG
#define L3_RX_LAG_DIAG false
#endif
#ifndef L3_ADMISSION_BUDGET
#define L3_ADMISSION_BUDGET false
#endif

// One fixed, default-off local scheduling experiment (stage153).
#ifndef L3_LOCAL_YIELD_BATCHES
#define L3_LOCAL_YIELD_BATCHES 0
#endif
#ifndef L3_LOCAL_YIELD_DIAG
#define L3_LOCAL_YIELD_DIAG false
#endif
#ifndef L0_DIRECT_SMALL
#define L0_DIRECT_SMALL false
#endif
#ifndef L0_SOURCE_SNAPSHOT
#define L0_SOURCE_SNAPSHOT false
#endif
#if (L3_LOCAL_YIELD_BATCHES < 0)
#error "local yield budget must be nonnegative"
#endif

// P2 persistent tile loan：默认关闭的 owner-preserving one-hop 接线实验。
// 资源/ABI 先独立于当前 BULK 数据面铺设；只有显式打开时才允许后续 work_block
// producer/helper 代码参与执行。它不改变距离 owner，也不借用对端 q2。
#ifndef L3_TILE_LOAN
#define L3_TILE_LOAN false
#endif
#ifndef L3_CONTINUATION
#define L3_CONTINUATION false
#endif
#ifndef L3_CONTINUATION_ROWS
#define L3_CONTINUATION_ROWS 128
#endif
#ifndef L3_COMPLETED_ROWS
#define L3_COMPLETED_ROWS false
#endif
#ifndef L3_CONTINUATION_WARP
#define L3_CONTINUATION_WARP false
#endif
#ifndef L3_MULTI_PRODUCER
#define L3_MULTI_PRODUCER false
#endif
#ifndef L3_MULTI_PRODUCER_PREFILTER
#define L3_MULTI_PRODUCER_PREFILTER false
#endif
#ifndef L3_PRODUCER_STOP_POLL
#define L3_PRODUCER_STOP_POLL false
#endif
#if (L3_PRODUCER_STOP_POLL == true && L3_MULTI_PRODUCER == false)
#error "stop poll requires monotonic multi-producer gates"
#endif
#define L3_PRODUCER_COUNTER_WORDS (L3_PRODUCER_STOP_POLL ? 12288 : 8192)
#if (L3_MULTI_PRODUCER_PREFILTER == true && L3_MULTI_PRODUCER == false)
#error "producer prefilter requires multi producer"
#endif
#if (L3_MULTI_PRODUCER == true && L3_CONTINUATION == false)
#error "multi producer requires continuation"
#endif
#if (L3_CONTINUATION_WARP == true && L3_CONTINUATION == false)
#error "warp continuation requires continuation resources"
#endif
#if (L3_COMPLETED_ROWS == true && L3_CONTINUATION == false)
#error "completed row evidence requires continuation"
#endif
#if (L3_CONTINUATION == true && (L3_TILE_LOAN == false || !defined(TYPE_INT)))
#error "continuation requires tile-loan resources and integer distances"
#endif
#ifndef L3_TILE_LOAN_MAX_SEEDS
#define L3_TILE_LOAN_MAX_SEEDS node_size
#endif
#ifndef L3_TILE_LOAN_MIN_QUEUE
#define L3_TILE_LOAN_MIN_QUEUE L3_TILE_LOAN_MAX_SEEDS
#endif
#ifndef L3_TILE_LOAN_POLL_INTERVAL
#define L3_TILE_LOAN_POLL_INTERVAL 64
#endif
#ifndef L3_TILE_LOAN_RESULT_CAP
#define L3_TILE_LOAN_RESULT_CAP 4096
#endif
#ifndef L3_TILE_LOAN_DIAG
#define L3_TILE_LOAN_DIAG false
#endif
#ifndef L3_TILE_LOAN_READ_RETRIES
#define L3_TILE_LOAN_READ_RETRIES 8
#endif
#ifndef L3_OWNER_COMMIT
// G3 experiment: helper expands a donor source and commits destinations
// directly to the authoritative owner through the existing loan slot.  It is
// intentionally disabled in every production build.
#define L3_OWNER_COMMIT false
#endif
#if (L3_CONTINUATION == true && L3_OWNER_COMMIT == true)
#error "continuation and direct owner commit are separate experiments"
#endif
#if (L3_TILE_LOAN_READ_RETRIES < 0)
#error "L3_TILE_LOAN_READ_RETRIES must be non-negative"
#endif
#if (L3_OWNER_COMMIT == true && L3_TILE_LOAN == false)
#error "L3_OWNER_COMMIT requires the bounded tile-loan slot and peer CSR binding"
#endif
#if (L3_OWNER_COMMIT == true && !defined(TYPE_INT))
#error "L3_OWNER_COMMIT currently requires VALUE_TYPE=int"
#endif
#if (L3_TILE_LOAN == true)
// Set once by the host before query worker threads launch. H/P controls
// therefore use one identical loan binary and differ only at runtime.
extern int g_l3_tile_loan_enabled;
#endif
// L3_RX_EXPRESS：把接收端真正改善的候选优先交给一个指定 work warp，
// 绕过“atomicMin -> dirty/write_through -> L2 -> mlmq.read”的等待链。
// 默认关闭；该实验只接入默认 BULK/DIRECT_RX 数据面，且不改变距离 owner、
// remote candidate 或 inbox generation/ACK 语义。
#ifndef L3_RX_EXPRESS
#define L3_RX_EXPRESS false
#endif
#ifndef L3_RX_EXPRESS_SLOTS
#define L3_RX_EXPRESS_SLOTS 64
#endif
#ifndef L3_RX_EXPRESS_BATCH
#define L3_RX_EXPRESS_BATCH 32
#endif
#ifndef L3_RX_EXPRESS_DIAG
#define L3_RX_EXPRESS_DIAG false
#endif
#ifndef L3_RX_EXPRESS_TEST_DELAY
#define L3_RX_EXPRESS_TEST_DELAY false
#endif
#ifndef L3_RX_EXPRESS_DELAY_LOOPS
#define L3_RX_EXPRESS_DELAY_LOOPS 0
#endif
#if (L3_RX_EXPRESS == true)
// Fixed by the host before the query threads launch.  It is passed as a
// runtime value so an express-compiled binary can run a disabled control with
// the same allocations, ABI, and termination checks.
extern int g_rx_express_enabled;
#endif
#if (L3_RX_EXPRESS_TEST_DELAY == true && L3_RX_EXPRESS_DIAG == false)
#error "L3_RX_EXPRESS_TEST_DELAY is diagnostic-only"
#endif
#if (L3_RX_EXPRESS_DELAY_LOOPS < 0)
#error "L3_RX_EXPRESS_DELAY_LOOPS must be non-negative"
#endif

// L3_RX_L2_PULL：RX manager 在完成一个真正 winner inbox 的 L2 提交后，
// 只发布一个 device-local 序号提示；每个 block 的 wid==0 worker 在安全边界看到
// 新提示且 L1 仍有工作时，额外直接读取一次 L2/DQ，然后继续现有
// simple_process/update_done 路径。默认关闭，且不与 RX express 叠加。
#ifndef L3_RX_L2_PULL
#define L3_RX_L2_PULL false
#endif
#ifndef L3_RX_L2_PULL_DIAG
#define L3_RX_L2_PULL_DIAG false
#endif
#ifndef L3_RX_L2_PULL_SINGLE_CLAIM
#define L3_RX_L2_PULL_SINGLE_CLAIM false
#endif
#if (L3_RX_L2_PULL == true)
// Fixed before query worker threads launch.  The pull binary can therefore
// provide an observe-only (disabled) control with the same ABI and allocation.
extern int g_rx_l2_pull_enabled;
#endif
#if (L3_RX_L2_PULL_DIAG == true && L3_RX_L2_PULL == false)
#error "L3_RX_L2_PULL_DIAG requires L3_RX_L2_PULL"
#endif
#if (L3_RX_L2_PULL_SINGLE_CLAIM == true && L3_RX_L2_PULL == false)
#error "L3_RX_L2_PULL_SINGLE_CLAIM requires L3_RX_L2_PULL"
#endif

// HANG_DIAG: 大图 n=2 卡死心跳诊断（周期性打印 warp0/warp17 迭代数 + 状态）
#ifndef HANG_DIAG
#define HANG_DIAG false
#endif
#ifndef HANG_DIAG_K
#define HANG_DIAG_K 10000
#endif

// 长尾诊断：默认使用原有 32-bit clock()；诊断构建可通过
// -DTIMELINE64=true 切换为 64-bit clock64()，避免长 kernel 运行时回绕。
#ifndef TIMELINE64
#define TIMELINE64 false
#endif

// exp-sp_async: SP Async 思想实验开关——注入顶点 BF 化（本地出边直接松弛，绕开 delta 桶序）
// 默认关（保持 para-frame3 生产路径）。开启时注入 warp 对注入顶点的本地出边做 relax_dst
// 松弛（BF 式），改进的本地邻居经 write_through 回 L2 交给 work。对照 gpu1 work 冗余是否下降。
#define SP_ASYNC_BF false

// exp-sp_async: 深度 BF 扩展参数（SP_ASYNC_BF=true 时生效）
// BF_MAX_DEPTH: 注入 BF 最大层数（每层 BF 赢的邻居作为下一层输入，逐层扩散）
// BF_DEPTH_CAP: 每层 BF buffer 容量（共享内存，per 注入 warp 专属）
#define BF_MAX_DEPTH 4
#define BF_DEPTH_CAP 128

// 前置实验（决定性对照，test.log SESSION 24 遗留）: gpu1 分区单独跑 + 边界最终值预置种子。
// 量化 gpu1 内部纯 delta-stepping 的固有 work（对照 phase化 PoC 的 4755万），
// 判定冗余是"算法固有（多波前 delta-stepping）"还是"注入路径增量处理浪费"。
// 默认 false（生产路径不变）。开启时 main.cu 走单卡 gpu1 子图 + sssp_set_seeds 种子注入。
#define SEED_EXP false

// 通用双卡协议：本地收敛后按 epoch 批量交换远程候选，默认开启。
// BULK_ROUND 是当前生产路径；SEED_BARRIER 仅保留为显式可选的 P2 快速路径。
#ifndef BULK_ROUND
#define BULK_ROUND true
#endif
#ifndef BULK_EPOCH
// Epoch frontier exchange 原型：默认关闭，先以独立构建验证 quiesce/inbox
// 协议；打开后才启用 manage/L3 的完整 epoch 状态机。
#define BULK_EPOCH false
#endif
#ifndef BULK_FRONTIER
// BULK_EPOCH 的接收结果走独立 frontier；默认只在显式 epoch 构建中生效。
#define BULK_FRONTIER true
#endif
#ifndef BULK_INBOX_BATCH
#define BULK_INBOX_BATCH 128
#endif
#ifndef L3_RX_COMMIT_BATCH
// winner跨输入warp积累到此阈值再提交，inbox末尾提交不足阈值的余量。
// 默认32不等于旧版每个输入warp立即提交；136号对照未证明恢复旧语义有净收益。
// 这是刷新阈值而非严格上限，单次提交最多多WARP_SIZE-1条；大批次未采纳。
#define L3_RX_COMMIT_BATCH WARP_SIZE
#endif
#if (L3_RX_COMMIT_BATCH < WARP_SIZE || \
     (L3_RX_COMMIT_BATCH % WARP_SIZE) != 0 || \
     L3_RX_COMMIT_BATCH > (2 * node_size * WARP_NUM_PER_BLOCK))
#error "L3_RX_COMMIT_BATCH must be a warp multiple within the manager journal buffer"
#endif
#ifndef BULK_L3_BATCH
#define BULK_L3_BATCH 128
#endif

// DW-L3：动态候选合并实验路径。默认关闭，打开后只改变 L3 扫描/发送时机，
// 不改变 remote_cand/remote_mark、inbox generation 或接收端 atomicMin 语义。
#ifndef L3_DYNAMIC
#define L3_DYNAMIC false
#endif
#ifndef L3_DYNAMIC_MAX_DELAY_ROUNDS
#define L3_DYNAMIC_MAX_DELAY_ROUNDS 32
#endif
#ifndef L3_DYNAMIC_COALESCE_ROUNDS
#define L3_DYNAMIC_COALESCE_ROUNDS 4
#endif
#ifndef L3_DYNAMIC_SIGNAL_THRESHOLD
#define L3_DYNAMIC_SIGNAL_THRESHOLD 8
#endif
#ifndef L3_FAULT_INJECT_PUBLISH_RETRY
// 仅用于 CUDA 合约/重试 smoke：让默认 BULK L3 的首次发布尝试返回失败，
// 验证调用方会原样回挂 candidate/mark 并在下一轮重试。生产构建关闭。
#define L3_FAULT_INJECT_PUBLISH_RETRY false
#endif
#ifndef L3_FAULT_INJECT_CLAIM_RETRY
// 仅用于 CUDA transport smoke：让接收端首次看到 READY 时暂不 claim，
// 下一次轮询必须重新读取并领取同一 generation。生产构建关闭。
#define L3_FAULT_INJECT_CLAIM_RETRY false
#endif
#ifndef L3_FAULT_INJECT_READY_DELAY
// 仅用于 CUDA transport smoke：让接收端首次观察到完整 READY/generation 时
// 暂不向上层报告 ready，下一次轮询重新观察同一 slot。生产构建关闭。
#define L3_FAULT_INJECT_READY_DELAY false
#endif
#ifndef L3_FAULT_INJECT_ACK_DELAY
// 仅用于 CUDA transport smoke：接收端首次完成一个 slot 时延迟一次 ACK，
// manager 下一轮重发同一 epoch。验证有限 ACK 可见性延迟下槽位仍可复用；
// 生产构建关闭，永久 ACK 丢失不由该模型表示。
#define L3_FAULT_INJECT_ACK_DELAY false
#endif
#ifndef L3_EVENT_RING
// CP1c：有界 L3 状态事件环。默认关闭；显式诊断构建才分配/记录事件，
// 不改变生产 kernel 的参数和数据通路。
#define L3_EVENT_RING false
#endif
#ifndef L3_EVENT_RING_CAP
#define L3_EVENT_RING_CAP 256
#endif
#if (L3_EVENT_RING_CAP < 8)
#error "L3_EVENT_RING_CAP must be >= 8"
#endif
#ifndef L3_LIVE_SNAPSHOT
// CP1d：Host 侧实时读取 L3/终止协议元数据，用于定位 kernel timeout 前的最后状态。
// 默认关闭；只在显式诊断构建中创建独立 nonblocking CUDA stream，不改变协议语义。
#define L3_LIVE_SNAPSHOT false
#endif
#ifndef L3_LIVE_SNAPSHOT_PERIOD_MS
#define L3_LIVE_SNAPSHOT_PERIOD_MS 250
#endif
#ifndef L3_TERM_HANDSHAKE
// 默认 BULK 路径使用独立的双端终止握手；GLOBAL_ROUND/BULK_EPOCH 等实验路径
// 保留各自已有的 quiesce/termination 状态机。
#define L3_TERM_HANDSHAKE true
#endif
#ifndef BULK_INBOX_SLOTS
// 通用 inbox 的流水槽数。1 保持单槽基线；2 允许相邻 epoch 重叠发布/消费。
#define BULK_INBOX_SLOTS 2
#endif
#if (BULK_INBOX_SLOTS < 1)
#error "BULK_INBOX_SLOTS must be >= 1"
#endif

// BULK inbox 槽位生命周期。epoch/count/ack 保留作为兼容诊断字段；state 和
// generation 是实际的所有权/可见性协议，避免双槽复用时仅凭 epoch 读到半成品。
#define BULK_SLOT_FREE    0
#define BULK_SLOT_WRITING 1
#define BULK_SLOT_READY   2
#define BULK_SLOT_READING 3
#define BULK_SLOT_DONE    4
// READING -> CLOSING 禁止新的 async receive claim；manager 等待已经登记的
// inflight claim 完成后，才发布 ACK/DONE 并允许发送端复用槽位。
#define BULK_SLOT_CLOSING 5

// GLOBAL_ROUND_ASYNC candidate bank ownership. 只有持有 ACTIVE bank lease 的
// work warp 才能生产候选；FROZEN bank 由 L3 独占消费；FREE bank 可切换为 ACTIVE。
#define ASYNC_CAND_BANK_FREE    0
#define ASYNC_CAND_BANK_ACTIVE  1
#define ASYNC_CAND_BANK_FROZEN  2

// GLOBAL_ROUND：通用双卡 round/barrier 协议实验路径。
// 与当前按 L3_BATCH 发布的 BULK_ROUND 分离：每一轮先冻结两卡本地 work，
// 再各自完整聚合 remote_cand/remote_mark、互发一份 round seed list，
// 双方消费完成后才解除 work gate。默认关闭，验证通过后再切为生产默认。
#ifndef GLOBAL_ROUND
#define GLOBAL_ROUND false
#endif
#if (GLOBAL_ROUND == true && BULK_ROUND == false)
#error "GLOBAL_ROUND requires BULK_ROUND"
#endif
#if (GLOBAL_ROUND == true && BULK_EPOCH == true)
#error "GLOBAL_ROUND and BULK_EPOCH are mutually exclusive"
#endif

// GLOBAL_ROUND_ASYNC：无全局 work quiesce 的候选双 bank 实验路径。
// work warp 只在一个短 lease 内使用 active bank；L3 将其冻结后切换到另一
// 个 bank，再等待旧 bank 的 lease 结束并打包。接收侧直接 atomicMin + 批量
// write_through，避免 GLOBAL_ROUND 的全局 round/barrier 和 dirty 注入链。
// 默认关闭，先作为独立构建验证正确性和长图性能。
#ifndef GLOBAL_ROUND_ASYNC
#define GLOBAL_ROUND_ASYNC false
#endif
// GLOBAL_ROUND_WINDOWED_ASYNC：窗口化异步实验路径。
// 复用 GLOBAL_ROUND_ASYNC 的双 candidate bank + generation-safe inbox 发送侧，
// 但接收侧回到已经过多图回归的 dirty_bitmap -> write_through 链路：
// work 不进入全局 quiesce，manager 只负责把 inbox 改进置脏，由 injection
// warp 按现有协议处理。该开关只用于独立实验构建，默认关闭。
#ifndef GLOBAL_ROUND_WINDOWED_ASYNC
#define GLOBAL_ROUND_WINDOWED_ASYNC false
#endif
#if (GLOBAL_ROUND_WINDOWED_ASYNC == true && GLOBAL_ROUND_ASYNC == false)
#undef GLOBAL_ROUND_ASYNC
#define GLOBAL_ROUND_ASYNC true
#endif
#if (GLOBAL_ROUND_WINDOWED_ASYNC == true && \
     (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true || \
      GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true))
#error "GLOBAL_ROUND_WINDOWED_ASYNC uses the dirty receive path, not async receive frontier"
#endif
#if (GLOBAL_ROUND_ASYNC == true && BULK_ROUND == false)
#error "GLOBAL_ROUND_ASYNC requires BULK_ROUND"
#endif
#if (GLOBAL_ROUND_ASYNC == true && (GLOBAL_ROUND == true || BULK_EPOCH == true))
#error "GLOBAL_ROUND_ASYNC is mutually exclusive with GLOBAL_ROUND/BULK_EPOCH"
#endif
#ifndef GLOBAL_ROUND_ASYNC_RX_FRONTIER
// 异步接收 frontier 实验：接收卡不再由 manager 执行
// atomicMin -> write_through，而由 work warp 直接领取 generation-safe inbox
// 条目并处理。默认关闭，保留 GLOBAL_ROUND_ASYNC 的旧 direct-apply 回退路径。
#define GLOBAL_ROUND_ASYNC_RX_FRONTIER false
#endif
#if (GLOBAL_ROUND_ASYNC_RX_FRONTIER == true && GLOBAL_ROUND_ASYNC == false)
#error "GLOBAL_ROUND_ASYNC_RX_FRONTIER requires GLOBAL_ROUND_ASYNC"
#endif
#ifndef GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER
// manager-owned receive frontier：warp0 manager 领取 generation-safe inbox，
// 只做 atomicMin(node_data)+ready bitmap 发布；唯一指定的 work warp 再从
// ready bitmap 领取顶点并调用现有 simple_process。这样不在 manager 中直接
// write_through，也不把 inbox claim 状态机和寄存器压力带入每个 work warp。
// 默认关闭，作为 GLOBAL_ROUND_ASYNC_RX_FRONTIER 的低寄存器替代实验路径。
#define GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER false
#endif
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true && GLOBAL_ROUND_ASYNC == false)
#error "GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER requires GLOBAL_ROUND_ASYNC"
#endif
#if (GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true && \
     GLOBAL_ROUND_ASYNC_RX_FRONTIER == true)
#error "GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER and RX_FRONTIER are mutually exclusive"
#endif
#ifndef GLOBAL_ROUND_DIAG
#define GLOBAL_ROUND_DIAG false
#endif
#ifndef GLOBAL_ROUND_STATS
// GLOBAL_ROUND_STATS：无 printf 的 round 级计数诊断；默认关闭，避免改变生产路径。
#define GLOBAL_ROUND_STATS false
#endif
#ifndef GLOBAL_ROUND_PROFILE
// GLOBAL_ROUND_PROFILE：同一 manager warp 内的低扰动阶段计时；默认关闭。
// 必须和 TIMELINE64 一起启用，避免 32-bit clock() 在长图上回绕。
#define GLOBAL_ROUND_PROFILE false
#endif
#if (GLOBAL_ROUND_PROFILE == true && TIMELINE64 == false)
#error "GLOBAL_ROUND_PROFILE requires TIMELINE64"
#endif
#ifndef GLOBAL_ROUND_APPLY_BATCH
// GLOBAL_ROUND 接收侧使用 manage warp 的 node_out 预留区做压实缓冲。
// 当前 shared-memory 布局可安全容纳 2 * node_size * WARP_NUM_PER_BLOCK 条目；
// 外部实验可以用 -DGLOBAL_ROUND_APPLY_BATCH=128/512 等较小值做对照。
#define GLOBAL_ROUND_APPLY_BATCH (2 * node_size * WARP_NUM_PER_BLOCK)
#endif
#if (GLOBAL_ROUND_APPLY_BATCH > (2 * node_size * WARP_NUM_PER_BLOCK))
#error "GLOBAL_ROUND_APPLY_BATCH exceeds the GLOBAL_ROUND shared buffer"
#endif
#ifndef GLOBAL_ROUND_DIRECT_STORE
// 在 GLOBAL_ROUND 的 work/injection quiesce 窗口内，接收侧没有并发
// node_data 写者；打开后用普通读写替代每条消息的 atomicMin，作为性能实验。
#define GLOBAL_ROUND_DIRECT_STORE false
#endif
#ifndef GLOBAL_ROUND_FRONTIER
// GLOBAL_ROUND 接收侧把真正改进的条目直接追加到本卡 frontier，由 work warp
// 领取处理，绕过“atomicMin -> dirty -> write_through -> L2DQ”链路。
#define GLOBAL_ROUND_FRONTIER false
#endif
#ifndef GLOBAL_ROUND_PARALLEL_APPLY
// GLOBAL_ROUND 接收侧并行应用实验：由 L2 manager warps 分片消费 round inbox，
// 仅在 work/injection 全部 quiesce 且 frontier 已启用时生效。
#define GLOBAL_ROUND_PARALLEL_APPLY false
#endif
#ifndef GLOBAL_ROUND_MULTI_PACK
// GLOBAL_ROUND_MULTI_PACK：quiesce 窗口内让全部 L2 manager warp + L3 warp
// 协作提取 remote_mark/remote_cand；warp0 manager 负责一次性发布 inbox。
// 默认关闭，保留单 L3 warp 路径作为正确性对照。
#define GLOBAL_ROUND_MULTI_PACK false
#endif
#ifndef GLOBAL_ROUND_CANDIDATE_PREFILTER
// GLOBAL_ROUND 发送侧候选级快筛：只读取本轮已经聚合的候选对应的 peer node_data，
// 不扫描完整对端分区；若候选不小于对端当前权威值，则安全丢弃 stale candidate。
#define GLOBAL_ROUND_CANDIDATE_PREFILTER false
#endif
#ifndef GLOBAL_ROUND_PEER_CACHE_FEEDBACK
// GLOBAL_ROUND 接收侧权威反馈：在 quiesce apply 窗口把本卡当前 node_data 反馈到
// 发送卡的 peer_cache，使后续本地候选快筛拥有对端收敛信息。
#define GLOBAL_ROUND_PEER_CACHE_FEEDBACK false
#endif
#ifndef GLOBAL_ROUND_DENSE_EXCHANGE
// GLOBAL_ROUND dense candidate exchange：发送侧在 quiesce 窗口把完整
// remote_cand 数组写入对端 staging buffer，接收侧按目标顶点区间并行扫描。
// 默认关闭；仅作为高候选密度图的协议实验，不改变生产 sparse inbox 路径。
#define GLOBAL_ROUND_DENSE_EXCHANGE false
#endif
#ifndef GLOBAL_ROUND_DIRECT_P2P
// GLOBAL_ROUND direct P2P：保留发送侧本地 remote_cand 聚合，但在两卡安全点
// 直接把候选 atomicMin 到对端权威 node_data，并置对端 dirty/hint。epoch slot
// 只承载“本轮 P2P 写入已完成”的标记，不再复制 NODE_TYPE inbox 或构造 frontier。
// 默认关闭，作为 sparse inbox/frontier 的架构对照路径。
#define GLOBAL_ROUND_DIRECT_P2P false
#endif
#ifndef GLOBAL_ROUND_DIRECT_TRACE
#define GLOBAL_ROUND_DIRECT_TRACE false
#endif
#ifndef GLOBAL_ROUND_DIRECT_BACKSTOP
// Direct P2P 的保守兜底：每轮 marker 收到后全扫 node_data/last_processed，
// 重新置 dirty。ACK/activity 握手验证完成后可用 -D...=false 测量去掉该全扫
// 的性能与正确性；默认保留，避免未验证构建改变安全路径。
#define GLOBAL_ROUND_DIRECT_BACKSTOP true
#endif
#ifndef GLOBAL_ROUND_DIRECT_EXACT_LIST
// Direct P2P 可选精确恢复列表：发送侧只把 atomicMin 真正胜出的候选写入
// epoch slot，接收侧按列表恢复 dirty，避免依赖整分区 backstop 扫描。
// 默认关闭；仅与 GLOBAL_ROUND_DIRECT_BACKSTOP=false 一起做协议实验。
#define GLOBAL_ROUND_DIRECT_EXACT_LIST false
#endif
#define GLOBAL_ROUND_DIRECT_ACK_ACTIVITY (1 << 30)
#define GLOBAL_ROUND_DIRECT_ACK_EPOCH_MASK (GLOBAL_ROUND_DIRECT_ACK_ACTIVITY - 1)
#if (GLOBAL_ROUND_FRONTIER == true && GLOBAL_ROUND == false)
#error "GLOBAL_ROUND_FRONTIER requires GLOBAL_ROUND"
#endif
#if (GLOBAL_ROUND_PARALLEL_APPLY == true && \
     (GLOBAL_ROUND == false || GLOBAL_ROUND_FRONTIER == false || \
      GLOBAL_ROUND_DIRECT_STORE == false))
#error "GLOBAL_ROUND_PARALLEL_APPLY requires GLOBAL_ROUND + FRONTIER + DIRECT_STORE"
#endif
#if (GLOBAL_ROUND_MULTI_PACK == true && \
     (GLOBAL_ROUND == false || GLOBAL_ROUND_FRONTIER == false || \
      GLOBAL_ROUND_DIRECT_STORE == false || GLOBAL_ROUND_PARALLEL_APPLY == false || \
      GLOBAL_ROUND_DIRECT_P2P == true))
#error "GLOBAL_ROUND_MULTI_PACK requires sparse GLOBAL_ROUND + FRONTIER + DIRECT_STORE + PARALLEL_APPLY"
#endif
#if (GLOBAL_ROUND_DENSE_EXCHANGE == true && \
     (GLOBAL_ROUND == false || GLOBAL_ROUND_FRONTIER == false || \
      GLOBAL_ROUND_DIRECT_STORE == false || GLOBAL_ROUND_PARALLEL_APPLY == false))
#error "GLOBAL_ROUND_DENSE_EXCHANGE requires GLOBAL_ROUND + FRONTIER + DIRECT_STORE + PARALLEL_APPLY"
#endif
#define BULK_FRONTIER_ENABLED \
    ((BULK_EPOCH == true && BULK_FRONTIER == true) || (GLOBAL_ROUND_FRONTIER == true))
#ifndef BULK_DIAG
#define BULK_DIAG false
#endif
#ifndef BULK_DIAG_K
#define BULK_DIAG_K 100000
#endif
#ifndef BULK_WORK_DIAG_K
#define BULK_WORK_DIAG_K 100000
#endif
#ifndef BULK_TRACE
#define BULK_TRACE false
#endif
#ifndef BULK_TRACE_NODE
// 低扰动候选/发送/接收跟踪的目标全局顶点；仅在 BULK_TRACE=true 构建中生效。
#define BULK_TRACE_NODE 191
#endif
#ifndef BULK_NO_CACHE
#define BULK_NO_CACHE false
#endif
#ifndef BULK_CACHE_PREFILTER
// 远程候选生成的安全快筛：先 volatile 读取单调不增的 peer_cache，
// 已有更小/相等值时跳过 atomicMin；未命中仍走原子裁决。默认关闭，作为
// 跨卡热路径实验开关。
#define BULK_CACHE_PREFILTER false
#endif
#ifndef BULK_L3_K
#define BULK_L3_K 1
#endif
#ifndef QUEUE_CROSS_RATIO_THRESHOLD
// 仅用于双卡本地队列选择；不改变 GPU 数量。低 cut 图可用 FIFO 风格
// L1V_L2DQ 减少本地队列重处理，高 cut 图保留 L1SLF_L2DQ。
#define QUEUE_CROSS_RATIO_THRESHOLD 0.0007
#endif

// 解法0 真机落地（SESSION 26 前置实验锁定方向）: phase 化 v2 + seed barrier。
// 源卡（含 src）本地收敛 → phase=2 灌值期 L3 只写 peer node_data（不置 dirty）→
// 全收敛+mark 全空 → fence → P2P 置接收卡 seed_ready → 接收卡见 seed_ready 才启动
// backstop/注入（一次性处理全部最终值，达到 SEED_EXP 的近最优 work，消除增量注入重处理）。
// 默认关闭：通用双卡正确性不再依赖源卡自足（P2）假设。
#ifndef SEED_BARRIER
#define SEED_BARRIER false
#endif
#if (GLOBAL_ROUND == true && SEED_BARRIER == true)
#error "GLOBAL_ROUND and SEED_BARRIER are mutually exclusive"
#endif
#if (GLOBAL_ROUND_DIRECT_P2P == true && \
     (GLOBAL_ROUND == false || BULK_ROUND == false || SEED_BARRIER == true || \
      GLOBAL_ROUND_FRONTIER == true || GLOBAL_ROUND_PARALLEL_APPLY == true || \
      GLOBAL_ROUND_DENSE_EXCHANGE == true))
#error "GLOBAL_ROUND_DIRECT_P2P requires sparse GLOBAL_ROUND without frontier/apply/dense"
#endif

// SESSION 31 诊断: work kernel 活跃处理时钟（warp 内 clock 差分，跨 kernel 不可比但
//   同 warp 内可靠）——测 gpu1 真机 vs SEED_EXP 的处理时间差。默认 false。
#define WORK_SEG_PROFILE false

// phaseI 方案 B（异步反馈环）: 在 SEED_BARRIER 之上放开反向 flush，让 P2 违规图的
// 反向贡献异步回流（gpu1 改进批量 flush 回 gpu0，gpu0 增量重处理，必要时再回流），
// 终止改为"双方 idle && 无真实跨卡改进落位（quiet 窗口）"。通用（不依赖图重排/P2）。
// 默认 false；通用 BULK_ROUND 不使用渐进式异步反馈。
#ifndef ASYNC_FB
#define ASYNC_FB false
#endif

// phaseI 方案 C（ghost 顶点最小可行版）: 源卡持有对端边界顶点的 ghost 副本（D 跳闭包），
// relax_dst 命中 ghost → 本卡原子更新 ghost 距离 + 置 ghost_mark → 注入 warp BF 式松弛
// ghost 出边（发现"穿到对端绕回本卡"的路径），SEED_BARRIER 灌值/屏障语义保留。
// GHOST_DEPTH: ghost 闭包跳数（0=关闭 C；1=边界顶点及其出边；2=闭包扩展一跳）。默认 0。
#define GHOST_DEPTH 0
// 方案 C: ghost BF 松弛最大深度（ghost 间边链）与每层缓冲容量（per 注入 warp，rshm 区）
#define GHOST_BF_DEPTH 16
#define GHOST_BF_CAP 128

// 阶段 E2: 注入多 warp 并行化（warp0 保留终止+backstop+slice0，其余 N 个 warp 各扫一个 dirty 区间）
// 注：寄存器预算受线程数约束（maxThreadsPerBlock）。MANAGE_PROFILE/HANG_DIAG 开 printf 会加寄存器，
//     需降 N 才能启动。默认 768 线程(N=6) 需 manage kernel ≤85 寄存器。
#ifndef INJECT_WARP_NUM
#if (HANG_DIAG == true)
#define INJECT_WARP_NUM 6
#else
#define INJECT_WARP_NUM 6
#endif
#endif

// phase2 实验：源卡 phase2 灌值由 L3 warp + 注入 warp 分片消费 remote_mark，
// 避免单 warp 扫描和逐候选 P2P seed_list atomicAdd。旧路径保持可回退。
#ifndef SEED_PHASE2_PARALLEL
#define SEED_PHASE2_PARALLEL true
#endif

// phase2 去冗余：P2 成立时只发布紧凑 seed_list；接收卡在解除 work gate
// 前把 seed_list 物化到本卡 node_data，避免源卡对 peer node_data 的重复随机写。
// 关闭时恢复“源卡写 peer node_data + seed_list”双写路径。
#ifndef SEED_PHASE2_LIST_ONLY
#define SEED_PHASE2_LIST_ONLY true
#endif

#define EDGE_WISE false

#define LARGEV 16

class node_struct
{
public:
    int id = 0;
#if (USE_DIST_IN_STRUCT == true)
    VALUE_TYPE dist;
#endif

    __device__ bool operator<(const node_struct& b);
    __device__ bool operator>(const node_struct& b);
    __device__ bool operator<=(const node_struct& b);
    __device__ bool operator>=(const node_struct& b);

    __host__ __device__ node_struct& operator=(const int id_in);

    __device__ bool filter();

    __device__ VALUE_TYPE get_data();

    __host__ __device__ node_struct(int id_in, VALUE_TYPE dist_in);

    __host__ __device__ node_struct();
};

#define NODE_TYPE node_struct

#ifndef L3_DIRECT_RX
#define L3_DIRECT_RX false
#endif
#ifndef L3_RETAIN_TX
#define L3_RETAIN_TX false
#endif
#ifndef L3_WINDOW_MODE
#define L3_WINDOW_MODE 0
#endif
#ifndef L3_COOPERATIVE_COLLECT
#define L3_COOPERATIVE_COLLECT false
#endif
#ifndef L3_QUIET_DRAIN
#define L3_QUIET_DRAIN false
#endif
#ifndef L3_BATCH_ORDER
#define L3_BATCH_ORDER false
#endif
#ifndef L3_LOCAL_SETTLE
#define L3_LOCAL_SETTLE false
#endif
#ifndef L3_SETTLE_CYCLES
#define L3_SETTLE_CYCLES 10000000ull
#endif
#ifndef L3_WINDOW_MIN_CYCLES
#define L3_WINDOW_MIN_CYCLES 25000ull
#endif
#if (L3_LOCAL_SETTLE == true && (L3_WINDOW_MODE != 2 || L3_SETTLE_CYCLES < L3_WINDOW_MIN_CYCLES))
#error "Local settle requires dynamic window mode and a bounded positive delay"
#endif
#if (L3_BATCH_ORDER == true && (L3_RETAIN_TX == false || (BULK_L3_BATCH & (BULK_L3_BATCH - 1)) != 0))
#error "Batch order requires retained TX and power-of-two lane batch capacity"
#endif
#if (L3_QUIET_DRAIN == true && L3_WINDOW_MODE == 0)
#error "Quiet drain requires the window data plane"
#endif
#if (L3_COOPERATIVE_COLLECT == true && (L3_WINDOW_MODE == 0 || BULK_L3_BATCH < 32))
#error "Cooperative collect requires window fullscan cadence and capacity >=1024"
#endif
#ifndef L3_WINDOW_MAX_CYCLES
// L3 window backoff doubles the scan interval on unproductive scans.  With the
// boundary index an empty scan only touches the static remote-candidate domain,
// so backoff buys nothing but adds cross-GPU candidate latency.  Cap it at the
// minimum cooldown whenever the boundary index makes scans cheap.
#if (L3_BOUNDARY_INDEX == true)
#define L3_WINDOW_MAX_CYCLES 25000ull
#else
#define L3_WINDOW_MAX_CYCLES 100000ull
#endif
#endif
#ifndef L3_RX_FEEDBACK_MODE
#define L3_RX_FEEDBACK_MODE 0
#endif
#ifndef L3_PROGRESS_DIAG
#define L3_PROGRESS_DIAG false
#endif
#ifndef L3_RECOVERY_MODE
#define L3_RECOVERY_MODE 0
#endif
#ifndef L3_RECOVERY_TEST_DELAY
#define L3_RECOVERY_TEST_DELAY false
#endif
#if (L3_RECOVERY_MODE < 0 || L3_RECOVERY_MODE > 2)
#error "Recovery modes: 0 legacy, 1 epoch with injection helpers, 2 epoch with L2 helpers"
#endif
#if (L3_RECOVERY_MODE > 0 && (L3_DIRECT_RX == false || L3_RETAIN_TX == false || L3_TERM_HANDSHAKE == false))
#error "Epoch recovery requires direct RX, retained TX and the default termination handshake"
#endif
#if (L3_RECOVERY_TEST_DELAY == true && L3_RECOVERY_MODE == 0)
#error "Recovery delay is an epoch-protocol test only"
#endif
#if (L3_PROGRESS_DIAG == true && L3_DIRECT_RX == false)
#error "Progress timestamps require the single-manager direct RX path"
#endif
#ifndef L3_EVENT_GATE
#define L3_EVENT_GATE false
#endif
#ifndef L3_EVENT_FULL_CYCLES
#define L3_EVENT_FULL_CYCLES (256ull * L3_WINDOW_MAX_CYCLES)
#endif
#if (L3_EVENT_GATE == true && (L3_WINDOW_MODE != 2 || L3_LOCAL_SETTLE == true || L3_EVENT_FULL_CYCLES < L3_WINDOW_MAX_CYCLES))
#error "Event gate requires dynamic window and an independent bounded fullscan timer"
#endif
#ifndef L3_RX_FEEDBACK_TRACE
#define L3_RX_FEEDBACK_TRACE false
#endif
#if (L3_RX_FEEDBACK_MODE < 0 || L3_RX_FEEDBACK_MODE > 2)
#error "RX feedback modes: 0 off, 1 observe, 2 control"
#endif
#if (L3_RX_FEEDBACK_MODE > 0 && (L3_WINDOW_MODE != 2 || L3_DIRECT_RX == false || L3_RETAIN_TX == false || L3_LOCAL_SETTLE == true))
#error "RX feedback requires the dynamic window, direct RX and retained TX"
#endif
#if (L3_RX_FEEDBACK_TRACE == true && L3_RX_FEEDBACK_MODE == 0)
#error "RX feedback trace requires feedback transport"
#endif
#if (L3_WINDOW_MODE < 0 || L3_WINDOW_MODE > 2 || L3_WINDOW_MIN_CYCLES == 0 || L3_WINDOW_MAX_CYCLES < L3_WINDOW_MIN_CYCLES)
#error "Invalid L3 window configuration"
#endif
#if (L3_WINDOW_MODE != 0 && (L3_RETAIN_TX == false || L3_DIRECT_RX == false || L3_DYNAMIC == true))
#error "Window modes require retained TX, direct RX, and old L3_DYNAMIC disabled"
#endif
#if (L3_DIRECT_RX == true || L3_RETAIN_TX == true)
#if (BULK_ROUND == false || BULK_EPOCH == true || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || SEED_BARRIER == true)
#error "L3_DIRECT_RX/L3_RETAIN_TX require the default BULK data plane"
#endif
#endif

#if (L3_TILE_LOAN == true)
#if (BULK_ROUND == false || L3_DIRECT_RX == false || L3_RETAIN_TX == false || \
     BULK_EPOCH == true || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || \
     SEED_BARRIER == true || L3_RX_EXPRESS == true || L3_RX_L2_PULL == true || \
     L3_WORKER_RECOVERY == true || L3_RECOVERY_MODE > 0 || GHOST_DEPTH > 0)
#error "L3_TILE_LOAN requires the default BULK/DIRECT_RX/RETAIN_TX path and no competing experimental protocol"
#endif
#if (L3_TILE_LOAN_MAX_SEEDS < 1 || L3_TILE_LOAN_MAX_SEEDS > node_size)
#error "L3_TILE_LOAN_MAX_SEEDS must fit the existing work node_in buffer"
#endif
#if ((L3_TILE_LOAN_MAX_SEEDS % l2_batch_size) != 0)
#error "L3_TILE_LOAN_MAX_SEEDS must be a multiple of the fixed q2 read batch"
#endif
#if (L3_TILE_LOAN_RESULT_CAP < 1)
#error "L3_TILE_LOAN_RESULT_CAP must be positive"
#endif
#if (L3_OWNER_COMMIT == true && L3_TILE_LOAN_RESULT_CAP < 1)
#error "L3_OWNER_COMMIT requires a positive source edge admission cap"
#endif
#endif

#if (L3_RX_EXPRESS == true)
#if (L3_DIRECT_RX == false || BULK_ROUND == false || BULK_EPOCH == true || \
     GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || \
     GLOBAL_ROUND_ASYNC_RX_FRONTIER == true || \
     GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true || SEED_BARRIER == true)
#error "L3_RX_EXPRESS requires the default BULK/DIRECT_RX receive path"
#endif

#if (L3_ADMISSION_BUDGET == true || L3_WORKER_RECOVERY == true || \
     L3_RX_PRIORITY_BOOTSTRAP == true || GHOST_DEPTH > 0 || L3_RECOVERY_MODE > 0)
#error "L3_RX_EXPRESS is isolated from admission/recovery/bootstrap/ghost variants"
#endif

#if (L3_RX_EXPRESS_SLOTS < 2 || (L3_RX_EXPRESS_SLOTS & (L3_RX_EXPRESS_SLOTS - 1)) != 0)
#error "L3_RX_EXPRESS_SLOTS must be a power of two and at least two"
#endif
#if (L3_RX_EXPRESS_BATCH < 1 || L3_RX_EXPRESS_BATCH > 32)
#error "L3_RX_EXPRESS_BATCH must be in [1, 32]"
#endif
#endif

#if (L3_RX_L2_PULL == true)
#if (L3_DIRECT_RX == false || BULK_ROUND == false || BULK_EPOCH == true || \
     GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || \
     GLOBAL_ROUND_ASYNC_RX_FRONTIER == true || \
     GLOBAL_ROUND_ASYNC_RX_MANAGER_FRONTIER == true || SEED_BARRIER == true || \
     L3_RX_EXPRESS == true || L3_ADMISSION_BUDGET == true || \
     L3_WORKER_RECOVERY == true || L3_RX_PRIORITY_BOOTSTRAP == true || \
     L3_RECOVERY_MODE > 0 || GHOST_DEPTH > 0 || L3_DYNAMIC == true)
#error "L3_RX_L2_PULL requires the default BULK/DIRECT_RX path"
#endif
#if (MLMQ_TYPE != L1SLF_L2DQ)
#error "L3_RX_L2_PULL is currently restricted to L1SLF_L2DQ"
#endif
#endif

#if (L3_LOCAL_YIELD_BATCHES > 0)
#if (MLMQ_TYPE != L1SLF_L2DQ || ACCESS_THROUGH == true || BULK_ROUND == false || \
     L3_DIRECT_RX == false || GLOBAL_ROUND == true || GLOBAL_ROUND_ASYNC == true || \
     BULK_EPOCH == true || SEED_BARRIER == true || GHOST_DEPTH > 0 || \
     L3_TILE_LOAN == true || L3_RX_EXPRESS == true || L3_RX_L2_PULL == true || \
     L3_ADMISSION_BUDGET == true || L3_WORKER_RECOVERY == true || L3_DYNAMIC == true)
#error "local yield is isolated to clean SLF/DQ BULK direct RX"
#endif
#endif
