# TPDS L3 复审问题：基于当前源码的逐项答复与改稿依据

核查日期：2026-09-21。对应文档：[TPDS_L3_review_and_revision_plan.md](TPDS_L3_review_and_revision_plan.md)。

本文用于论文修改参考。结论来自当前源码、冻结源码包、现有实验原始输出和本轮执行的主机端 shortcut 检查。本轮未修改求解器、原审阅文档或论文，也未提交 GPU 作业。对稿件章节及图 10 的定位沿用审阅文档；本文没有独立逐页复核其引用的 `TPDS (8).pdf`。

## 0. 核查对象和最重要的结论

当前分支为 `para-frame3-multisource`，Git HEAD 为 `b842b8a`。工作区包含未提交改动，因此不能只用 HEAD 标识已测实现。

本次主要核查 [L3_VERSION.md](L3_VERSION.md) 标记的 `L3-ROAD-CHAIN-WIN25-W512-CUT-20260916`：

- 冻结二进制：`tmp/base_today/l3road_w512_frozen/mlmq`，SHA256 为 `9731384959c0e39b8f860b5f37677f8b03272cb31fffe109f8923d4c1d2b43ac`。
- 同批三方实验使用的 `tmp/base_today/prod_nogate512/mlmq` 与上述二进制哈希一致。
- 冻结源码：`tmp/base_today/l3road_w512_frozen/source.tgz`，SHA256 为 `acbfee83bab62f991c73f29a2a59dfeb20668bf4a14ff0969a259ab3e4f1f8aa`。本轮逐文件复核，74 个归档源码文件与当前工作区对应文件完全一致。
- 单卡无 L3 基线：`tmp/base_today/nol3/mlmq`，SHA256 为 `2eca02408fa819b061063f106bfa781c3a0e34a74555ec6e51f3a800bb344d5b`。
- ADDS 对照：`tmp/adds_official_v4/adds`，SHA256 为 `511e88f56762914829a055aea0215a6c8e3c0d1dfc1c078f5f2247a559638ec2`。其冻结包中的 `ADDS/official_bench.cu` 与当前对应文件相同。

**总体判断：审阅计划指出的两个核心问题成立，但解决程度不同。异步传播的大部分责任交接已经有具体实现，可以据此补写论文；“收益独立来自 L3”则仍不能由现有完整配置对比推出。** 此外，源码核查暴露出需要单独处理的容量、内存模型和统计来源问题，不能全部通过补文字关闭。

| 编号 | 根据源码可以回答的部分 | 仍不能宣称已经关闭的部分 |
|---|---|---|
| L3-01 | 双卡在主机端构造增强 CSR；单卡对照不加 shortcut | 本次核查的 W512/cut 归档未提供完整、匹配图表示的四配置消融 |
| L3-02 | 实际顺序是先领取并清 mark，再交换取得 candidate；并发新值可以被本批带走或留给后续批次 | 需要把跨地址顺序约束写进证明；现有静态收集测试不等于并发交错验证 |
| L3-03 | retained journal、两槽 generation/epoch、发布前背压、接收提交后 ACK 均有代码 | P2P 控制字段的原子作用域与读取顺序需要正式核查；不支持任意故障恢复的宽泛结论 |
| L3-04 | 接收严格改进进入接收暂存区，再提交 L2；实际松弛读取 owner 距离 | 当前 DQ 没有容量满检测/失败重试，不能写成有界队列背压保证 |
| L3-05 | 有带 token 的冻结、worker/injection/TX 确认、重新核查、恢复和双边 READY | 有限存储、计数器范围、内存可见性、内核驻留仍是进展论证的前提 |
| L3-06 | 严格 shortcut 条件、方向、原边保留、保距论证均可明确；本轮 CPU 测试通过 | GPU 路径和与距离加法的整数范围不能由主机构造检查替代 |
| L3-07 | 能补出设备拓扑记录、分区、源点、窗口、批次、角色及真实计时边界 | 编译 manifest 不完整，部分硬件字段/实际容量仍应补齐归档 |
| L3-08 | 现有结果支持完整双卡配置的 solve-time 改善 | 图表来源说明与数据不一致；最新同批结果也不是独立 L3 消融 |

本文后面使用三种证据层级：**源码事实**、**在明确前提下的推论**、**实验记录支持**。三者不能互相替代。

## 1. 先统一论文中的执行模型

### 1.1 L3 是什么，如何接到原来的 MLMQ

可以将每张卡的 L0/L1/L2 及其正在执行的任务抽象为本地处理系统 `Q_p`。真实调用链是：

```text
work_block_kernel
  -> mlmq.read
  -> simple_process
  -> relax_dst
       本地目标：atomicMin(owner distance) -> node_out -> mlmq.write -> 本地 Q_p
       远端目标：peer_cache 过滤 -> remote_cand/remote_mark
                   -> TX journal -> peer inbox
                   -> RX atomicMin(owner distance) -> write_through -> owner L2
```

源码定位：`SSSP/sssp_run.cu:3491`、`:2282`、`:2062`；`core/src/ml_queue.cu:225`、`:323`、`:359`。

这里的 L3 **不是跨两卡统一出队的全局优先队列**。当前发送批次也不承诺按距离全局排序；`L3_BATCH_ORDER` 默认关闭。它是拥有者划分之上的候选聚合、传输、接收激活和完成协议。

每卡拥有一个连续顶点区间及这些顶点的出边 CSR，列索引仍指向全局顶点。权威距离由顶点 owner 维护；普通远端松弛不把远端顶点加入发送方的本地队列，也不直接在该路径上修改对端权威距离。源码中存在其他实验路径，不能把那些路径与当前配置拼成一个算法。

源码定位：`SSSP/graph_partition.h:107`；`SSSP/sssp_run.cu:2079`、`:2111`、`:2174`。

### 1.2 逻辑角色和实际并发角色

| 逻辑角色 | 当前实现位置 | 主要职责 |
|---|---|---|
| work warps | 每卡 `work_block_kernel` | 读本地队列、展开边、产生本地/远端更新 |
| block local manager | 每个 work block 额外的一个 warp | 更新本 block 的队列本地信息 |
| manager/RX | 每卡 `manage_block_kernel` 的 warp 0 | 接收批次、提交有效激活、处理恢复和终止 |
| L2 managers | 同一个 manage kernel 的若干 warps | 推进 DQ 各桶的可读状态 |
| TX | 同一个 manage kernel 的专用 warp | 收集候选、保留 journal、发布到对端 inbox |
| injection | 同一个 manage kernel 的 6 个 warps | 将恢复等路径留下的 dirty 工作注入 L2 |

`MLMQ_WORKER_THREADS=512` 指每个 work block 的 512 个工作线程，即 16 个 work warps；实际 launch 加上一个本地管理 warp，为 **544 threads/block**，不能在复现实验表中直接写成 512 threads/block。

`work_blocks=107` 时每卡有 1,712 个 work warps。manage kernel 的线程数为 `32 × (1 + M + 1 + 6)`，其中 `M=mlmq.manage_warp_num()`；DQ 返回 bucket 数，16 桶时为 24 warps，即 768 threads。

源码定位：`core/include/GPU_setup.h:3`；`SSSP/sssp_run.cu:2940`、`:2953`、`:11144`、`:11225`；`core/cu_delta_queue/cu_delta_queue.cuh:219`。

当前性能配置启用了 direct RX，因此主接收路径是 **manager warp 直接提交 L2**。不要沿用历史默认路径“收到消息只置 dirty，然后由 injection 完成正常接收”的叙述；dirty/injection 在当前配置中仍用于恢复等职责。

## 2. L3-01：完整双卡配置的收益，不能直接等同于 L3 的独立收益

### 2.1 源码明确做了图增强

`SSSP/main.cu:474` 设置 `l3_chain_active = (n_gpu > 1)`；`:483` 在主机端调用 `chain_view.build(...)`；`:487` 将新数组组成 `chain_graph`；`:591` 用它构造双卡分区。

这说明：

1. shortcut 的确已集成进多卡执行入口，由多卡条件控制启用；
2. 实现手段仍然是求解前的主机端 CSR 增强；
3. 新增边由本地 work warps 通过原来的边遍历路径使用，改变传播路径和实际处理工作；
4. `n_gpu>1` 这个开关本身不能建立收益归因。

`main.cu:470` 附近注释中“仅双卡构造，所以收益可归因 L3”的推断不成立。[L3_VERSION.md](L3_VERSION.md) 中相同含义的表述也应在论文准备阶段修正。**可以称它为 L3 扩展配置中的图优化，但不能因此证明它是跨 GPU 通信机制带来的加速。**

### 2.2 应怎样完成匹配输入对照

设经过既有顶点重排后的图为 `G`，固定双卡分区 `π` 和构造源点 `s` 后，当前增强图为：

\[
G^+ = F(G,\pi,s).
\]

把分区和源点写入定义是必要的：eligible 条件排除 source，并要求链内部和两端不跨 owner。

| 配置 | GPU 数 | 实际求解图 | 用途 |
|---|---:|---|---|
| M1-original | 1 | `G` | 原始单卡基线 |
| M1-shortcut | 1 | 固定的 `G+` | 测量相同图增强对单卡的收益 |
| M2-original | 2 | `G` | 固定原图表示下的双卡扩展 |
| M2-shortcut | 2 | 与 M1-shortcut 完全相同的 `G+` | 当前完整配置及增强图上的扩展 |

当前二进制简单运行 `-n 1` 不会自动得到 M1-shortcut，因为该条件直接关闭构造。应通过固定增强 CSR 输入或明确的测试适配入口完成这项实验，并核查顶点/边/权重/源点映射。不能让单卡按自身分区重新构造一个不同的增强图。

这些对照还应固定或明确披露 L0–L2 改动、worker 几何、队列参数及编译优化；不能只固定图文件，却将另一项本地执行优化的差异混入设备数量对比。

在相同统计方式下，分别报告：

\[
S_{\rm original}=T_{1,G}/T_{2,G},\qquad
S_{\rm shortcut}=T_{1,G^+}/T_{2,G^+},\qquad
S_{\rm complete}=T_{1,G}/T_{2,G^+}.
\]

此外，`T1,G / T1,G+` 反映 shortcut 的单卡收益。完整配置的比值可分解为：

\[
S_{\rm complete}=(T_{1,G}/T_{1,G^+})\,(T_{1,G^+}/T_{2,G^+}).
\]

因此完整配置超过 1，并不要求同图双卡扩展也超过 1。进一步归因窗口、收集、接收提交等 L3 子机制时，还需要**固定双卡数、图表示和分区的逐项消融**；四配置实验本身也不是全部通信机制的因果分解。

### 2.3 不能忽略已有的历史同图对照

[224 阶段记录](knowledgebase/l3_dynamic/224_idle_token_probe.md) 中的作业 29054 已包含“原始单卡、同捷径单卡、双卡”八图对照。记录中的同捷径单卡/双卡比值为：NY 0.963、BAY 0.982、COL 1.041、FLA 0.982、CAL 0.995、E 1.023、W 0.855、USA 1.085。

这份历史证据说明审阅中的归因疑问有实际依据。它不能直接代表当前 W512/cut 版本，也不能与后来的双卡耗时拼成一个新的加速比。当前版本的匹配输入复验仍然必要。

### 2.4 ADDS 对照和论文定位

ADDS 在 `G` 上与双卡 MLMQ 在 `G+` 上比较，可以作为**完整系统配置的稳态 solve-time 对照**，前提是明确各自图表示和计时边界。若称“端到端”，则还必须把图增强、初始化等实际成本计入，现有 solve 比值不够。

不必为了保留系统对照立即改动 ADDS；优先补齐 MLMQ 四配置即可回答最关键的归因问题。论文主张宜分成两层：L3 提供多卡传播与完成机制；shortcut、worker 几何和分区参数构成已测完整配置的优化组合。

## 3. L3-02：候选与 mark 的交接，实际怎样避免旧快照清掉新更新

### 3.1 各状态的真实含义

| 抽象状态 | 源码对象 | 含义和边界 |
|---|---|---|
| owner 距离 `D_p[v]` | `node_data` | owner 上的权威距离 |
| 发送方历史过滤值 `C_pq[v]` | `peer_cache` | 发送方已接受的较优候选阈值；不是对端距离的精确副本，也不等于已 ACK 值 |
| 可领取候选 `R_pq[v]` | `remote_cand` | 尚未转交的候选最小值，领取时交换回 `DIST_MAX` |
| 领取标记 `M_pq[v]` | `remote_mark` | 候选是否需要发送扫描的权威位图 |
| 扫描提示 | `mark_hint/mark_hint2` | 降低扫描代价，不能代替权威位图 |
| 改进事件计数 | `g_bulk_mark_signal` | 通过前置过滤且严格改进 candidate 数组的事件数，不是不同顶点数 |

源码定位：`SSSP/sssp_run.cu:2130`、`:2153`；`SSSP/l3/l3_candidate.cuh:13`。

普通路径先尝试 `atomicMin(peer_cache, new_dist)`，胜出才调用 `l3_record_candidate`。该函数先 `atomicMin(remote_cand, new_dist)`，严格胜出后再 `atomicOr(remote_mark, bit)`，然后更新提示、signal 和本地 idle 状态。

因此 peer cache 的安全性依赖后续责任链：某个值一旦降低了 cache，必须仍有正在执行的 producer、pending candidate、journal、inbox 或 owner 工作承接它。不能说“cache 小于新值，所以 owner 一定已经处理过”。

### 3.2 关键顺序是先清 mark，再取值

权威扫描分支的实际操作为：

```text
CAS(mark_word, observed_bits, 0)       // 先领取一个位图字
for each claimed bit:
    value = atomicExch(candidate[v], INF)
    append (v, value) to journal
```

源码定位：`SSSP/sssp_run.cu:9447`、`:9449`、`:9457`。协作收集路径 `SSSP/l3/l3_collect.cuh` 采用相同的清标记后取值原则。

这与审阅中“先读 10，再产生 4，最后清 mark”的朴素错误顺序不同。**在所需原子操作顺序得到保证的前提下**，可以分三种情况解释：

| 更优值 4 到达的时机 | 后续行为 |
|---|---|
| 清 mark 之前 | 后面的 candidate 交换取得 4，当前批次覆盖更新 |
| 清 mark 之后、candidate 交换之前 | 本批可以取得 4；producer 的重新置位可能留下一个冗余 mark |
| candidate 交换之后 | 4 写入已重置的 candidate，并重新置位，留待下一批 |

若 producer 的 candidate 更新已发生，但 mark 写入较晚，本批可能已经取走该值，之后留下空 candidate 对应的冗余 mark。后续扫描读到 `DIST_MAX` 可以清理这一冗余状态。协议允许重复发现和冗余标记，关键是必要更新不能消失。

位图字上其他 bit 的并发变化会使 CAS 失败，发送方下轮重试。字已领取但批容量耗尽时，未处理 bit 会 OR 回；已取值而不能写入 journal 的路径会将值以 `atomicMin` 放回并重新置位。不能把一次领取动作描述成“无条件清空全部 pending”。

源码定位：`SSSP/sssp_run.cu:9462` 至 `:9480`。

### 3.3 boundary index 的正确定位

当前 boundary index 是**静态的远端目标 mark-word 索引**。主机构造时检查本地 CSR 出边，记录可能指向的对端位图字；较密时退回连续扫描。当前配置没有开启 compact-candidate 存储，因此 candidate/cache 仍按对端顶点域分配。

它减少的是扫描地址范围，不能直接写成“所有 L3 数据结构只为边界顶点分配”。严格 owner-local shortcut 不增加跨 owner 边，因此不会破坏这个静态候选域的覆盖关系。若将来启用能产生新远端目标的其他路径，则必须重新审核这一前提。

源码定位：`SSSP/l3/l3_boundary_index.cuh:64`、`:80`、`:91`、`:97`。

hint 只是加速结构。发送路径保留周期性权威扫描；当前常量 `L3_FULL=256` 表示收集轮次周期，不是固定的墙钟时限。终止确认阶段还会检查权威 mark。

### 3.4 25,000 cycles 的准确含义

`l3_window_state` 初始 budget 为 25,000。动态扫描完成后根据事件数和本次收集数调整 budget，并夹在 `[25000, maximum]`。但当前边界索引配置的 `maximum` 也为 25,000，所以**当前普通窗口实际固定在 25,000 cycles**。

源码定位：`SSSP/l3/l3_window.h:11`、`:30`、`:66`；`SSSP/sssp.cuh:769`。

具体解释应写成：

- 计时来自 TX 所在 GPU 上该执行角色读取的 `clock64()`，不比较两卡的时钟值；
- 首次调用决策函数时建立窗口起点；收集扫描完成后重置起点，包括收集到零条目的扫描；
- `decision()` 先检查到期 SCAN，再检查 PROBE。budget 为 25,000 时，在普通路径中到达 PROBE 门槛就已满足 SCAN，因此 PROBE 不会成为独立的更早触发；
- `new mark event` 统计 candidate 严格改进事件，mark 已经为 1 时也可能增加，不是仅统计 0→1；
- 当前另有终止检查反馈 `l3_drain_requested`，可以要求 TX 排空候选；这是普通窗口之外的进展机制；
- 发送方持有未发布 journal 时先重试它，新产生的候选继续积累在共享 candidate/mark 中。

调用点：`SSSP/sssp_run.cu:9105`、`:9231`、`:9248`、`:9267`、`:9729`。

因此论文不宜直接写“每有新事件立即发送”或“自适应选择不同长度的窗口”。历史记录中“候选延迟 ≤18.5 μs”的说法也不能作为硬界：即使按假设频率换算得到 25,000 cycles 的时间，调度、轮询、收集、等待槽位以及 owner 激活都可能增加实际延迟。

### 3.5 这里仍有一个形式化证明前提

上述交错分析是在所需跨地址顺序成立时的算法论证。当前 candidate/mark 使用普通 CUDA 原子，代码不能仅凭两个操作的源码排列就宣称获得了完整的跨地址同步证明。NVIDIA 的 CUDA 12.4 文档将这些原子描述为 relaxed ordering；原子作用域和内存排序需要分别处理。[CUDA 12.4 原子操作说明](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#atomic-functions)

应核查 producer 的“值先于标记”和 consumer 的“领取先于取值”在所用内存模型下的保证，再写无丢失证明。本轮没有观察到由此造成的 GPU 失败，也没有执行该交错的新 GPU 实验；这属于必须交代的证明边界。

## 4. L3-03：journal、发布、接收和 ACK 的生命周期

### 4.1 journal 是值快照，发布后即可释放本地副本

TX 的 `l3_lidx[]/l3_nd[]` 是 manage block 中的共享数组，容量为 `32 × BULK_L3_BATCH = 4096` 条；只有 TX 角色按协议使用它们。它们保存已领取的值，**不是对共享 candidate 数组的引用**。

收集前设置 `l3_busy`，因此清 mark 到写入 journal 之间的执行也受 busy 责任覆盖。若不能取得对端可写槽位，设置 `retained_count`，保持该 journal，并在下一轮收集前先重试。当前 retained 分支也位于 TX 终止确认安全点之前。

源码定位：`SSSP/sssp_run.cu:9012`、`:9089`、`:9105`、`:9142`、`:9389`、`:9844`。

成功发布后，发送方记录最新 published epoch，清理本地 retained 状态，允许收集下一批。**本地 journal 不需要一直保留到 ACK；未完成责任已经转移到对端槽位及未确认 epoch。** 但接收槽位必须等接收提交完成且满足复用检查才能覆盖。

### 4.2 槽位与 epoch

每方向有两个 inbox 槽位，`slot = epoch % 2`。每槽维护 state、generation、published epoch、count、ACK。发送 epoch 和终止 token 是两套不同的序号，不应混写。

实际主要状态序列是：

```text
FREE -> WRITING -> READY -> READING -> DONE -> FREE/WRITING
```

源码定位：`SSSP/l3/l3_transport.cuh:33`、`:38`、`:70`、`:92`；`SSSP/sssp.cuh:406`。

准备发送 epoch `e` 时，槽位若为 FREE 可以使用；否则要求 DONE，并检查其旧 generation 不晚于 `e-2`、ACK 已覆盖该旧轮次。接收方只领取恰好等于 `rx_epoch+1` 的 READY 槽，且 generation 和 published epoch 都要吻合，再将其 CAS 为 READING。

这套代码依赖每方向只有一个 TX producer，并非可直接推广到多个发送者竞争同一槽的通用 MPMC 队列。原子调用出现于状态机中，不等于状态机自动支持任意生产者数量。

### 4.3 发布顺序和可重试失败的范围

`bulk_publish_l3_batch` 的主要顺序：

1. 取得可写槽，进入 WRITING，并设置 generation；
2. warp 各 lane 写 payload；
3. 同步 warp，每个写 payload 的 lane 执行 system fence；
4. lane 0 发布 count，再 fence；
5. 发布 epoch，再 fence；
6. 发布 READY，再 fence。

源码定位：`SSSP/l3/l3_bulk.cuh:433`、`:476`、`:493`、`:497`、`:500`、`:503`。

正常的 `false` 返回发生在无法取得槽位等发布前条件下；故障注入的 publish retry 也模拟有限的发布前失败。当前机制没有实现“任意一半 payload 写完后设备故障，再自动恢复”的事务协议。论文宜称**对槽位暂不可用的保留和重试**，不要宽泛称为容错传输。

接收端会对 READY 的观察做 warp leader 读取和广播，避免不同 lane 看到不同状态后进入不一致分支。这个细节与实际 CUDA warp 控制流有关，可以放在补充材料。

### 4.4 ACK 确认的是接收责任已转入本地求解

当前 direct RX 只有在本批全部必要激活提交 L2 后，才调用 `bulk_inbox_finish_read`。该函数按 fence → ACK → fence → DONE → fence 的顺序结束接收。

ACK 表示该批记录已被 owner 处理，胜出的更新已交给本地求解责任；它不表示这些顶点的所有出边已经松弛完，更不表示后续跨卡传播已经结束。

两个槽位为已发布批次提供有限并发。槽位忙时 TX 保留自己的批次，owner 的 RX 和本地 workers 继续推进。总传播状态还包括共享 candidate，因此“4096 条 journal”不是系统全部未完成更新的容量上限。

### 4.5 P2P 原子和可见性需要补充审核

代码的 peer 状态访问使用普通 `atomicAdd/atomicExch/atomicCAS`，并配合 `__threadfence_system()`；当前初始化检查并启用 peer access，但未见对应的 native peer atomic 能力检查。

不能直接把“使用 system fence”改写成“所有跨卡控制访问均为 system-scope acquire/release 原子”。CUDA 12.4 文档区分无后缀 device-scope 原子与 system-scope 原子；fence 的作用也不等于扩大原子访问本身的作用域。[CUDA 12.4 原子操作与作用域](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#atomic-functions)

应逐字段核查 state/generation/epoch/count/ACK 的发布、读取、原子范围及目标硬件要求。现有双 A100/NVLink 实验提供了运行证据，但不能代替面向任意 CUDA P2P 平台的内存模型证明。本文将其列为待核查项，没有将它判作已复现错误。

## 5. L3-04：owner 距离改进后，谁保证后续传播

### 5.1 严格胜出的接收记录生成激活

direct RX 在 owner 上对每条记录执行：

```text
won = candidate_value < atomicMin(owner_distance[v], candidate_value)
```

只有严格胜出的 lane 进入 winners ballot；其 `(vertex, value)` 压缩到 manager/RX 的暂存 journal。当前 `L3_RX_COMMIT_BATCH=32`，累积批次和最后不足一批的尾部都调用 `queue.write_through()`。ACK 位于所有这些调用之后。

源码定位：`SSSP/l3/l3_receive.cuh:88`、`:98`、`:149`、`:197`、`:242`。

这是接收 journal，与发送的 4096 条 journal 不同。距离已经变小但尚未完成 write-through 时，由 RX 的执行状态、READING 槽位及接收暂存区共同保留责任。零有效改进的批次可以正常 ACK，因为其中的距离更新已经被相同或更小距离支配；这一结论依赖 owner 的已有激活/执行/恢复责任没有丢失。

主接收路径没有“只要顶点已经在队列中，就跳过本次严格改进激活”的通用去重位。不能把审阅中的条件性反例误写成当前已经存在的行为。

### 5.2 旧 key 不等于旧距离展开

`mlmq.read` 返回的是一批带 key 的 `node_struct`。key 用于队列调度和与 `last_processed` 的过滤。但实际展开边时，当前普通路径读取权威 `node_data[src]`，再加上边权。

源码定位：`SSSP/sssp_run.cu:2296`、`:2366`、`:2448`、`:2625`。

对“队列中已有 key=10，owner 后来收到 4”的问题：

- 接收 4 若严格胜出，会生成自己的激活；
- 旧 key=10 的任务若仍获准展开，它使用当前 owner 距离，而不是必须按 10 松弛所有边；
- 若展开过程中又出现更小值，后续激活或冻结时的未处理改进扫描承担补充传播。

因此稿件中“remote candidates are generated from dequeued distance keys”应收紧。更准确的句子是：**队列 key 支持调度和过滤；边松弛基于 owner 当前的权威距离。**

### 5.3 `last_processed` 不是“全部出边已完成”的独立证书

源码在展开开始处更新 `last_processed`，然后才执行后面的边处理。因此它更像“本次处理所依据的距离进度标记”，不能单独解释成“该距离对应的所有出边已经完成传播”。并发路径使用普通 store，也不应未经证明称其为严格单调的原子完成变量。

正确性叙述必须把它与以下条件组合：worker 的在途责任未提前退休；安全点在本轮处理和输出发布之后；冻结后检查 `node_data < last_processed` 的残余改进，并通过 dirty/injection 恢复。

源码定位：`SSSP/sssp_run.cu:2360`、`:3838`、`:4019`、`:4152`、`:8245`。

### 5.4 L2 提交究竟意味着什么

`ml_queue::write_through()` 直接调用 `q2.write()`。当前 RX 限制为 DQ：DQ 的写入先预留位置，各 lane 写数据，执行 device fence，再发布 `block_write_done` 等完成计数；L2 manager 据此推进可读位置。

源码定位：`core/src/ml_queue.cu:359`；`core/cu_delta_queue/cu_delta_queue.cuh:427`、`:460`、`:479`、`:483`。

所以“提交 L2”不要求此时消费者已经取到记录，也不要求整个桶已经立刻可读；它要求数据及提交记录已经交给 L2 的推进机制。队列仍需继续保留这项未完成工作。

L2 的全局 outstanding 计算为累计写预留减去累计完成数；读出的工作通过 `on_the_fly_num` 保留在这项责任中，直到本地后续工作处理完成。该计数不同于“当前立即可读条目数”。源码专门区分了 `get_global_queue_size()` 与 `get_available_queue_size()`。

源码定位：`core/cu_delta_queue/cu_delta_queue.cuh:588`、`:602`、`:614`；`core/src/ml_queue.cu:254`；`SSSP/sssp_run.cu:3838`。

### 5.5 审阅中“队列满怎么办”的真实答案

**当前 DQ 写路径没有检测满后返回失败、阻塞等待空位或重试的分支。** 它将预留位置按 `total_size` 取模后写入，并总是返回 `WRITE_SUCCESS`。因此 RX 中 `assert(status == WRITE_SUCCESS)` 不能证明没有覆盖仍有效的环形存储。

论文必须明确容量使用前提，或后续补充实际容量检测/保留机制；不能在伪代码里擅自增加“L2 满则等待并保留激活”，然后把这当作当前代码的合同。

此外，`write_reserve/read_done` 等是有限宽度计数；DQ 文件本身注明尚未考虑其溢出。`GPU_MEMORY` 宏配置为 2 GiB，但容量经 `int max_size` 传递，2 GiB 边界的类型转换和实际分配结果也应核对。论文容量表宜报告实际分配及可用条目数，不应只照抄宏。

源码定位：`core/cu_delta_queue/cu_delta_queue.cuh:83`、`:120`、`:467`、`:489`；`SSSP/sssp.cuh:7`；`SSSP/sssp_run.cu:1177`。

本轮没有构造容量溢出的 GPU 反例；这里的结论是**源码没有提供审阅所要求的一般满队列保证**，并非声称已有道路实验发生了覆盖。

## 6. L3-05：终止不是两次看到队列为空

### 6.1 当前确有 token 化的冻结协议

普通双卡路径使用本地 `ACTIVE → QUIESCING → READY` 状态，以及每次确认请求的 token。manager 在初步检查本地 L2、dirty、候选、inbox/outbox、注入等条件后，发出新的 token 并进入 QUIESCING。

源码定位：`SSSP/l3/l3_termination.cuh`；`SSSP/sssp_run.cu:8096`、`:8144`。

**两张卡的终止 token 是各自的序号，不要求数值相同。** 它们用于区分本卡本次冻结与历史确认；全局完成通过稳定状态与通信责任检查建立。不要改写成“所有 GPU 使用同一个全局 epoch”，也不要与 inbox epoch 混同。

### 6.2 谁在什么条件下确认

| 角色 | 本次 token 的确认条件 | 确认后行为 |
|---|---|---|
| work warp | 当前处理已完成，输入为空、L1 为空、on-the-fly 为零，并满足 warp 一致性检查 | 保持冻结，不恢复普通求解；可参与受控的恢复扫描 |
| injection warp | 私有待注入批次和恢复请求等已清空 | 确认并冻结普通注入 |
| TX warp | 已完成 retained journal 重试，位于下一次领取之前的安全点 | 确认后不再开始普通候选领取 |
| manager/RX | 不把自己冻结成无法处理消息的角色 | 继续观察消息、推进确认或撤销冻结 |

源码定位：`SSSP/sssp_run.cu:4019`、`:4053`、`:9105`、`:9142`、`:10177`。

manager 检查每个 worker 的独立 ACK slot 是否等于当前 token，再检查 injection 和 TX ACK；不能用“总共收到这么多个 ACK”代替“每个角色确认了本次请求”。

### 6.3 收齐确认之后还要重查什么

收齐本次确认后，manager 重新检查：

1. **权威 candidate mark** 是否为空；若不为空，取消或反馈排空，不能只相信 hint；
2. TX 确认之后的 **最新 published epoch**，以及相应对端 ACK；
3. `l3_busy`、inbox/outbox、dirty、注入在途和 L2 outstanding；
4. 冻结 worker 协作扫描本 owner 的 `node_data < last_processed`，发现未覆盖改进则生成 dirty 并恢复求解。

源码定位：`SSSP/sssp_run.cu:8173`、`:8209`、`:8224`、`:8245`。

这里“TX 确认后读取最新 published epoch”尤其重要：初步检查读到旧的已 ACK epoch，不足以证明 TX 在进入冻结前没有又发出一批。

发送方某个 producer 已降低 cache/candidate 却尚未标记时，它尚不能通过 worker 安全点确认；已经领取但尚未发布的更新由 TX busy/retained 状态覆盖；收到并更新距离但未完成激活的更新由 RX 执行及接收槽位覆盖。论文应利用这些覆盖关系解释责任不会落入终止检测的空隙。

### 6.4 新消息如何使旧确认失效

manager 观察到新 inbox，而当前有终止请求时，会先撤销请求、恢复 ACTIVE、清 idle，再回到处理流程。这样 workers 有机会重新工作，然后 RX 再提交新激活。

源码定位：`SSSP/sssp_run.cu:7009`。

等待最新发送 ACK 时，`L3_TERM_WAIT_ACK` 保持在 manager 服务循环中检查，而不是进入禁止接收推进的永久等待。最终条件成立后进入 READY；两边 READY 且相关条件仍成立，才设置结束标志。条件失效、token 被取消或对端恢复 ACTIVE 时，相应确认会失效。

源码定位：`SSSP/sssp_run.cu:8277`、`:8303`、`:8355`、`:8374`。

因此可以给出条件性的完成论证：冻结排除新的普通 producer；最后通信与 owner 工作责任被逐项检查；若仍有必要工作则取消确认恢复；双边稳定 READY 时已无可继续产生必要工作的责任。

### 6.5 运行进展的证据与边界

每卡先 launch manage kernel，再在另一条 nonblocking stream launch work kernel；两卡由两个主机线程并发启动。这些设计是为了让通信和求解能同时推进。

源码定位：`SSSP/sssp_run.cu:1205`、`:11144`、`:11181`、`:11225`；`SSSP/main.cu:764`。

但 launch 顺序不是任意 GPU/任意资源配置上的形式化驻留保证。当前代码对 320/384 配置有 worker occupancy 检查，W512 不在该条件分支中。论文应说明已测 A100 上的 block 数、资源占用和并发安排，并将进展推论限制在管理与工作角色能持续获得执行机会的条件下。

当前 DQ 不存在“等待空位”的常规分支，因此不能用虚构的 L2 背压等待规则证明无死锁；其实际限制是容量安全。若后续增加有界等待，还必须重新验证“冻结 workers—接收等待 L2—ACK 等待接收”的潜在等待环。

已测作业正常退出说明这些样本没有挂起；它不等于任意图、任意 epoch 长度、任意设备资源配置下的无死锁定理。

## 7. L3-06：shortcut 的精确定义、保距证明与数值限制

### 7.1 严格模式下的内部顶点条件

当前已测日志为 `L3_CHAIN_SETUP loose=0`。对顶点 `u`，严格模式要求：

1. `u` 不是构造时指定的 source；
2. CSR 中恰有两条出边，目标分别为 `a,b`；
3. `a != u`、`b != u`、`a != b`，即排除自环和两条边指向同一邻居的情况；
4. `u,a,b` 同属一个 owner；
5. `u` 恰有两条入边，且来自 `a,b` 的反向连接都存在。

第 5 点要求双向连接，但**不要求两个方向的权重相等**。入度统计按实际边计数，因此额外平行入边或第三个入口会使该顶点不能作为严格内部顶点。

源码定位：`SSSP/l3/l3_chain_shortcuts.h:26`、`:39`、`:45`、`:62`。

不能只把这个条件缩写成无定义的“degree-2 vertices”；至少要说明出度、入度、两个不同邻居、owner 和 source 条件。未满足条件的顶点在这里作为 anchor。

### 7.2 具体添加哪些边

算法从一个 anchor `a` 的每条出边出发，沿符合条件的内部顶点继续走。在内部顶点处选择另一个邻居方向，即不原路退回前驱的出边，累加沿行进方向的边权；遇到下一个 anchor `b` 时，若跨越至少两条原边且 `a != b`，则添加 `a → b`。

反方向的 shortcut 由对应反向遍历独立生成，其权重按反向有向边求和，通常不等于正向权重。实现不是“每个链内部顶点都额外连接两个端点”，也不是删掉链内部顶点。

新增边先放在 anchor 的 CSR 行中，随后按原有顺序保留所有原始出边及其权重。**求解器因此同时看见 shortcut 和原边**，原顶点的距离仍需逐顶点求出。

源码定位：`SSSP/l3/l3_chain_shortcuts.h:68`、`:73`、`:90`、`:96`、`:99`、`:102`。

这也解释了性能作用：shortcut 可以提前将较好距离传播到链另一端，但原边仍会被遍历；最终减少或增加多少工作取决于调度与重复松弛，不能仅从新增边数推出加速。

### 7.3 环、跨分区链、source 和不可达区域

| 情形 | 当前严格模式行为 |
|---|---|
| 无 anchor 的纯链环 | 没有 anchor 起点去生成捷径；原图保留，算法正常在原边上求解 |
| 从 anchor 绕回自身的路径 | `current == u` 时不添加自返 shortcut |
| source 位于链内部 | source 强制作为 anchor，构造结果可能随 source 改变 |
| 链靠近分区边界 | owner 条件阻止跨 owner 折叠；原始跨分区边仍保留 |
| 额外入边/平行边/自环 | 相应顶点不能作为严格内部顶点，成为停止折叠的位置 |
| 不可达分量 | 可构造其内部 shortcut，但没有源点可达路径时，不会因此变为可达 |

构造还带有超过顶点数的行走保护。论文可以用这些规则明确覆盖范围，不需要宣称对每个环都做了特殊压缩。

### 7.4 可用于正文的保距证明

设增强图 `G+` 保留 `G` 的全部顶点和有向边。对每条新增边 `e=(a,b)`，构造过程给出原图中的一条有向路径 `P_e`，并赋值：

\[
w(e)=\sum_{f\in P_e}w(f).
\]

在路径求和和距离比较不溢出的前提下，对任意 source `s` 和顶点 `v`：

- `G` 中的路径仍存在于 `G+`，故 `d_{G+}(s,v) ≤ d_G(s,v)`；
- 将 `G+` 中任一路径的每条 shortcut 展开成其原图路径，可得同权重的原图有向 walk；非负权下可去除环而不增加权重，因此 `d_G(s,v) ≤ d_{G+}(s,v)`；
- 两者相等，不可达关系也保持。

证明的核心是**每条新边都对应真实原图路径、原边全部保留、数值表示正确**。严格入出度条件定义当前构造与工作量特征，并不是所有保距图增强都必须满足的必要条件。

固定构造出的 `G+` 具有上述任意源点保距性质；但这不意味着当前构造性能、source 作为 anchor 的选择及其收益不依赖 source。多源评估时应区分“复用同一增强图”和“按每个源点重新构造”。

### 7.5 数值语义还不能只写“非负权”

当前 `VALUE_TYPE=int`，`DIST_MAX=INT_MAX`。shortcut 构造使用 `long long` 累加，并拒绝路径和达到或超过 `INT_MAX` 的新增边；遇到这种情况保留原图，并记录 `overflow_paths`。

源码定位：`core/include/common.h:11`；`SSSP/l3/l3_chain_shortcuts.h:71`、`:92`、`:95`。

这只检查了 **shortcut 边权本身**。GPU 普通松弛仍直接计算 `node_data[u] + edge_weight`，没有在相应加法处看到宽整数求和、饱和或越界检查。

因此如果保持现有实现，支持域应包含：非负整数权重，所有实际执行的有限距离加边权均可在类型范围内表示，且有限值不能与 `INT_MAX` 无穷哨兵混淆。仅保证最终最短距离不溢出仍可能不够，因为算法也会尝试更长的非最短路径。

CPU 图增强测试使用 64 位参考距离，不能由此推出 GPU 的 32 位求解算术已获得相同范围保证。

### 7.6 本轮已完成的主机端验证

本轮实际执行：

```bash
g++ -std=c++14 -O2 scripts/multigpu/test_chain_shortcuts.cpp -o /tmp/tpds_l3_chain_review_test
/tmp/tpds_l3_chain_review_test
```

退出码 0，输出：

```text
CHAIN_SHORTCUT_CPU fixtures=213 all_sources/original_edges/owner/asymmetric/zero/disconnected/overflow/reset PASS
```

该现有测试逐行检查原边保留，对构造视图用所有源点比较 64 位 Dijkstra 距离，覆盖不对称方向权重、零权边、断连、额外入口、环、溢出路径处理和重复构造；另有负权输入拒绝检查。

这提供了当前严格 host 构造的实际证据。它不是 GPU 测试，也不是对所有图的穷举证明。`LOOSE` 为另一个关闭的实验分支，放宽了入边和反向连接条件，不能把本节严格模式的构造合同直接套用到它。

## 8. L3-07：可以怎样补写双卡实验设置

### 8.1 当前能填入的内容

| 项目 | 当前核查结果及论文应写的边界 |
|---|---|
| 设备与互连 | 项目已测平台为 ada-A100 的两张 A100 80 GB；同批作业 33155 输出 `GPU0↔GPU1 NV12` 拓扑。完整型号、UUID、驱动和频率应随该批实验单独归档，不能用本机信息替代 |
| GPU 数 | 双卡日志逐条 `gpu_count=2`；无 L3 MLMQ 与 ADDS 对照均为 1 |
| 通信 | 程序启用 CUDA peer access；普通求解期间 TX 写 peer inbox，不经过主机逐批转发。主机仍负责启动、计时和结果收集 |
| 编译 | 冻结编译器记录为 CUDA 12.4，nvcc 12.4.131；目标 sm80 |
| 队列 | 对照脚本固定 MLMQ `L1SLF_L2DQ`，delta=200000；work blocks=107 |
| worker | W512；每 block 16 个 work warps，另有一个本地管理 warp，实际 544 threads |
| 典型本地参数 | 源码默认 `node_size=32`、L1 SLF 大小 32、L2 batch=8、16 桶、同时考虑 4 桶；需将最终 setup 值而非仅默认宏写入构建清单 |
| L2 存储 | 配置宏 2 GiB；实际条目容量及有符号接口范围仍需核对，见 §5.5 |
| 候选存储 | 当前为对端顶点域 candidate/cache，加位图及静态 mark-word boundary index；不是已经压缩成仅边界顶点存储 |
| 发送 journal | 上限 4096 条；实际每批条目数受当前待发候选及接收容量限制 |
| inbox | 每方向 2 个槽；底层每槽按 owner 顶点域分配，不能将物理槽容量等同于通常发送的 4096 条 |
| RX commit | 严格改进压缩后按 32 条提交 L2，并提交尾批 |
| 窗口 | 普通扫描 budget=min=max=25000 cycles；另有终止排空反馈 |
| 终止/恢复 | 当前 token 的 worker/injection/TX ACK、权威 mark、最新 published ACK、owner 残余改进扫描 |
| 支持范围 | 当前 shortcut 入口明确拒绝 `n_gpu>2`。不能由 `MAX_GPU=8` 推出本配置支持 8 卡 |

源码定位：`SSSP/main.cu:478`、`:622`；`SSSP/sssp.cuh:7`、`:332`、`:343`、`:406`、`:682`；`core/include/common.h:24`；`SSSP/l3/l3_receive.cuh:89`。

冻结 `macros.txt` 并非完整命令行：它列出了 window、recovery、ACK、W512、boundary、shortcut、idle-token 等开关，但缺少 direct RX、retained TX 等必要开关。相邻 235 阶段文档给出了更完整的编译命令，其中还有 `WORK_COUNT=false`、`L3_COOPERATIVE_COLLECT=true`、`L3_DIRECT_RX=true`、`L3_RETAIN_TX=true`。窗口模式的编译守卫也要求 direct RX/retained TX。

这足以识别当前采用的协议族，但**还不足以把一个不完整宏文件称为完整复现清单**。应补存当前构建的完整命令行或预处理宏转储，并明确区分源码默认开关、实验显式开关及其派生开关。

### 8.2 图布局、source 和切分

三方实验脚本读取 `tmp/landmark206_matrix/<graph>/landmark.gr`，source 来自相同目录的 `layout.json` 中 `source_new_id`。因此“原图”在本对照里应解释为**经过既有编号重排、但尚未增加 chain shortcuts 的输入图**；不应让读者误解为原始下载文件的字节布局。

| 图 | 本批 source（重排后 0-based） | GPU0 顶点切分百分比 | 新增 shortcut 数 | r0 host 构造/ms |
|---|---:|---:|---:|---:|
| NY | 132173 | 60 | 58,564 | 15.185 |
| BAY | 160635 | 50 | 84,614 | 19.021 |
| COL | 217833 | 55 | 167,438 | 29.739 |
| FLA | 0 | 50 | 308,828 | 67.467 |
| CAL | 945407 | 50 | 639,572 | 131.442 |
| E | 1799311 | 50 | 1,130,372 | 243.885 |
| W | 3131052 | 40 | 2,309,198 | 443.904 |
| USA | 11973673 | 60 | 8,976,160 | 1,701.669 |

数据来自 `tmp/base_today/run_threeway_samebatch/r0_*_l3cut_2.log` 的 `L3_CHAIN_SETUP`；构造时间只是 r0 实测值，不是两轮中位数或端到端成本。该批八图均记录 `loose=0`、`overflow_paths=0`。

切分由 `floor(|V| × percent / 100)` 确定并进行边界限制，不是按边数均分。每图选取不同切分比例属于配置调优，应说明参数如何选取，不能把它写成自动、自适应且无需调参的分区算法。

例如 USA 本批两个分区为 `[0,14368408)` 与 `[14368408,23947347)`，增强后各有 40,197,322 和 26,487,462 条出边；它们并不边数均衡。日志中的 boundary words 也不是边界顶点数，若论文需要跨分区边数和不同边界顶点数，应另行准确统计。

源码定位：`scripts/multigpu/run_l3_threeway.py:13`、`:17`、`:39`、`:82`；`SSSP/graph_partition.h:81`。

### 8.3 solve 的真实起止点

MLMQ 使用主机 `steady_clock`：两卡完成各自求解准备和 pre-launch synchronization 后进入 barrier，最后到达者记录开始时间；两卡随后启动 manage/work kernels；每卡 `cudaDeviceSynchronize()` 返回后调用 finish，最后一个完成者记录结束时间。

因此：

\[
T_{solve}=t_{\text{last GPU host finish}}-t_{\text{all GPU hosts ready}}.
\]

它覆盖两卡 launch、求解期间的计算/通信/恢复/终止及最后同步，不是两张卡时间相加，也不是取较快卡的局部时间。它是包含启动及同步开销的求解区间主机墙钟。

源码定位：`SSSP/benchmark.h:22`、`:36`、`:46`；`SSSP/sssp_run.cu:11163`、`:11307`。

现有 `solve_ms` 不计入的部分包括：输入读取、CPU reference、主机 shortcut 构造、分区和主要通信结构准备、求解前的每次 reset，以及求解后的结果 D2H 和逐顶点检查。主程序另外输出 `BENCH_SETUP`、`prepare_ms`、`collection_ms`、`query_wall_ms`。

源码定位：`SSSP/main.cu:469`、`:497`、`:745`、`:755`、`:774`。

ADDS 的 `solve_ms` 从 driver kernel launch 前开始，直到其设备同步完成；每次查询的 reset 和 profiler/set_param 在该区间之外，另有 `parameter_ms`、`complete_solve_ms`。其当前冻结 wrapper 的这段代码与工作区一致。

源码定位：`ADDS/official_bench.cu:103`、`:109`、`:127`、`:142`。

论文可以据此比较 solve，但应同时披露两边排除的工作。若要报告一次查询的实际总开销，应采用一致的 `setup/init/solve/result` 边界重新汇总，而不是把 solve 比值命名为端到端加速。

### 8.4 多查询摊销不能任意假定

当前普通 benchmark 进程对一个 source 构造一次增强图，然后预热和重复求解；同一进程可复用图和工作区。多源在同一个进程连续运行、每个源点都重新构造，以及固定增强图跨源复用，是不同的实验协议。

可以在论文中给出：

\[
\bar T(N)=T_{setup}/N+\bar T_{init}+\bar T_{solve}+\bar T_{result}.
\]

但 `T_setup/N` 仅适用于实际能复用的那部分。USA 本批约 1.70 秒的 host shortcut 构造明显不是 67.97 ms solve 可以忽略为零的工作。本轮没有测量完整端到端回本次数，不建议从零散数字推算后写成实验结论。

## 9. L3-08：图 10、当前数值和结论边界

### 9.1 当前图表的数据与来源说明没有对齐

本轮核对：

- [metrics_threeway_bars.csv](figures/l3_evaluation/metrics_threeway_bars.csv) 对应 `tmp/base_today/run_threeway_preview/records.json`，统计方式为先取每轮中位数、再取两轮中位数的中位数。
- [provenance_l3_road_bars_threeway.json](figures/l3_evaluation/provenance_l3_road_bars_threeway.json) 的 records 路径和 SHA256 仍指向这个 preview 文件，却将 batches 描述为后来“同批交错、两轮、每轮 1 预热+5 正式”的协议。
- 实际 preview 记录为 48 个进程记录、144 个正式样本，即每轮 3 个正式样本；真正的同批 5 正式结果在 `run_threeway_samebatch`。

preview 文件 SHA256：`17cb799fa234613e9656c2749f9ed64cb06c448bfb923bea9f8be76cf092eb21`。

samebatch 文件 SHA256：`1c8b00122182fbe7b9fd54c4d8c6325210472cfe6d7595a7a4e8e51c6f569df0`。

这是复现元数据的一致性问题。修论文时应选择一个明确批次，统一原始记录、统计脚本、CSV、图、caption 和正文。只修改 provenance 的文字，不能把旧数字变成新实验结果。

### 9.2 本轮从同批原始日志重新核对的结果

作业记录：[slurm_33155.out](tmp/base_today/slurm_33155.out)，末尾 `THREEWAY_DONE errors=0`、`THREEWAY_SAMEBATCH_DONE rc=0`。本轮核查 48 个进程的原始日志与 JSON 样本逐项一致；各进程记录 rc=0，240 个正式样本及预热日志均有正确结果标记，双卡配置均为 `gpu_count=2`，未检出 `Error at node`。

每种配置每图为两轮，各 1 次预热和 5 次正式；下表使用**合并 10 个正式样本后的中位数**，保留全部慢样本。本轮只是重新读取并计算已有实验，没有重新运行 GPU。

| 图 | ADDS 单卡/ms | 无 L3 MLMQ 单卡/ms | 完整 L3 MLMQ 双卡/ms | 无 L3 单卡/完整双卡 |
|---|---:|---:|---:|---:|
| NY | 10.660158 | 7.295258 | 6.425840 | 1.1353× |
| BAY | 13.076920 | 5.373355 | 4.708523 | 1.1412× |
| COL | 25.004295 | 9.217434 | 7.020116 | 1.3130× |
| FLA | 54.699159 | 21.947872 | 17.298266 | 1.2688× |
| CAL | 37.110311 | 26.379711 | 18.042411 | 1.4621× |
| E | 39.654055 | 30.486472 | 24.633992 | 1.2376× |
| W | 55.199059 | 41.161457 | 29.896466 | 1.3768× |
| USA | 136.189181 | 101.195605 | 67.972080 | 1.4888× |

按每图比值取几何平均：

- ADDS 单卡 / 无 L3 MLMQ 单卡：**1.7277×**；
- ADDS 单卡 / 完整 L3 MLMQ 双卡：**2.2408×**；
- 无 L3 MLMQ 单卡 / 完整 L3 MLMQ 双卡：**1.2969×**。

这一批的每图中位比值都超过 1.1，不应据此恢复“每图跨批稳定超过 1.1”的要求或结论；此前 NY 等图已有波动记录，用户也已允许存在未达 1.1 的图。报告保留实际结果即可。

如果沿用旧绘图脚本“每轮中位数再取中位数”的方式，同一 samebatch 数据的对应几何平均为 1.7277×、2.2397×、1.2963×。两种统计在一般情况下并不等价；请选择并写清一种，不能混用它们的逐图数值。

### 9.3 为什么还有 1.26、1.275 等数字

| 数字 | 实际对应内容 | 当前可如何使用 |
|---|---|---|
| 约 1.259× / 2.201× | 现有三柱图 preview 记录，逐轮中位数聚合 | 仅保留该批次来源和原统计口径时使用 |
| 约 1.275× | step3 作业 32444，单卡/双卡各图 6 正式样本合并中位数 | 是另一个已有批次，不与 ADDS 后测数据混称同一次三方实验 |
| 约 1.297× / 2.241× | 作业 33155，真正三方同批交错，10 正式样本合并中位数 | 可以作为更新图表的已有数据依据，但需同步更新图和全部 provenance |

这些数字的差异包括批次和聚合方式差异，不能简单解读为又产生了一个更快的算法版本。三者也都没有单独隔离图增强、worker 几何、切分与 L3 传播机制的贡献。

### 9.4 图表标签和结论建议

现有三柱图宜明确标成：

```text
ADDS, 1 GPU, G
MLMQ without L3, 1 GPU, G
MLMQ with L3 and chain shortcuts, 2 GPUs, G+
```

caption 写清：八个图、固定源点、同批交错与重复协议、统计方法、solve 边界、shortcut 和分区设置。匹配输入的四配置结果宜单独制表或增加一幅消融图，避免把组合配置柱图称为强扩展证明。

在补齐消融前，正文可以说“完整双卡配置相对所报告单卡配置改善了 solve time”，不能说“L3 本身带来 1.30× 加速”“仅通信聚合解释全部收益”或“已经证明任意多 GPU 可扩展”。

机制指标也应与主张对应：

| 拟写的机制主张 | 最少需要的解释性证据 |
|---|---|
| 候选合并减少传输 | 远端尝试、candidate 严格改进、发送条目和 owner 胜出数；明确定义各自分母 |
| 批处理形成有效聚合 | 批大小分布、批次数、空扫描、retained 重试次数或等待时间 |
| 两卡并行提高处理效率 | 每卡展开边数、有效工作量、活动/等待区间及负载差异 |
| shortcut 缩短传播过程 | 固定设备数和其他条件时，shortcut 开关的时间与工作量变化 |
| 终止开销得到控制 | QUIESCING 次数、撤销次数、恢复次数、尾部时间占比 |
| 窗口选值稳健 | 合法邻近参数的时间、批大小和延迟统计；当前 25,000 下限使 0.5 倍并非直接可运行的同一配置 |

当前源码提供若干诊断开关及 L2 final counts，但诊断版与正式性能版应分别标记。并行角色活动时间可能重叠，不应相加后当作 solve 的耗时分解。

## 10. 对审阅计划“最小验证集”的逐项回应

| 审阅建议 | 已有证据或源码支持 | 本轮结论/剩余工作 |
|---|---|---|
| 连续改进同一远端目标，在领取处制造交错 | 先清 mark 后交换 candidate；存在 `test_l3_collect.cu` | 该测试主要预填状态后收集，并非并发 producer 的交错证明；专门 interleaving 仍需补 |
| 接收槽位忙、延迟消费和 ACK | retained TX 和两槽协议；fault512 记录 transport recovery | 支持有限故障注入场景；不等同于任意容量缩小、长时间背压、部分发布故障全部通过 |
| 距离更新后延迟激活，并同时触发终止 | direct RX 先提交后 ACK；新消息先取消冻结 | fault512 隐藏一次 RX 通知验证恢复；仍应单独覆盖提交窗口与终止的受控交错 |
| 旧 key 遇到更小距离 | 读取 owner 距离，严格胜出生成激活，冻结残差扫描 | 源码能排除审阅描述的简单反例；应保留指定 key=10/4 的定向可重现证据 |
| 两侧交替空闲和晚到消息 | token、TX 确认后重查、取消和排空反馈 | 道路/fixture 运行成功不等于可控地覆盖所有关键晚到时序 |
| 连续不同源点与状态复用 | 有 reset 入口和历史多源检查记录 | 当前三方性能实验每图固定 source 重复，不应写成同进程多源复用已全面验收 |
| shortcut 非对称、环、零权、不可达、数值边界 | 本轮现有 CPU 测试 213 fixtures PASS | 严格 host 构造有新验证；GPU 数值范围仍需另查 |
| L2 容量接近上限 | 当前总是成功的取模写入 | 未关闭；优先明确安全容量不变量或实现容量处理，再验证 |
| P2P 可见性/作用域 | fence 与协议字段可定位，已有双 A100运行记录 | 未完成正式内存模型审核；应补目标平台能力和明确同步语义 |

故障记录：[run_fault512/verified.json](tmp/base_today/run_fault512/verified.json)。作业 32906 对 NY/W/USA、`late_fan` 两源、`zero_fan` 两源共 7 次运行记录均为 correct、transport_recovered、rx_injected_and_recovered，问题列表为空。它们是**此前运行的故障构建证据**，本轮没有重新执行，也不是当前正式二进制天然会注入故障。

建议新增检查按以下顺序推进，目的在于补齐论文证据而不是追求每个图都超过任意门槛：

1. 明确 candidate/mark 与 P2P 控制字段的内存顺序合同，核实 L2 有界存储与计数范围；这些影响正确性和进展表述。
2. 固定当前 W512、cut、source 和完全相同的 `G+`，做四配置对照；额外源点作为泛化检查，不与调参用源点混在一起汇总。
3. 在真实双卡上补关键交错、迟到消息、ACK 延迟和多查询 reset 的小用例，记录完整输出与退出码。
4. 在相同图和配置下补必要的机制计数，再决定哪些机制性结论值得放进正文。
5. 统一统计口径和 artifact provenance，更新图 10、实验设置与结论。

上述为后续建议，本轮没有启动这些 GPU 任务。

## 11. 可直接用于改稿的组织与英文段落

### 11.1 章节修改地图

| 稿件位置（沿用审阅定位） | 建议写法 |
|---|---|
| III-E | 定义 L3 为 owner-based 传播层；本地 Q 与远端候选分离；用责任链连接发送、接收和完成 |
| IV-A | 将逻辑角色对应到每卡 work/manage kernels，解释额外本地管理 warp、RX/TX/injection、启动和资源前提 |
| IV-C 的 aggregation | 明确 peer cache 语义、candidate/mark 领取顺序、boundary word index 和当前真实窗口 |
| IV-C 的 publication | 给出 retained journal、槽位复用、publication/ACK 的不同责任转移点；同步语义需完成审核后定稿 |
| IV-C 的 owner activation | 严格胜出、RX 暂存、L2 commit、旧 key 与 owner 距离的关系；标注容量前提 |
| IV-C 的 completion | token 化安全点、冻结后重新核查、取消与恢复、双边 READY；不将 L2 empty 当充分条件 |
| IV-C 的 shortcut | 独立一段定义严格条件、沿有向路径求和、保留原边、保距证明和整数范围 |
| V-A | 图 10 专属设备、图布局、source/cut、宏、计时与统计方法 |
| V-B/V-C 与图 10 | 统一已有批次，标明完整配置；增加匹配图表示的消融以回答 L3-01 |
| VI/VII | 限定双 A100 和已测输入，区分完整配置改进、传播机制与图优化的贡献 |

正文不必复制全部实现变量。可以保留简短的发送/接收伪代码，把窗口规则、slot 状态、容量和终止确认细节放在相邻合同说明或附录中；但不能省去责任转移与同步前提。

### 11.2 III-E 概述段

以下为依据源码重新撰写的建议文本，不是从原稿逐句摘录：

> MLMQ extends its local L0–L2 hierarchy to two GPUs through an owner-based propagation layer. Each GPU stores the outgoing edges and authoritative distance labels of its owned vertices. Local improvements enter the local hierarchy, while remote improvements are aggregated by destination and transferred through bounded transmission batches. The receiving owner submits winning distance improvements to its L2 queue before acknowledging the batch. Global completion additionally requires local quiescence and the discharge of outstanding candidate, transmission, reception, and activation responsibilities.

这里的 bounded 只限定 transmission batches，不宣称当前 L2 已具备满队列处理。

### 11.3 IV-C 候选与领取段

> A sender maintains a candidate minimum and a pending bit for each potential remote destination. A worker first reduces the candidate value and then marks it for collection. The transmission warp claims pending bits before exchanging their candidate values with infinity. Under the required ordering guarantees, a concurrent improvement is either included in the claimed batch or remains discoverable for a subsequent batch. Scan hints accelerate collection, whereas the authoritative pending bitmap is retained for complete checking. Claimed values are stored in a transmission journal, which is retried before any new collection if an inbox slot is temporarily unavailable.

其中 “under the required ordering guarantees” 不是用于回避实现审核。提交论文前应在正文或附录明确说明该保证如何由具体 CUDA 操作建立，见 §3.5 和 §4.5。

### 11.4 IV-C 接收与完成段

> The owner applies each received candidate using an atomic minimum and buffers only strict improvements for local activation. All resulting activations are committed to L2 before the corresponding inbox acknowledgement is published. Queue keys support scheduling and redundant-work filtering, whereas edge relaxation reads the owner's current distance label. Completion uses token-specific quiescence acknowledgements from work, injection, and transmission roles. After these acknowledgements, the manager rechecks pending candidates, the latest published transmission, and uncovered owner-side improvements. Newly observed work cancels the confirmation and resumes processing.

这段还应与 L2 容量合同、epoch 范围和平台同步前提一起出现；它不意味着这些边界已经在本轮全部验证。

### 11.5 Shortcut 定义段

> The evaluated configuration also augments each owner's graph with chain shortcuts constructed on the host before solving. Internal chain vertices have two distinct local outgoing neighbours and exactly two incoming edges from those neighbours; the query source remains an anchor. Traversal from an anchor follows the directed chain to another anchor and assigns the shortcut the sum of the traversed edge weights. All original vertices and edges are retained. Each shortcut therefore represents an existing directed path, preserving shortest-path distances within the supported arithmetic range. This graph augmentation is evaluated separately from cross-GPU propagation when attributing performance gains.

最后一句要求实际完成相应消融后使用；消融未补齐时，改为 “A matched-graph comparison is required to separate its performance contribution from that of cross-GPU propagation.”

### 11.6 实验计时和结果段

> Solve time is measured on the host from the point at which all participating GPUs have completed query preparation to the completion of device synchronization on the last GPU. This interval includes kernel launches, local processing, cross-GPU propagation, and termination. Graph loading, shortcut construction, query initialization, and result collection are reported separately. For ADDS, the reported solve interval excludes its per-query parameter profiling, which is recorded separately.

若统一采用 §9.2 的 samebatch 数据和合并样本中位数，结果段可以写为：

> Across the eight evaluated road networks, the complete two-GPU configuration achieves geometric-mean solve-time speedups of 1.297× over the reported single-GPU MLMQ configuration and 2.241× over single-GPU ADDS. These comparisons use two reversed interleaved rounds, each with one warm-up and five measured runs, and pool the ten measured runs per graph and configuration before taking the median. The two-GPU configuration uses owner-local chain augmentation, while the single-GPU configurations use the input graph without this augmentation. These results characterize the complete tested configurations; they do not isolate the contribution of L3 propagation from graph augmentation and configuration tuning.

该段适合在同步更新图表后使用。若保留旧图，则必须保留旧批次、旧数值和旧重复协议，不应只替换正文中的倍数。

## 12. 改稿前的最终判断

当前源码足以把 L3 写成一套有实际职责划分的多卡传播实现：候选合并、先清标记后取值、保留未发布 journal、接收提交后 ACK、token 化终止及残余改进恢复，都可以具体说明。

论文仍有三类尚未完成的工作：

1. **实现合同与证明边界**：CUDA 跨地址/跨卡同步语义、L2 容量与计数范围、GPU 整数加法支持域、驻留和进展前提。
2. **实验归因**：当前版本在固定 `G` 与固定 `G+` 上的单/双卡对照，以及保留机制性主张所需的计数。
3. **证据一致性**：将最新同批数据、统计方式、图、caption、配置 manifest 和结论统一起来。

完成这些工作后，论文能够同时回答“多卡传播怎样正确完成”和“性能收益分别来自哪里”。现阶段最有依据的定位是：**已实现并测得收益的双 GPU 完整配置，包含 L3 传播协议和 owner-local 图增强；L3 独立收益仍需匹配输入实验来确定。**

## 附录：关键源码导航与统计复核

下列链接指向本次核查的工作区；行号对应 2026-09-21 快照。

| 内容 | 入口 |
|---|---|
| shortcut 的多卡启用和主机构造 | [main.cu](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/main.cu:469) |
| 严格链条件与增强 CSR | [l3_chain_shortcuts.h](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/l3/l3_chain_shortcuts.h:24) |
| 本地/远端松弛分流 | [relax_dst](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/sssp_run.cu:2062) |
| candidate、mark 和事件计数 | [l3_candidate.cuh](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/l3/l3_candidate.cuh:7) |
| 先清 mark 后取 candidate | [TX 收集](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/sssp_run.cu:9443) |
| 窗口决策与 budget 更新 | [l3_window.h](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/l3/l3_window.h:11) |
| 槽位复用、领取和 ACK | [l3_transport.cuh](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/l3/l3_transport.cuh:33) |
| payload/count/epoch/READY 发布 | [l3_bulk.cuh](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/l3/l3_bulk.cuh:433) |
| owner 改进、L2 提交、ACK | [l3_receive.cuh](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/l3/l3_receive.cuh:88) |
| DQ 写入与 outstanding | [cu_delta_queue.cuh](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/core/cu_delta_queue/cu_delta_queue.cuh:427) |
| 终止确认后的重查与恢复 | [manager 终止路径](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/sssp_run.cu:8096) |
| 主机求解计时 | [benchmark.h](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/benchmark.h:27) |
| 同批实验参数与执行顺序 | [run_l3_threeway.py](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/scripts/multigpu/run_l3_threeway.py:37) |

在仓库根目录执行以下只读脚本，可复算 §9.2 的表和几何平均；它不启动任何 GPU 程序：

```python
import json
import math
from pathlib import Path
from statistics import median

records = json.loads(Path(
    "tmp/base_today/run_threeway_samebatch/records.json"
).read_text())
ratios = {"ADDS/SG": [], "ADDS/DG": [], "SG/DG": []}
for graph in ("NY", "BAY", "COL", "FLA", "CAL", "E", "W", "USA"):
    times = []
    for variant in ("adds1", "nol3_1", "l3cut_2"):
        rows = [r for r in records
                if r["graph"] == graph and r["variant"] == variant]
        assert len(rows) == 2
        assert all(r["rc"] == 0 and r["correct"] for r in rows)
        samples = [t for r in rows for t in r["solve_ms"]]
        assert len(samples) == 10
        times.append(median(samples))
    adds, single, dual = times
    ratios["ADDS/SG"].append(adds / single)
    ratios["ADDS/DG"].append(adds / dual)
    ratios["SG/DG"].append(single / dual)
    print(graph, *(f"{t:.6f}" for t in times), f"{single / dual:.6f}")
for name, values in ratios.items():
    print(name, math.exp(sum(map(math.log, values)) / len(values)))
```

该脚本从记录重算统计；本文另外完成了 JSON 与原始日志的逐样本核对。它不把记录中的 `correct` 布尔值升级为新的独立 GPU 复验。
