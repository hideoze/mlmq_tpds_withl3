# L3 补充实验执行结果（2026-09-21）

本轮范围：同图单/双 GPU 对照、少量并发正确性测试、memory ordering 与 L2 capacity 核查。主表使用容量修复后的作业 **37327**，不混入修复前的时间。最终诊断作业 **37329** 已完成，Slurm `COMPLETED / ExitCode=0:0`；32 个容量诊断查询、4 个同进程换源查询及小型并发测试全部通过。

## 1. 可以用于论文修改的主要结论

**完整配置的收益主要来自 shortcut，当前数据不支持“L3 在相同图表示上普遍加速”的表述。** 八图的完整配置比 `T(1,G)/T(2,G+)` 几何均值为 **1.2136×**；相同增强图的 `T(1,G+)/T(2,G+)` 为 **0.9494×**；单卡自身的 shortcut 收益为 **1.2782×**。三者满足乘法分解。USA 的增强图同图比为 **1.1047×**，其余七图均小于 1；原图同图比的几何均值为 **0.9037×**。

这里的“同图”已经固定 CSR 的顶点编号、行偏移、邻接顺序、目标数组和权重，不是只固定图名。每张 G+ 按当前既定的双卡切分和构造源点导出一次，单、双卡读取同一个文件，运行时都关闭再次构造 shortcut。

**L2 核查发现并修复了两个具体错误：** 2 GiB 浮点常量传给 `int` 字节参数越过可表示范围；按桶向上取整可能把未分配的尾部记录和块计数器算入容量。现在采用可表示的 `INT_MAX` 字节预算和按完整块向下取整的桶容量。

**Memory ordering 的核查结论仍是“存在未闭合的源码同步合同”。** 实测的候选竞争、P2P 槽位复用和 ACK 边界可以提供运行证据；它们不能把当前 relaxed、device-scope 原子自动提升为完整的跨设备内存模型证明。论文必须区分这两个证据层级。

## 2. 固定版本、平台与比较边界

- 分支 `para-frame3-multisource`，Git HEAD `b842b8a`；工作区已有未提交改动，**实际版本由源码快照、构建命令和二进制 SHA256 标识**。
- 两张 NVIDIA A100 80GB PCIe，每卡 108 SM，CC 8.0；驱动 `570.133.20`；实测拓扑 `NV12`。两方向 `cudaDevP2PAttrNativeAtomicSupported=1`、peer access=1。
- CUDA 编译目标 `sm_80`，`-O3 -rdc=true`；NVCC 版本归档在 `tmp/l3_supplement_20260921_v2/nvcc_version.txt`。
- 两张卡取得 Slurm 分配后检查为空闲，所有 GPU 运行均在 A100 分区。双卡各自实际参与同一个图查询；没有自动降成单卡，也没有使用多源数据并行路径冒充双卡图分区。
- 固定 `L1SLF_L2DQ`、delta=200000、每卡 107 work blocks、512 个工作线程加 32 个本地管理线程，即 544 threads/block；16 个桶，batch=8，扫描桶上限为 4。
- 双卡保持当前 direct RX、retained TX、cooperative collect、boundary index、window mode 2、ACK scan、worker recovery 和 idle token probe；只关闭 host 的自动 shortcut 构造，改读冻结 G/G+。

### 单卡基线怎样保持独立

单卡使用历史 `nol3_213_preload/source.tgz` 中的**独立无 L3 worker/manager 执行路径**，把 `core/` 同步为本轮双卡使用的同一份本地队列源码，并对两边应用相同的容量修复。它不是双卡程序的 `-n 1` 模式。四配置分别复用同一个单卡二进制或同一个双卡二进制。

共享 `core/` 的 19 个文件已逐一核对内容一致，清单在 [identical_core_files.json](knowledgebase/l3_supplementary_execution_20260921/identical_core_files.json)。单卡、双卡的完整 worker 并非逐字节相同：双卡还包含 owner 分派、远端 candidate、L3 接收、终止握手及恢复。这里测量的是**匹配 CSR 与本地队列的双 GPU 扩展系统表现**，不把差值解释为某一个通信函数的独立成本。

本次重建单卡基线与历史 `tmp/base_today/nol3/mlmq` 的二进制不同。因此本表不能替换到旧 ADDS 三方主图后仍沿用旧 ADDS 数据，也不能与旧批次拼接加速比。

正式单卡二进制 SHA256 为 `07baa7447bd942a955d7381ff418d312456946767ead7d71829a15303db0c0da`；正式双卡为 `12c97d8405fd2acddc8c296c182bc7bf5b8e51b55bd267fc552ab3545e2d3c08`。四套正式/诊断源码与构建哈希集中在 [audit.json](knowledgebase/l3_supplementary_execution_20260921/audit.json)。

### 图和源点

| 图 | 查询/构造源点 | 切分比例 | cut（0-based 半开边界） |
|---|---:|---:|---:|
| NY | 132173 | 60% | 158607 |
| BAY | 160635 | 50% | 160635 |
| COL | 217833 | 55% | 239616 |
| FLA | 0 | 50% | 535188 |
| CAL | 945407 | 50% | 945407 |
| E | 1799311 | 50% | 1799311 |
| W | 3131052 | 40% | 2504841 |
| USA | 11973673 | 60% | 14368408 |

图文件和 oracle 哈希、实际 cut、导出记录见 [graphs.json](tmp/l3_supplement_20260921_v2/matrix/graphs.json)。导出器采用 strict chain builder，回读后逐数组检查 CSR 顺序不变；独立 int64 Dijkstra 同时检查 G 与 G+ 的距离一致。正式 GPU 查询逐顶点核对从 **G** 生成的 oracle，包含不可达点。宽整数参考解是结果正确性证据，不等于对所有 GPU 中间加法范围的独立动态证明。

## 3. 八图四配置结果

每图每配置两轮交错，第二轮反转配置顺序。每轮一次预热、五次正式查询；每格取全部十个正式样本的中位数，保留慢值。共 64 个进程、384 个查询，其中正式查询 320 个，全部正确且正常退出。

下表时间单位为 ms。`S_same` 为 `M1-shortcut / M2-shortcut`，`S_complete` 为 `M1-original / M2-shortcut`。

| 图 | M1-original | M1-shortcut | M2-original | M2-shortcut | S_same | S_complete |
|---|---:|---:|---:|---:|---:|---:|
| NY | 6.4630 | 6.0098 | 7.1297 | 6.3752 | 0.9427× | 1.0138× |
| BAY | 5.1241 | 4.1287 | 5.7296 | 4.7419 | 0.8707× | 1.0806× |
| COL | 8.6291 | 5.7925 | 9.7735 | 6.9372 | 0.8350× | 1.2439× |
| FLA | 20.2768 | 17.3081 | 21.4211 | 17.6975 | 0.9780× | 1.1457× |
| CAL | 25.0761 | 17.1468 | 27.6907 | 18.0617 | 0.9493× | 1.3884× |
| E | 29.1492 | 23.4167 | 30.5892 | 24.2286 | 0.9665× | 1.2031× |
| W | 38.2061 | 29.1672 | 43.2860 | 30.0120 | 0.9718× | 1.2730× |
| USA | 96.6751 | 75.3454 | 112.1410 | 68.2066 | 1.1047× | 1.4174× |
| 几何均值（逐图比值） | — | — | — | — | **0.9494×** | **1.2136×** |

完整比值、分轮中位数和绝对时间见 [same_graph_summary.csv](knowledgebase/l3_supplementary_execution_20260921/same_graph_summary.csv)。每格的 Q25、Q75、IQR、MAD、最小值和最大值见 [timing_dispersion.csv](knowledgebase/l3_supplementary_execution_20260921/timing_dispersion.csv)。IQR 按十样本的 inclusive quantile 计算；没有根据单次最好结果挑选配置，也没有把相邻单/双卡样本误当成同步配对实验。

这些是 **solve-time**：不包含图导出、CPU reference、图加载、setup、查询重置与结果收集。新增逐顶点 oracle 检查在计时结束后执行。双卡起点为两侧准备完成的主机屏障，终点为两侧 GPU 同步完成后的最后结束时间。单卡计时覆盖其 worker/manager launch 和 device synchronization。两者仍有各自必要的启动路径，不声称“纯 kernel 指令执行时间”。

### 对归因的影响

`S_complete = S_shortcut_single × S_same`，几何均值同样有 `1.2136 ≈ 1.2782 × 0.9494`。这说明去掉单卡也能获得的图增强收益后，当前双卡扩展在这八个固定源点上的平均结果没有超过独立无 L3 单卡。USA 表明某些输入可以获得同图收益，但不能由一个图推广为总体收益。

“完整双 GPU 配置优于未增强单 GPU 基线”仍被本批支持；“L3 本身普遍提供额外加速”应删除或收窄。更具体的 TX/RX、聚合或窗口收益不能由这四组直接分解。

## 4. 小型并发正确性测试

测试源码：[test_l3_supplement.cu](scripts/multigpu/test_l3_supplement.cu)、[test_dq_capacity.cu](scripts/multigpu/test_dq_capacity.cu)、[run_supplement_checks.py](scripts/multigpu/run_supplement_checks.py)。

| 用例 | 具体覆盖 | 证据边界 |
|---|---|---|
| Candidate 受控交错 | 每卡 3 种时序各 32 次：第二次改进发生在 claim 前、claim 与 exchange 之间、exchange 后；检查两次领取值及最小值 | 直接调用生产 `l3_record_candidate`；拆分领取使用相同 CAS/Exch 顺序；测试 gate 用于强制时序，不是弱内存测试 |
| Candidate 并发压力 | 每卡 32 轮，8 个 producer warps 与一个 collector warp 并发；每轮 32768 次候选更新，1024 个目标 | 调用生产 cooperative collector，并保留 authoritative mark 扫描；比较每个目标收集到的最小值；fixture 关闭 peer-cache 更新，隔离 candidate/mark 责任 |
| 双槽占满、重试与 ACK 延迟 | 每方向 256 个 epoch、每批 32 条；先填满两槽并确认第三批发布失败，再允许 RX 推进；检查失败重试期间 journal 不变、全部 epoch/记录正确、延迟 ACK 实际触发且 pending 最终归零 | 使用生产 slot acquire、publish、claim、receive、finish/retry ACK helpers；sender 的重试循环为测试代码，不冒充完整 manager retained-TX 分支的独立证明 |
| RX 距离已更新、L2 未提交 | 每方向 256 次检查：状态仍为 READING、权威距离已更新、ACK 小于当前 generation；随后执行真实 DQ write | 同一 RX helper 的 `write_through` 测试包装点；对有限暂停前后的责任交接进行检查 |
| DQ 并发发布 | 400 个批大小/基址组合实例，32 producer blocks 与消费者并发 | 调用真实 DQ write，检查 payload/checksum 与记录唯一性 |
| L2 容量边界 | 容量 1024 时恰好写满通过；尝试写到 1056 时必须触发预期 device assert | 诊断版的受控负例；该进程期望检测到越界后返回成功，不把原始 assert 文本当作无意的运行失败 |
| 同进程换源 | 固定 2052 顶点 G+，源点序列 `0 → 1030 → 2050 → 0`；分别位于 GPU0、GPU1、孤点、GPU0 | 真正 `n=2` L3 图分区路径；四次查询复用 workspace 和图；每次逐顶点核对原图 int64 oracle，检查每卡 reset 次数与 L2 最终守恒 |

小图含非对称边权、零权环、重复 fan-in、跨 owner 边以及孤点。G+ 固定为 source=0 时的构造结果，换源时不重新构造；因此 reset 测试覆盖的是同一图表示和存储的复用。

最终实际触发记录：两卡合计 **192** 个受控 candidate 交错实例、**64** 个并发压力轮次；两个 P2P 方向各完成 **256** 个 epoch，第三批被背压及后续重试共记录 **44/47** 次。两个方向的 `delayed_ack_fired=1`、`pending=0`，各核对 **256** 次 RX 提交前边界和 **8192** 条写入 L2 的记录。4 次换源查询均 `correct=1`，每卡 workspace reset 序号递增至 4。

详细执行结果以最终 [checks/results.json](tmp/l3_supplement_20260921_v2/checks/results.json) 和 [audit.json](knowledgebase/l3_supplementary_execution_20260921/audit.json) 为准。本轮没有把普通查询成功自动记作“最后一批恰逢 QUIESCING”的受控触发证据；也没有把上述 helper 测试写成完整弱内存正确性证明。论文可引用已经触发并核对的具体覆盖，不宜直接照搬补充计划中全部五类事件均已独立关闭的句子。

## 5. Memory ordering：核实到哪一层

CUDA 12.4 文档将无后缀原子函数描述为 relaxed ordering、device scope；system fence 与原子作用域是不同概念。A100 的双向 native atomics 能力说明链路能力，不能单独补足程序的同步合同。[CUDA 12.4 Atomic Functions](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#atomic-functions)

| 接口 | 当前源码实际操作 | 要形成严格论证仍需明确的关系 |
|---|---|---|
| candidate → mark | `l3_candidate.cuh` 先 `atomicMin(remote_cand)`，严格改进后 `atomicOr(remote_mark)`；两者间无显式 release fence/原子 | 标记发布必须把候选值更新带给领取方；同设备 device-scope release/acquire 或等价、可论证的 fence 合同 |
| claim → candidate exchange | `l3_collect.cuh` 和 `sssp_run.cu` 的 fallback 先 CAS 清 mark，再 Exch candidate=INF | acquire 必须覆盖每一条领取入口，不能只修 cooperative 分支而遗漏 full scan；重挂候选也需同样的发布合同 |
| payload/count/generation → READY | `l3_bulk.cuh` 使用逐 lane payload 写、warp 同步和 system fences，最后无后缀 `atomicExch(READY)` | READY 是跨设备发布点；需要覆盖双方的 system-scope 同步，以及 RX 读取 payload 前的 acquire |
| RX commit → ACK/DONE → slot reuse | `l3_receive.cuh` 在 DQ 提交后才 `finish_read`；transport helper 以 generation/ACK 判断槽复用 | ACK/DONE 的发布和发送方读取需要一致的跨设备作用域；接收“已看到 READY”与安全读取/重用不能仅靠普通原子名称推定 |
| L2 payload → 可读位置 | producer lanes 在 block completion 前有 device fence；manager 更新读位置，worker 从本地缓存读可见范围 | 应给出完整的发布—观察链，包括 manager 的普通 load/store；只证明 producer 有 fence 不足以自动证明全部消费者访问 |
| 双卡终止控制 | token/ACK/frozen-state、最终 mark 扫描和最新 TX ACK 检查有明确算法顺序；peer 控制读仍包含无后缀原子 | 需要把本地控制同步与 peer state/ACK 的 system-scope 同步分别映射到源码；算法时序论证不能替代作用域论证 |

因此本轮可以写“目标 A100 平台上的定向并发测试通过”，不能写“所有 memory ordering 问题已经证明不存在”。如果论文希望保留更强的无丢失/终止定理，后续应把上述字段的发布点、观察点、scope 和 memory order 明确化，覆盖 full-scan、重挂、延迟 ACK 恢复及终止读取的所有入口；同步路径调整后需使用新版本复验，现有性能表不自动代表该新版本。

## 6. L2 capacity：修复与受限安全结论

生产代码只作了两处相关修正：

1. [SSSP/sssp.cuh](SSSP/sssp.cuh) 的 `GPU_MEMORY` 改为 `INT_MAX`，即 2 GiB−1 的可表示字节预算。
2. [core/cu_delta_queue/cu_delta_queue.cuh](core/cu_delta_queue/cu_delta_queue.cuh) 的 `host_init` 与 `host_reinit` 均先确定实际记录数，再按每桶完整块向下取整。

在当前 8 字节记录、16 桶和 512 记录/块下：

```text
allocated_records = floor(2147483647 / 8) = 268435455
old_per_bucket    = floor(ceil(268435455 / 16) / 512) * 512 = 16777216
old_total_claimed = 16 * 16777216 = 268435456  > allocated_records

new_per_bucket    = floor(floor(268435455 / 16) / 512) * 512 = 16776704
new_total_usable  = 16 * 16776704 = 268427264 <= allocated_records
```

旧算法不仅多宣称一个记录，还可能访问未分配的最后一个 `block_write_done` 项。修复后，每桶容量为 **16,776,704 条**，各桶完整块均在分配范围内；未使用的尾部空间是 **65,528 字节**，不是额外可借用的最后一桶容量。

诊断构建在每次 `atomicAdd(write_reserve)` 后、计算取模地址和写 payload 前，检查 `old >= 0 && (int64)old + count <= capacity`；每次读预留也检查有符号范围。每个写预留增量至多一个 warp 的条目数，故完整通过该检查的查询不会先跨过容量、再经计数器回绕掩盖越界。查询结束进一步逐桶检查 `reads == writes`，并检查全局 completed 守恒。

这证明的是：**成功通过 guard 的被测查询未发生存储取模覆盖**。它不证明任意图、任意源点均不会把队列写满，也不证明 wrap 后非顺序读者的槽位复用协议。生产 DQ 仍没有通用满队列背压；不能只用 `write_reserve - bucket_read_done` 的瞬时差值替代连续释放前沿证明。

八图四配置的 **32 个诊断查询**全部通过，合计核查 **768 条逐桶记录**。最大单桶累计预留来自 **CAL / M2-shortcut / GPU0 / bucket 7**，为 **3,222,849 条**，占容量 **19.2103%**。这是本查询的累计写入上界，不是用最终计数冒充某个时刻的在途峰值；逐次 guard 保证该次运行没有越过桶容量。

逐桶统计见 [capacity_by_bucket.csv](knowledgebase/l3_supplementary_execution_20260921/capacity_by_bucket.csv) 和 [audit.json](knowledgebase/l3_supplementary_execution_20260921/audit.json)。诊断版与计时版是不同次执行；不能把 guard 版无 wrap 表述成对每一个历史计时样本的逐次动态记录。

### 诊断失败和修复过程

首次 guard 插桩使 W512 kernel 资源需求超限；第二次将寄存器限制为 112，实际 occupancy 检查仍为零。日志保留在作业 37326 和 37328 对应目录，后者在确认重复资源失败后仅取消本任务作业。错误属于诊断内核 launch/资源配置，不能算作 SSSP 已运行后的距离错误，也不能把这些样本记为通过。

最终诊断采用 `--maxrregcount=64`；ptxas 记录 worker 使用 64 个寄存器并存在 spill。实测 W512 为 `threads=544, blocks=107, resident=1, ack_slots=1712, shm=68608`。该构建用于 correctness/capacity 检查，**其 solve time 不进入正式主表**。正式二进制没有这项寄存器限制，也没有新增热路径容量 guard。最终 37329 已完成复验，之前的失败没有被删除。

## 7. 论文可直接采用的表述

> We evaluate four configurations using matched original and shortcut-augmented CSR representations. Across eight fixed-source road graphs, the complete two-GPU configuration achieves a geometric-mean speedup of 1.214× over the one-GPU solver on the original graph. Applying the same shortcuts to the one-GPU solver yields 1.278×, whereas the two-GPU speedup on the identical augmented graph is 0.949×. USA attains a same-graph speedup of 1.105×. These results attribute much of the complete-configuration gain to graph augmentation and do not establish a general additional speedup from the current two-GPU extension.

该段只描述已测 solve-time 范围；不要附加端到端收益、跨平台可移植性、任意源点收益或某个通信子机制的独立贡献。正确性部分可按第 4 节列明实际覆盖，并将第 5、6 节的作用域/容量前提保留在实现与限制说明中。

## 8. 复现入口与证据

| 内容 | 入口 |
|---|---|
| 主机 CSR 导出与宽整数 oracle | [supplement_graph.cpp](scripts/multigpu/supplement_graph.cpp) |
| 独立单卡与当前双卡适配构建 | [prepare_supplement.py](scripts/multigpu/prepare_supplement.py) |
| 四配置运行 | [run_supplement_matrix.py](scripts/multigpu/run_supplement_matrix.py) |
| 容量 guard / 同进程换源适配 | [prepare_supplement_diagnostics.py](scripts/multigpu/prepare_supplement_diagnostics.py) |
| 定向测试和容量扫描 | [run_supplement_checks.py](scripts/multigpu/run_supplement_checks.py) |
| 小型 GPU 测试构建 | [build_supplement_fixtures.py](scripts/multigpu/build_supplement_fixtures.py) |
| 独立回读原始日志的审计 | [audit_supplement.py](scripts/multigpu/audit_supplement.py) |
| 修复前数据与首轮失败 | `tmp/l3_supplement_20260921/`，作业 37325/37326 |
| 修复后正式证据 | `tmp/l3_supplement_20260921_v2/matrix/`，作业 37327 |
| 112 寄存器失败证据 | `tmp/l3_supplement_20260921_v2/checks_r112_failed/`，作业 37328 |
| 最终诊断与统计 | `tmp/l3_supplement_20260921_v2/checks/`，作业 37329；`knowledgebase/l3_supplementary_execution_20260921/` |

重建时给准备脚本传入**新的、不存在的输出目录**，避免覆盖已有证据；运行脚本使用相同目录参数。两个 sbatch 脚本显式申请双 A100 和内存，GPU 检查在分配内完成。源码快照、精确命令、二进制哈希、CSR/oracle 哈希、完整 stdout、退出码和分轮样本均保留；不以 Git HEAD 单独标识实验版本。

例如新建复现目录后，按顺序执行 `prepare_supplement.py 新目录`、`prepare_supplement_diagnostics.py 新目录`、`build_supplement_fixtures.py 新目录`，再提交两个带该目录参数的 sbatch 脚本；checks 作业用 `afterok:矩阵作业号` 依赖主矩阵。最后执行 `audit_supplement.py 新目录`。已测冻结版应从本轮归档的 `scripts/` 和源码快照恢复，而不是默认未来工作区仍与此版本一致。

Slurm accounting 在本集群未启用；37327 的控制器条目在最终查询时已过期，其实验通过依据是完整的 64 个进程退出码、384 个查询结果、`MATRIX_PASS` 与只在其 `afterok` 后启动的后续作业。37329 的控制器完成记录已保存为 [slurm_37329_final.txt](tmp/l3_supplement_20260921_v2/slurm_37329_final.txt)。本轮任务没有剩余排队或运行作业。
