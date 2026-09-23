# L3 30 小时性能执行报告

## 当前结论

当前已完成阶段 0、阶段 1、Route A、Route B1 与 Route B2 的探索和止损，进入
**最终源码冻结与定向验证（计划 21--24 h 段）**。Route C 因作业 38122 的
`PATH`、`/usr` 与 `/opt` 探针未找到可用 NVSHMEM 安装而按止损条件停止；这不是
机器范围的“绝对未安装”声明。compact-candidate 只有小幅探索信号且仍低于目标，
不进入最终候选；最终候选是 USA、delta=400000、blocks=107、W512、固定
25000-cycle 的 control 配置。

30 小时 GPU 窗口从 Slurm 首次实际授予两张 A100 的
`2026-09-23 15:43:29 +08:00` 起算。下文 Stage 1 性能数均是
dirty-worktree、`07df3ad` 回退同步运行时上的探索样本（每进程 1 次预热 + 3 次
正式）；它们可以用于淘汰或选候选，不能替代恢复同步后的 clean-SHA 两轮反序
正式结果。

## 阶段状态

| 阶段 | 状态 | 结论/证据 |
|---|---|---|
| 版本与任务书固定 | 已完成 | 分支 `l3-30h-performance-20260923`；任务书 HEAD `be4f6b2`；历史运行时基线 `07df3ad` |
| 38082 原始证据导出 | 已完成 | 64 个进程、384 个查询、320 个正式有效样本；`evidence/l3_latest_rerun_38082/` |
| 六图 CPU 数据准入 | 已完成 | 六图格式全量扫描；`nlpkkt80` 因负权淘汰；其余五图 oracle 已生成；transient-add 仍由最终 checked-add GPU 门关闭 |
| 最终 USA G+ 数据准入 | 已完成 | 对精确哈希 `85c273...` 的 66,684,784 条边做全量流式扫描；非负 int32、CSR/长度/目标范围通过 |
| 双 A100/拓扑预检 | 已完成 | 作业 38122：2x A100 80GB、NV12、双向 P2P/native atomics、预检时无 GPU 进程 |
| Route C 条件检查 | 已停止 | 在该 allocation 的固定搜索范围内未发现 NVSHMEM 头文件或库，仅发现 `mpirun` |
| 五图探索与参数校准 | 已完成 | USA 是唯一稳定正扩展主攻图；RGG 按要求保留为 B1/B2 对照 |
| Route A | 已完成并止损 | blocks、backoff、固定短窗口均未形成可冻结净收益 |
| Route B1 | 已完成并止损 | RGG BFS 有小幅改善；USA layer-split 严重退化；重排未达到 1.20 |
| Route B2 | 已完成并止损 | work/wait 诊断完成；compact-candidate 有小幅信号但仍未达到 1.20，最终拒绝 |
| 同步/容量实现收口 | 已实现，待正式验证 | 恢复 `567e47f` 默认 BULK 活跃路径同步合同；固定 25k；加入生产计数 overflow guard 与双卡完成 barrier；dirty 双 A100 smoke 通过 |
| clean-SHA 定向验证 | 未开始 | 预冻结 dual/single 实际 NVCC pair 已编译，46/46 CPU 合同单测通过；必须先提交 clean SHA，不采信 dirty 产物 |
| 正式验收 | 未开始 | clean-SHA、两轮反序、独立作业确认、附加源点和八图回归仍待执行 |

## 38082 历史批次

对保留目录 `tmp/l3_no_sync_rerun_20260923/` 的 records、manifest、原始日志、
图/oracle 和二进制哈希重新审计后，内部一致性检查通过：

- 同原 G 的单卡/双卡几何平均：`0.895482x`；
- 同 G+ 的单卡/双卡几何平均：`0.959050x`；
- USA G+：`75.324406 / 69.102925 = 1.090032x`；
- 原 G 单卡 / G+ 双卡的交叉比较：`1.211661x`，不属于纯扩展收益；
- 原 G 单卡到 G+ 单卡本身有 `1.263398x` 算法/图变换收益，必须与扩展收益分开。

该批使用回退同步版本。查询输出正确不构成跨设备 memory-order 证明，不能与
最终候选混批。

同样地，作业 38131--38180 的全部新探索也基于回退同步运行时。其
`measurement_valid` 只表示进程返回码、样本、oracle 与当时日志合同有效，不是
跨设备 ordering 安全证明。

## 数据准入与探索基线

六个固定输入均为格式有效的 GR v1/4-byte payload。`nlpkkt80` 含
26,618,272 条负权边，已在 GPU 前淘汰。`atmosmodm` 与 `rmat22` 经源点 0 的
int64 Dijkstra 核实后可安全转为当前 int32 oracle；RGG 和 RMAT 的不可达语义
显式保留。文件字节只能证明编码与数值范围，不能单独证明数据转换来源。
最终 USA G+ 另以 `usa_augmented_dataset_manifest.json` 全量核对；`atmosmodm`、
`rmat22` 与 USA 三个冻结源点的每次活跃候选加法仍必须由最终 clean pair 的
checked-add 派生构建运行后，才能把数值合同由 `NOT_PROVEN` 改为通过。

作业 38131 在同一 allocation 内串行运行五组配对，全部 rc=0、oracle PASS：

| 输入 | T1 solve median ms | T2 solve median ms | `S=T1/T2` | 决定 |
|---|---:|---:|---:|---|
| USA G+ | 74.819361 | 69.067022 | 1.083286 | 主攻 |
| atmosmodm | 5.638706 | 6.333652 | 0.890277 | 校准后淘汰 |
| rgg_n_2_20_s0 | 9.005817 | 10.164907 | 0.885971 | 保留布局/结构对照 |
| delaunay_n23 | 12.740569 | 7354.439989 | 0.001732 | 淘汰 |
| rmat22 | 7.520888 | 1507.891725 | 0.004988 | 淘汰 |

`delaunay_n20` smoke 的 `S=0.008676`，同样因固定与通信开销远大于计算而淘汰。

## Route A：参数与窗口

USA 的 delta=400000、blocks=107 是本路线最好探索点：作业 38136 为
`75.977491 / 63.956841 = 1.187949`；另一次 control 重跑为
`76.572622 / 64.158405 = 1.193493`。二者均接近但未达到 1.20，且不是正式样本。

- USA blocks=80 为 `1.177460`，blocks=64 为 `0.825310`；
- idle backoff 使 USA 降至 `1.1533`，RGG 为 `0.9036`；
- 固定 12500/6250 cycle 窗口的 USA 均约 `1.158`，RGG 也未改善；
- RGG delta=32 的双卡 solve 最快约 9.596 ms，但以严格最好单卡 9.006 ms 比较仍
  只有约 `0.939`。

因此 Route A 保留窗口参数化、有效性检查和 ACK 几何日志这些工程修正，但回退
backoff/短窗口性能候选，不用参数笛卡尔积追逐噪声。

作业 38145 的首次 USA control case 使用了不存在的旧相对路径，rc 非零并完整
保留；同一作业中的 RGG 成功，USA 随后在 38147 以冻结源点和正确路径重跑。失败
不是 0 ms，也未被后续重跑从候选账本中删除。

## Route B1：布局

新增的全量 verifier 检查 perm/invperm、每一 CSR 行、全部边与 payload，并可
接 oracle。三种生成布局均通过结构等价性检查。

- RGG 原布局跨区边 17,822；BFS 布局 15,580（-12.6%）；layer-split 199,778。
  BFS 样本为 `9.264663 / 9.541566 = 0.970980`，相对原布局有所改善但仍未达到
  1.20；按最好单卡/该双卡严格比较约 `0.924`。
- USA 的 Route-B1 GPU 对照使用 G-only 原图（不是 G+）。layer-split 将跨区边从
  9,760 增至 320,056，
  实测 `98.352761 / 488.084645 = 0.2015`，立即淘汰；BFS-on-current 的静态
  跨区边也增至 15,512，未进入 GPU 计时。

## Route B2：诊断与 compact-candidate 原型

work/wait 热路径诊断只能使用 W320；W512 因 kernel resource、W384 因 resident=0
失败，失败证据均保留。W320 诊断显示 USA 两卡合计约 956.5M 次本地边扩展、
203,735 次 cross update，跨卡更新只占约 `0.0213%`，说明当前瓶颈主要是重复的
本地工作/进度，而不是可直接由传输带宽解释的成本。诊断构建改变了占用，不能拿
其 223.7 ms solve 做性能比较。

`L3_COMPACT_CANDIDATES` 只压缩 boundary-candidate bitmap 的扫描域，并未缩小
全部分配，也没有把 owner 的完整恢复域错误裁成边界域。USA 的 bitmap words 从
700/662 降至 137/108，RGG 从 1546/1596 降至 75/77。原始 W512 compact 构建
寄存器过高，作业 38177 launch 失败；随后 control 与 compact 同时使用
`--ptxas-options=-maxrregcount=96`，保持相同 single 二进制与其他参数，在作业
38180 串行公平对照。

| 输入/构建 | T1 solve median ms | T2 solve median ms | `S` | measurement |
|---|---:|---:|---:|---|
| USA control-r96 | 76.005804 | 65.958054 | 1.152335 | valid, target false |
| USA compact-r96 | 75.402160 | 64.782621 | 1.163926 | valid, target false |
| RGG control-r96 | 8.721809 | 10.412447 | 0.837633 | valid, target false |
| RGG compact-r96 | 9.266393 | 9.829623 | 0.942701 | valid, target false |

按同轮 control/compact 的双卡中位数比较，compact 对 USA 改善约 1.8%，对 RGG
改善约 5.9%。这是探索信号，不是正式速度结论：两个 sweep 虽处于同一 allocation
并串行执行，但没有把 control/compact 本身做反序交错；且 compact 将目标
SLF+delta 工作核从 84 个寄存器推至 96 个并增加栈开销。两图均未达到 1.20，
因此该原型不直接晋级正式候选。

全部 job 38180 运行日志、样本、manifest、构建命令/状态/哈希已归档到
`evidence/l3_30h_20260923/stage1/route_b2_control_r96_job38180/` 和
`evidence/l3_30h_20260923/stage1/route_b2_compact_r96_job38180/`；未归档大二进制
与 source tarball。

## 最终冻结候选与同步边界

最终候选恢复提交 `567e47f` 的完整同步修复，并只叠加本轮固定窗口参数化、有效
配置日志和查询后容量门禁。默认 BULK 活跃路径的 candidate/mark、payload/READY、
RX/L2/ACK、DQ publication、retained/requeue、worker recovery 与终止链已经逐项
静态审计；字段、发布点、观察点和作用域记录在
`evidence/l3_30h_20260923/final/sync_contract.md`。结论不外推到未启用的
`SEED_BARRIER`、tile-loan、owner-commit 或其他实验分支。

正式构建对每个逐桶权威 DQ `write_reserve` RMW 启用
`DQ_COUNTER_OVERFLOW_GUARD`：原子操作返回的旧值与 int64 新值必须在该桶
容量/`INT_MAX` 内，否则置 sticky flag 并 fail-stop。后续 read/completion 计数由
唯一写预留上界约束，并由 `total_capacity <= INT_MAX` 和终态精确守恒交叉验证；
independent single 使用相同共享-core guard。`mlmq_benchmark::finish` 在计时终点后
作为两卡 host barrier；两卡都完成 `cudaDeviceSynchronize` 后才读取 `L2_FINAL`。
最终门强制逐桶 drain、guarded writes/总读写/完成守恒、容量交叉、32 位配置、
`overflow_detected=0` 和 `no_wrap=1`。这些检查不会被移出证据边界；并发发布测试仍
不能由终态守恒替代。

预冻结脏树上的 W512 双 A100 smoke（delaunay_n20）已 `rc=0`，两卡
`WIDE_ORACLE correct=1`、`FINAL_AUDIT` 零 mismatch/residual，且新 `L2_FINAL`
合同均为 guard=1、detected=0、no-wrap=1。当前正式 dual 工作核的编译资源
为 78 寄存器/72-byte stack，W512 可启动。这只是预冻结工程门，不是
clean-SHA 验收或性能证据。

## 工程口径

- runner 显式接受图、oracle、源点、delta、cut、blocks、期望窗口配置和采样
  次数，在同一双卡 allocation 内交错运行独立 single 与真实 dual，无单卡
  fallback。
- `measurement_valid` 与 `target_met` 分离；正式模式还要求 clean worktree、完整
  期望参数、精确 pair-build provenance、两轮反序采样和每个双卡样本的 L2
  conservation/no-wrap 门禁。
- 独立 single 在 staging 源码中使用同名且带范围检查的 `MLMQ_WORK_BLOCKS`，并以
  `NO_L3_LAUNCH` 记录实际值；不是把 dual 二进制用 `-n 1` 伪装成基线。
- 验收仍以 solve 中位数为准。query-wall、setup 或预处理收益单独报告，不能替换
  `S=median(T1 solve)/median(T2 solve)`。
- GPU 性能作业串行独占；CPU 审计、构建、测试和报告整理可并行。

## 下一步

1. 将已恢复同步、固定 25k 且带精确容量门禁的 control 候选提交为 clean SHA，
   从该 SHA 独立构建 single/dual pair。
2. 在最终 SHA 上运行 candidate 交错、满槽重试、延迟 ACK、RX 提交边界、DQ
   publication、capacity full/overflow、顺序换源/reset 和最终完成检查。
3. 即使探索值低于 1.20，也按任务书采集两轮反序、每轮 1+5 的正式配对样本，
   给出达标或未达标结论；不得把探索最好值或 query-wall 比值冒充正式通过。
4. 正式候选完成后再运行附加源点和八图正确性回归；未执行项明确标 NOT_RUN。
