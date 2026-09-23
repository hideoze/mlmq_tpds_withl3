# L3 最小收尾任务：交给服务器 agent

## 范围与基准版本

审查对象：`hideoze/mlmq_tpds_withl3`，提交 `5f5430e66e863e9a591311de1185e2226492d5c1`，以及本次提供的 `TPDS.pdf`。

本文件依据 GitHub MCP 读取的源码、构建命令及归档结果制定。没有在本次审查中重新运行 GPU 测试；服务器未提交的工作区不属于已核查范围。

目标：维持 L3 为一个小规模的两 GPU 扩展，只处理同步保证、结果表述和论文配置复现三个问题。不要开展新架构、更多 GPU、新图类别、大规模多源点扫描或新的多 GPU 基线工程。论文不新增算法、主图或长篇证明。

## 已有工作：不要按“缺失”重做

- `L3_adds_samegraph_sixconfig_results.md` 已有 ADDS / 独立单 GPU MLMQ / 双 GPU MLMQ 在 G、G+ 上的六配置比较。
- `knowledgebase/l3_supplementary_execution_20260921/audit.json` 记录 MLMQ 384 个查询、320 个正式样本、32 个容量诊断查询，以及双向槽位重试、延迟 ACK 和提交边界检查。
- `knowledgebase/l3_supplementary_execution_20260921/timing_dispersion.csv` 已有十样本中位数、四分位数、IQR、MAD、最小值和最大值。
- `SSSP/l3/l3_receive.cuh::l3_receive_to_l2` 已实现严格距离改进后写入 L2，再调用 `bulk_inbox_finish_read`。不要把“需要补充文字说明”误当成“需要重新实现先提交后 ACK”。
- `SSSP/l3/l3_chain_shortcuts.h` 保留原始顶点/边，使用整数路径权重、宽整数累加，并跳过达到 INT_MAX 的 shortcut 路径。无需新增浮点 shortcut 方案。

以上是当前归档版本的证据，不自动证明后续修改版本，也不是任意执行下的弱内存正确性证明。

## P0：完成同步发布/观察关系的核查与必要修复

这项属于实现收尾，不是新增研究内容。先检查当前访问是否已有可依据 CUDA 12.4.1 文档说明的等价同步；无法说明的地方再做针对性修复，不预设必须重构整个系统。

重点位置：

| 交接 | 现有位置 | 收尾要求 |
|---|---|---|
| candidate 更新 / mark 发布 / candidate 领取 | `SSSP/l3/l3_candidate.cuh`、`SSSP/l3/l3_collect.cuh` | 明确同 GPU 的发布和获取关系。当前代码是无后缀 atomicMin/atomicOr、CAS/Exch 组合，不能仅凭源码书写顺序宣称保证成立。 |
| payload / count / generation / READY，以及 ACK / DONE / 槽位复用 | `SSSP/l3/l3_bulk.cuh`、`SSSP/l3/l3_transport.cuh` | 明确跨 GPU 的同步作用域和发布/获取关系；保留 generation 与 ACK 复用条件，并覆盖实际读 payload 的各 lane。 |
| fallback、候选重挂、终止控制和 L3 到 L2 的提交交接 | `SSSP/sssp_run.cu`、`bulk_requeue_l3_batch`、`l3_receive_to_l2` 及其队列提交路径 | 核对同一字段的所有入口，不能只修 cooperative collector 或正常接收分支；终止控制所依赖的 peer state/ACK 也须使用一致的同步合同。 |

实现可采用适当 device/system scope 的 acquire/release 原语，或有文档依据的等价同步。不要把所有原子无差别替换为 system-scope；仅添加 `_system` 也不等于同时解决 memory order。区分参与正确性的字段和纯调度提示，避免无谓增加热路径成本。

核查依据：NVIDIA CUDA C++ Programming Guide 12.4.1 的 Atomic Functions 与 Memory Fence Functions，以及仓库 `L3_supplementary_experiments_results.md` 第 5 节。该报告已经明确记录同步保证尚未核实完整；不能以原有测试通过为由直接改成“全部解决”。

验收：提供一个简短的字段/发布点/观察点/作用域与顺序表，并链接实际代码位置。仍不能说明的关系必须如实列出，不要求新增正文定理。

## P0 后续：复用已有测试，只更新受影响的实验

1. 使用新的输出目录保留新版本源码、构建参数和结果，不覆盖原有 evidence。
2. 复用 `scripts/multigpu/run_supplement_checks.py`、`test_l3_supplement.cu` 及现有容量/换源测试。核对 candidate 交错、满槽重试、ACK 延迟、L2 提交边界、最终完成和逐顶点 oracle 结果。
3. 如果实际修改了运行时同步路径，用最终二进制重新测量受影响的 MLMQ 配置，沿用原有八图、G/G+、固定源点、两轮交错、每轮一次预热和五次正式测量。不要扩展实验维度。
4. 若修改共享 `core/`，同步修改独立单 GPU 基线并重测对应结果；不能只增强双 GPU 路径或用普通 `-n 1` 代替既有独立单卡基线。
5. ADDS 未改动时不要求重做基线实现。其旧结果能否复用，应核对输入哈希、硬件、配置、计时协议与运行条件，并保留批次来源。新程序不得继续冒用旧二进制的时间。

测试通过只说明相应测试覆盖的结果；同步保证的说明不能被测试数量替代。

## P1：改结果表述，不增加主图

当前冻结版已有结果：

| 比值 | 当前结果 |
|---|---:|
| 单 GPU MLMQ(G) / 双 GPU MLMQ(G) | 0.9037x，八图几何平均 |
| 单 GPU MLMQ(G+) / 双 GPU MLMQ(G+) | 0.9494x，八图几何平均 |
| 单 GPU MLMQ(G+) / 双 GPU MLMQ(G+)，USA | 1.1047x |
| ADDS(G+) / 双 GPU MLMQ(G+) | 约 1.7035x |

G+ 上只有 USA 的单卡到双卡比值大于 1。保留相对 ADDS 的积极结果，但明确它不是第二张 GPU 的净增量收益；shortcut 的收益也不是仅由 L3 才能取得。

正文只替换 V-F 中现有比较与解释的几句话。不必新增六列表格或实验小节。可以删去跨 G/G+ 的 2.242x 重复强调，腾出同图 scaling 的位置。

可参考的短段落（仅适用于当前冻结数据；代码变更后必须由最终结果更新）：

> On identical shortcut-augmented inputs, the two-GPU configuration achieves a geometric-mean speedup of 1.703x over single-GPU ADDS. Relative to one-GPU MLMQ, same-input speedups are 0.904x on G and 0.949x on G+, with USA reaching 1.105x on G+. Shortcuts improve the two-GPU solve time by 1.343x and also benefit the single-GPU solvers. These results demonstrate a working two-GPU extension while showing that additional speedup remains workload dependent.

保持 solve-only 边界明确，不把上述结果改称端到端加速。论文中的同步措辞须与最终实现相符；没有修复完成前，不声称已有明确的 release/acquire 实现。

## P1：修正论文配置的构建入口，并补一条验证说明

`README.md` 当前给出 `cd SSSP && make`，但 `SSSP/Makefile` 没有论文冻结构建使用的 L3 功能宏，默认产物名称也是 `main`。它不等于正式实验构建。

以 `evidence/l3_supplement_20260921_v2/dual_build/command.json` 为依据，提供或准确指向一个论文配置构建入口。保留实际宏配置、worker threads、队列、delta、分区与源点信息，明确图数据和参考结果如何定位；允许修正本机路径，不要静默改变算法开关。不要顺带做大型打包或容器化工程。

正文可用一句话交代已做的逐顶点验证与定向并发测试，详细结果留在仓库。修复后的版本只能在实际通过后写“通过”。现有容量修复与检查结果不需要重新扩展成一个论文贡献。

## 最终交付与停止条件

只交付：必要的同步补丁、准确的论文配置构建入口，以及一份简短的 `L3_minimal_revision_report.md`。报告记录最终版本、同步关系、实际回归结果、最终论文数字及拟替换的短段落。原始日志继续沿用现有归档方式。

停止条件：最终实现的关键发布/观察关系能够说明；现有相关回归通过；性能数字与实际受测版本一致；正文明确区分 ADDS 对照、shortcut 收益和单卡到双卡净收益。完成后不要继续增加多 GPU 基线、更多图、多源点性能矩阵、完整消融或长篇协议描述。
