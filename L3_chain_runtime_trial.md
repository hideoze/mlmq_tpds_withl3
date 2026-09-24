# L3_CHAIN_PARTITION 运行时链处理限时验证

日期：2026-09-24
结论状态：`VALID_BELOW_TARGET / STOP_RUNTIME_CHAIN_BRANCH`

## 1. 结论

本轮在原始 CSR `G` 上完成了可归因的 A/B/C/D 正式验证。正确性成立，但性能目标不成立，因此停止扩张这条 runtime-chain 路线，不将其并入当前最佳配置，也不继续 RGG 或八图性能矩阵。

- Slurm 作业 `38312` 使用同一节点的 2×NVIDIA A100 80GB PCIe，NV12，终态 `COMPLETED/0:0`。
- 30 个进程记录、138 次查询全部 `rc=0`、`correct=1`；外部逐顶点 oracle、双卡协议、L2 守恒/容量/no-wrap、输入和二进制哈希门全部通过。
- 主口径是包含完整距离 D2H 物化的 `query_wall_ms`，每个性能配置有 2 轮×5 个正式样本，所有慢样本均保留。
- USA：`T(B)/T(C)=0.8471`，即 C 比匹配 B 慢 `18.05%`；虽然 `T(A_mlmq)/T(C)=1.4046`，但优势并非来自 chain，因为关闭 chain 的 B 更快。
- COL：`T(B)/T(C)=0.9748`，C 比 B 慢 `2.59%`；`T(A_mlmq)/T(C)=0.8331`，同图双卡目标也失败。
- solve-only 口径下 C 在 USA/COL 都慢于单卡 MLMQ，`A/C` 分别只有 `0.8261/0.8088`。
- USA/COL 的重复整链展开占 closure 的 `93.10%/93.30%`，同一 segment 单查询最多访问 `385/116` 次。入队抑制确实发生，但重复展开和热路径资源成本抵消了收益。

正式汇总见 [summary.json](evidence/l3_chain_runtime_20260924/final/summary.json)，终态判定见 [complete.json](evidence/l3_chain_runtime_20260924/final/complete.json)。

## 2. 版本、隔离与改动边界

- 分支：`l3-chain-runtime-20260924`
- 正式 clean SHA：`a91a61e5c8949b29dccaea1e8d8abc242a71a7de`
- 任务文档父提交：`f1547173ab869ce86e3a1f427b4ff336a6a6efac`
- 工作树在构建前、运行前和运行后均为空；[final_integrity.json](evidence/l3_chain_runtime_20260924/final/final_integrity.json) 重新核验了 HEAD、合同、构建 manifest、六个二进制和四个输入哈希。
- CUDA：12.4 / nvcc 12.4.131；nvcc SHA256 `bbcea3024f3ada9aafcbf9a8370c14dad1f4fb04b48147ad83dd7e13854ba0ed`。
- 本轮没有构造或读取 G+；合同显式拒绝已知 augmented SHA。

提交 `a91a61e` 的最小实现/验证改动如下：

1. `SSSP/l3/l3_chain_partition.h`
   - 严格链内部点现在要求两个不同邻居各有且仅有一条 reciprocal incoming edge。
   - 修复旧聚合计数可能把同一邻居的两条平行入边误当成严格链、随后构建异常退出的问题；非严格拓扑回退普通 CSR。
2. `SSSP/l3/l3_chain_partition.cuh`
   - 索引安装发生在 benchmark enable 之前，因此 CUDA 分配、上传、symbol bind、释放和诊断 reset 改为无条件检查；失败打印 `L3_CHAIN_CUDA_ERROR` 并终止。
   - 诊断版增加逐 segment 访问数组，并单列其额外显存。
3. `SSSP/sssp_run.cu`
   - 增加 `unique_segments`、`repeated_closures`、`max_segment_visits`，并强制 `closures=unique+repeated`。
4. `SSSP/sssp.cuh`
   - 增加 diagnostics 必须依赖 chain partition 的编译期门。
5. `scripts/multigpu/`
   - 修复旧 `.gr` 解析器把 Galois v1 的 n 个 cumulative row ends 误读为 n+1 个的问题。
   - 新增 exact runtime-builder 分析器、206-fixture CPU 测试、clean-SHA 构建器和 Slurm-only 正式 runner。

RX tag、内部点抑制、端点继续传播、`last_processed`、ACK/termination/recovery 主协议没有重写。静态审计确认在非负 int、owner-local、简单严格双向链的受支持域内，未发现遗漏必要松弛或提前 termination 的阻断性路径；最终结论仍以本轮可执行验证为准。

## 3. 实验合同与对照

[formal_contract.json](evidence/l3_chain_runtime_20260924/formal_contract.json) 在本轮 GPU 性能运行前提交并冻结。

| 角色 | GPU | 配置含义 |
|---|---:|---|
| A_mlmq | 1 | 独立 no-L3 单卡 MLMQ，W512/BNUM8/batch8，原始 G |
| A_adds | 1 | 冻结的官方 ADDS；binary SHA256 `511e88f...38ec2` |
| B_control | 2 | 当前数据面，term-only worker，runtime chain 关闭 |
| C_chain | 2 | 与 B 相同，只开启 runtime chain；正式计时关闭诊断 |
| D_best | 2 | 原计划当前最佳参考，正常的非 term-only worker 路径 |
| C_diagnostic | 2 | C 加诊断计数，仅用于正确性/命中分析，不进入性能统计 |

[bc_command_contract.json](evidence/l3_chain_runtime_20260924/build/bc_command_contract.json) 对完整 nvcc argv 归一化后确认 B/C 除 `L3_CHAIN_PARTITION=false/true` 外完全一致。B/C 都显式采用：W512、BNUM8、BUCKET_MAX4、batch8、fixed window 25000、DIRECT_RX、RETAIN_TX、WORKER_RECOVERY、ACK_SCAN、IDLE_TOKEN、TERM_ONLY_WORKER 和 degree gate；静态 shortcuts 关闭。

目标队列 `L1SLF_L2DQ + l2_delta_queue` 的资源用量为：

| 构建 | REG | STACK (bytes) | SHARED (bytes) |
|---|---:|---:|---:|
| B_control | 76 | 72 | 1040 |
| C_chain | 85 | 72 | 1040 |
| C_diagnostic | 85 | 72 | 1040 |

C 相对 B 增加 9 个寄存器；W512 可以正常编译和启动，没有为了 chain 改成 W320。

## 4. 输入与预选

两张正式图均为已冻结布局的原始 `G`：

| 图 | V | E | source | delta | cut | graph SHA256 | oracle SHA256 |
|---|---:|---:|---:|---:|---:|---|---|
| USA | 23,947,347 | 57,708,624 | 11,973,673 | 400,000 | 60% / 14,368,408 | `7a4e9747...33b8d7` | `19b674d7...757a44` |
| COL | 435,666 | 1,057,066 | 217,833 | 200,000 | 55% / 239,616 | `7158d69e...9b3f02` | `095a726b...9e018` |

已知 USA/COL G+ SHA 分别为 `85c27390...d6eb` 和 `ef77be4c...59e19`，均不等于正式输入；runner 还在运行前后重新计算输入哈希。

第二张道路图在当前性能结果产生前，按 exact current-builder 的 indexed-interior 覆盖率从八张道路图中选定：

| NY | BAY | FLA | E | CAL | W | USA | COL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6.6091% | 9.2495% | 10.3032% | 13.9247% | 14.4461% | 16.5694% | 17.7244% | **19.1867%** |

因此第二图固定为 COL，而不是观察速度后挑图。RGG 已纳入候选考虑，但合同要求先完成第二张道路图；由于 USA 和 COL 的 matched B/C 均为负收益，按止损规则不再扩展 RGG。

## 5. 正确性与协议验证

### 5.1 CPU 与编译门

- CPU exact-builder/model：`206` 个 fixture、`2050` 个被索引内部点，全源距离一致。
- 覆盖：双向链、两个方向权重不同、零权、累计权重溢出回退、并行边非严格拓扑回退、偏移 owner、重复 build/reset。
- clean-SHA 五套 CUDA 构建全部 rc=0；CPU 测试日志见 [test_l3_chain_partition.log](evidence/l3_chain_runtime_20260924/build/tools/test_l3_chain_partition.log)。

### 5.2 GPU 定向门与正式图

在作业 `38312` 内，性能计时之前依次通过：

- reciprocal/asymmetric：两个 owner 上的长双向异权链，两端源点；
- zero：零权双向链；
- branches_concurrent：链端点/内部点的后续与竞争改进；
- parallel_fallback：旧实现可能误接纳的平行入边拓扑，closure 必须为 0；
- overflow_fallback：索引 setup 必须记录 overflow，closure 必须为 0；
- USA/COL C-diagnostic：逐顶点 oracle 正确且实际 closure/materialization 非零。

总计：

- 30/30 进程有效，138/138 BENCH 查询正确；
- 102 条 `WIDE_ORACLE`（含独立单卡每进程的 trailing full audit）；
- 180 条两卡 `L2_FINAL`，全部 reads=writes=completed、overflow_detected=0、no_wrap=1；
- 36 条诊断 GPU counter，全部满足 closure accounting；
- 未出现 `Error at node`、`ORACLE_MISMATCH`、`FINAL_AUDIT_ERROR`、`L3_CHAIN_CUDA_ERROR` 或 `correct=0`。

GPU 为两张 A100 80GB CC8.0，UUID 和 NV12 拓扑见 [gpu_query.log](evidence/l3_chain_runtime_20260924/final/gpu_query.log) 与 [gpu_topology.log](evidence/l3_chain_runtime_20260924/final/gpu_topology.log)。`sacct` storage 未启用，但作业结束后立即保存的 [slurm_job_terminal.log](evidence/l3_chain_runtime_20260924/final/slurm_job_terminal.log) 明确给出 `JobState=COMPLETED`、`ExitCode=0:0`；原始限制见 [sacct_unavailable.log](evidence/l3_chain_runtime_20260924/final/sacct_unavailable.log)。

## 6. 正式性能

所有配置在同一独占 allocation 中串行执行。每个 graph/config 启动两个进程，每进程 1 warmup + 5 formal；第二轮同时反转 graph 和 configuration 顺序。表内是 10 个正式样本的 `median (IQR)`，单位 ms。

### 6.1 主口径：完整 query wall

| 图 | A_mlmq | A_adds | B_control | C_chain | D_best | B/C | A_mlmq/C | D/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| USA | 177.583 (5.461) | 145.498 (0.569) | 107.098 (0.556) | 126.432 (19.246) | 107.437 (1.411) | **0.8471** | **1.4046** | 0.8498 |
| COL | 11.239 (0.229) | 25.290 (0.106) | 13.150 (0.470) | 13.490 (0.303) | 13.035 (0.384) | **0.9748** | **0.8331** | 0.9662 |

`B/C` 是 runtime chain 的净收益；两个图都小于 1。C 相对 B 的退化为 USA `18.05%`、COL `2.59%`。

两轮中位数也保持相同方向，不是单轮顺序偶然：

| 图 | B r0 / r1 | C r0 / r1 |
|---|---:|---:|
| USA | 107.211 / 106.986 | 127.186 / 125.679 |
| COL | 13.107 / 13.192 | 13.463 / 13.771 |

USA 的 A/C 达到 1.20，但不能据此认定 chain 有收益：matched B 比 C 更快，D 也比 C 更快。COL 的 A/C 则直接低于 1。合同要求在两图上稳定达到目标，因此总体 `target_met_on_all_graphs=false`。

### 6.2 次口径：solve-only

| 图 | A_mlmq | A_adds | B_control | C_chain | D_best | B/C | A_mlmq/C |
|---|---:|---:|---:|---:|---:|---:|---:|
| USA | 96.118 | 136.362 | 97.897 | 116.345 | 98.172 | 0.8414 | 0.8261 |
| COL | 8.660 | 25.021 | 10.132 | 10.707 | 10.099 | 0.9463 | 0.8088 |

solve-only 同样显示 C 慢于 B 和单卡 MLMQ。USA C 的 10 个 solve 样本范围为 `102.795–169.656 ms`，慢尾全部保留；对应 IQR 为 `16.840 ms`。

## 7. 链覆盖、命中与重复展开

静态索引统计：

| 图 | segments | indexed interiors | 覆盖率 | mean interiors/segment | p50 / p95 / max | 生产索引显存 | 诊断 visits 额外显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| USA | 1,521,266 | 4,244,523 | 17.7244% | 2.790 | 2 / 5 / 127 | 308,171,672 B (293.90 MiB) | 29,148,220 B (27.80 MiB) |
| COL | 28,713 | 83,590 | 19.1867% | 2.911 | 2 / 6 / 34 | 5,741,600 B (5.48 MiB) | 564,064 B (0.54 MiB) |

单次不计时道路图诊断：

| 图 | RX-derived sources | route reads | closures | unique segments | repeated closures | repeat share | max visits | materialized interiors | queued tails |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| USA | 216,152,966 | 44,169,397 | 15,852,413 | 1,093,044 | 14,759,369 | **93.10%** | 385 | 23,834,178 | 10,432,430 |
| COL | 3,432,003 | 726,133 | 253,477 | 16,980 | 236,497 | **93.30%** | 116 | 417,266 | 178,167 |

chain 确实命中并物化内部点：USA/COL 每次 closure 平均成功物化 `1.504/1.646` 个内部点。但大部分 closure 是已访问 segment 的重复整链展开，且索引覆盖仍不到 20%。这与 C 的 9-register 热路径增量共同解释了“减少部分内部入队但完整时间反而增加”的结果。

## 8. 索引构建、上传与额外显存

构建和上传发生在查询计时之外，分别报告。每图有 diagnostic、r0 C、r1 C 三个独立进程；下表为两卡合计的 `median [min,max]`。

| 图 | index build ms | upload ms | production bytes |
|---|---:|---:|---:|
| USA | 1785.763 [1775.438, 1788.537] | 27.329 [27.312, 27.484] | 308,171,672 |
| COL | 30.481 [30.288, 30.545] | 6.547 [6.133, 6.554] | 5,741,600 |

诊断版 visits 数组不属于正式 C，已在上一节单列。由于 C 的每查询时间本身已经慢于 B，索引一次性成本不存在正的 amortization/crossover；把 build/upload 排除出 query wall 也不能改变停止结论。

## 9. 与已有原型证据的关系

本轮没有无分析地重复旧实验：

- 历史 Stage230（作业29262，`COMPLETED/0:0`）已经证明原始 G runtime chain 正确，但 solve-only matched control/chain 为 NY `0.906`、W `0.916`、USA `0.883`，均为负收益。
- Stage231 的 degree gate 相对 gate-off 在 USA solve-only 回收约 5.27%，NY/W 不到 1%，但该作业缺少调度器终态，整体 evidence 明确标为 invalid；它也没有证明相对 matched no-chain B 的正收益。
- 当前复核补上了最新 W512/BNUM8 数据面、严格 reciprocal 判定、安装期 CUDA 强检查、重复 segment 诊断、COL 预选和完整 query-wall 统计。结果仍与历史失败方向一致。

历史原文位于：

- `/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/knowledgebase/l3_dynamic/230_chain_partition.md`
- `/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/knowledgebase/l3_dynamic/231_partition_route_gate.md`

## 10. 完整构建与运行命令

工作目录：`/mnt/709/data3/home/Dingzhong/.codex/worktrees/bbd4/mlmq_tpds_withl3`

构建入口：

```bash
env -u LD_LIBRARY_PATH \
  NVCC=/usr/local/cuda/bin/nvcc \
  BOOST_INCLUDE_DIR=/a100-data/wyh/boost_1_87_0 \
  scripts/multigpu/build_l3_chain_trial.sh \
  /mnt/709/data3/home/Dingzhong/l3_chain_runtime/a91a61e/build
```

每个二进制的完整 nvcc argv：

- [A single command](evidence/l3_chain_runtime_20260924/build/reference_pair/single_build/command.json)
- [D current-best command](evidence/l3_chain_runtime_20260924/build/reference_pair/dual_build/command.json)
- [B control command](evidence/l3_chain_runtime_20260924/build/dual_control/command.json)
- [C chain command](evidence/l3_chain_runtime_20260924/build/dual_chain/command.json)
- [C diagnostic command](evidence/l3_chain_runtime_20260924/build/dual_chain_diag/command.json)

正式运行入口：

```bash
srun --partition=a100 --nodes=1 --ntasks=1 \
  --gres=gpu:a100:2 --cpus-per-task=8 --mem=64G --exclusive \
  --time=02:00:00 --job-name=l3chain-a91a61e \
  python3 scripts/multigpu/run_l3_chain_trial.py \
  --build-dir /mnt/709/data3/home/Dingzhong/l3_chain_runtime/a91a61e/build \
  --adds /mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/tmp/adds_official_v4/adds \
  --spec USA \
    /mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/tmp/landmark206_matrix/USA/landmark.gr \
    /mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/tmp/l3_supplement_20260921_v2/graphs/USA/oracle.i32 \
    11973673 400000 60 \
    7a4e9747b3e1febc4ad452e1b0156e636882482b34d2462f8308e70de533b8d7 \
    19b674d79f5bc48d26107facb6327c8f8661d40a5bb80db67dd8feb260757a44 \
  --spec COL \
    /mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/tmp/landmark206_matrix/COL/landmark.gr \
    /mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/tmp/l3_supplement_20260921_v2/graphs/COL/oracle.i32 \
    217833 200000 55 \
    7158d69e3e6c8fb9b9b699de1a423c0a21eab146793ea865cafb25a4d59b3f02 \
    095a726b9de16780d6ab9ccc7e55bf2876035ab2398e51df41566e4b13f9e018 \
  --sampling formal --timeout 900 \
  --out /mnt/709/data3/home/Dingzhong/l3_chain_runtime/a91a61e/formal
```

## 11. 证据位置与完整性

仓库内可提交镜像：

- [正式 manifest](evidence/l3_chain_runtime_20260924/final/manifest.json)
- [全部统计](evidence/l3_chain_runtime_20260924/final/summary.json)
- [30 个原始 record](evidence/l3_chain_runtime_20260924/final/records.json)
- [静态 topology](evidence/l3_chain_runtime_20260924/final/topology.json)
- [fixture manifest](evidence/l3_chain_runtime_20260924/final/fixtures.json)
- [最终完整性门](evidence/l3_chain_runtime_20260924/final/final_integrity.json)
- [145 个证据文件 SHA256](evidence/l3_chain_runtime_20260924/final/SHA256SUMS)
- [完整 driver 输出](evidence/l3_chain_runtime_20260924/formal_driver.log)
- `evidence/l3_chain_runtime_20260924/final/` 下保留每个进程的完整 stdout、JSON、运行前后 GPU snapshot 和定向 fixture。
- `evidence/l3_chain_runtime_20260924/build/` 下保留构建命令、日志、版本、资源用量、provenance 和状态；二进制与源码 tar 的哈希保留在 manifest/hashes 中。

未压缩的完整构建产物和源码 snapshot：

- `/mnt/709/data3/home/Dingzhong/l3_chain_runtime/a91a61e/build`
- `/mnt/709/data3/home/Dingzhong/l3_chain_runtime/a91a61e/formal`

## 12. 最终决策

runtime chain 在受测域内正确且实际命中，但没有稳定降低完整求解时间；matched B/C 在两张预选原始道路图上均失败，COL 未达到同图双卡目标，USA 的表面 A/C 优势也不能归因给 chain。重复整链展开、短链、有限覆盖和额外寄存器成本仍然存在，索引构建与显存又是纯新增负担。

因此：

1. 停止 `L3_CHAIN_PARTITION` 性能分支，不改默认配置，不合并到当前最佳方案。
2. 保留本提交中的正确性修补、工具和完整负结果证据，仅用于复核/研究档案。
3. 不继续 RGG、更多源点、八图性能或单卡同索引控制；这些只在 B/C 已显示正潜力时才合理。本轮归因边界明确为：已证明 chain treatment 的净效果为负，未声称 runtime-chain 收益独属于 L3。
