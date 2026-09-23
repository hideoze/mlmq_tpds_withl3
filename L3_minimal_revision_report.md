# L3 最小收尾报告

> 本报告主体记录同步补丁版的核查与 job `38059` / `38075`；按用户要求回退补丁后的工作区复测结果见文末。

## 版本与范围

- 基础版本：`main` / `a50f719`（本轮从远端拉取后开始）；最终源码版本由本报告提交的 Git 历史标识。
- 同步回归候选源码、矩阵与 fixture 构建产物：`tmp/l3_minimal_revision_20260923_final_r4/`；论文配置构建单独保存在 `tmp/l3_minimal_revision_20260923_final/paper_config_build_current/`。均使用新目录，不覆盖旧 evidence。
- 论文配置：CUDA 12.4.131，`sm_80`，`MLMQ_WORKER_THREADS=512`；构建参数以 `tmp/l3_minimal_revision_20260923_final/paper_config_build_current/command.json` 为准，构建状态 `rc=0`，二进制 SHA256 `35af0a88b85bdbffd4b57a2ed772a1742b536902811a15b9900e6e5cdf48101a`。
- `TPDS.pdf` 和论文 `.tex` 源码不在当前仓库中；没有可安全修改的正文文件，本文记录应回填的最终短段落。

## 同步发布与观察关系

| 数据/交接 | 发布点 | 观察点 | 作用域与顺序 |
|---|---|---|---|
| L3 candidate → authoritative mark → collector | `remote_cand` 先做 atomic min，再对 `remote_mark` 执行 device-scope release `fetch_or`；见 [`l3_candidate.cuh`](SSSP/l3/l3_candidate.cuh) 和 [`l3_sync.cuh`](SSSP/l3/l3_sync.cuh)。候选回挂和 ghost event 使用同一发布 helper。 | cooperative/full-scan collector 对 mark 执行 acquire CAS 后才 exchange candidate；见 [`l3_collect.cuh`](SSSP/l3/l3_collect.cuh) 与 [`sssp_run.cu`](SSSP/sssp_run.cu)。 | 同一 GPU 使用 device scope。hint/pending/signal 仅作提示；权威 mark 和周期性全扫负责不丢候选。 |
| peer distance → owner dirty mark | peer distance 以 system-scope CAS atomic min 更新，随后对 owner `dirty_bitmap` 执行 system-scope release `fetch_or`；见 [`l3_sync.cuh`](SSSP/l3/l3_sync.cuh)、[`l3_bulk.cuh`](SSSP/l3/l3_bulk.cuh) 和 [`l3_owner_commit.cuh`](SSSP/l3/l3_owner_commit.cuh)。 | owner 对 dirty word 执行 system-scope acquire CAS claim，再处理对应顶点；fallback/backstop 也使用同一 dirty 原语，见 [`sssp_run.cu`](SSSP/sssp_run.cu)。 | 跨 GPU 的权威 bitmap 使用 system scope；只把发布/领取信号升级为 acquire/release，距离 CAS 保持 relaxed 原子更新。 |
| inbox payload/count/epoch/generation/READY | sender 通过 system-scope state CAS 取得 `WRITING`，写完 payload 后先做 system fence 与 warp 同步，再写 relaxed count/epoch，最后 system-scope release 发布 `READY`；generation 在 READY 发布前写入。见 [`l3_bulk.cuh`](SSSP/l3/l3_bulk.cuh) 与 [`l3_transport.cuh`](SSSP/l3/l3_transport.cuh)。 | receiver acquire 读取 READY，再读取 relaxed metadata；成功将 READY CAS 为 READING 后，每个 payload-reading lane 都 acquire 读取 READING，之后才读 payload。 | system scope 覆盖两张 GPU；generation 与 epoch 相等校验保留。 |
| RX → L2 queue commit → ACK/DONE/slot reuse | 默认 direct-RX 路径对 owner `node_data` 做 atomic min、记录获胜项并调用 `queue.write_through`；见 [`l3_receive.cuh`](SSSP/l3/l3_receive.cuh)。另一 BULK apply 路径在 node update 后以 system-scope release 发布 dirty mark；见 [`l3_bulk.cuh`](SSSP/l3/l3_bulk.cuh)。DQ producer lanes 写 payload、fence + warp sync 后 release 增加 `block_write_done`；manager acquire 该计数后 release 发布 `read_pos`，reader acquire `read_pos` 后读取 payload；见 [`cu_delta_queue.cuh`](core/cu_delta_queue/cu_delta_queue.cuh)。 | direct-RX 完成队列提交后才 system-scope release 发布 ACK、再 release 发布 DONE；sender acquire 观察状态及 ACK，并在 generation/ACK 条件满足后复用 slot；见 [`l3_transport.cuh`](SSSP/l3/l3_transport.cuh)。 | DQ 本卡交接使用 device scope；inbox 跨卡状态和复用使用 system scope。测试覆盖有界反压和延迟 ACK；测试不替代内存模型论证。 |
| idle 与终止控制 | 本卡 `local_idle` 以 system-scope release/exchange 写入；本地 `l3_term_req` 使用 device-scope release；本地 `manager_end` 使用 block-scope release；[`sssp_run.cu`](SSSP/sssp_run.cu)、[`l3_candidate.cuh`](SSSP/l3/l3_candidate.cuh)。 | peer idle 和 `l3_term_state` 使用 system-scope acquire；term request/ACK slots、`global_exit` 分别用 device-scope acquire，`manager_end` 用 block-scope acquire；最新 inbox ACK 以 system-scope acquire 检查。 | `local_idle` 只是活动提示，不能代替最终 peer state/ACK、完整 mark 检查和 L2 空队列确认。 |

本表依据 CUDA 12.4.1 对 legacy atomic scope/order 和 fence 的定义；本补丁明确了冻结论文配置实际经过的同步链，但不把定向测试数量表述成对所有编译分支和所有执行轨迹的形式化证明。[CUDA 12.4.1 Programming Guide：Atomic Functions](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#atomic-functions)、[Memory Fence Functions](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#memory-fence-functions)。

## 构建和 CUDA 回归

| 检查 | 当前状态 | 证据 |
|---|---|---|
| 冻结论文宏配置构建 | PASS，`rc=0` | `tmp/l3_minimal_revision_20260923_final/paper_config_build_current/{command.json,build.log,status.json,mlmq.sha256}` |
| 独立 no-L3 单卡 + 当前双卡源码准备 | PASS，`rc=0` | `single_build/status.json`、`dual_build/status.json`；二进制 SHA256 分别为 `5a8ef4f45a14237759c3e34519738d65ae159689fbfac9e11b374c9dff27f571` 和 `496ab4182997f8f938c145537225097497515403e4213df8536fd5b95cad0ff7`。单卡使用冻结 213 独立 worker；`core/` 与双卡版本同步。 |
| capacity / sequential-source 诊断构建 | PASS | `single_diagnostic_build/`、`dual_diagnostic_build/` 的 `status.json` 与完整构建日志 |
| 三个 GPU fixture 编译 | PASS | `test_primitives_build.log`、`test_dq_build.log`、`test_capacity_build.log` 和对应 SHA256 文件 |
| 八图 G/G+ paired matrix | PASS，Slurm job `38059`，ada-A100 2×A100 80GB / NVLink | `matrix/complete.json`：64 个配置进程、384 查询、320 正式样本，所有运行 `rc=0`、逐查询正确；driver 570.133.20。图、G+、oracle 哈希和 source/cut 与冻结归档 8/8 相同。二进制 SHA256：single `5a8ef4f45a14237759c3e34519738d65ae159689fbfac9e11b374c9dff27f571`，dual `496ab4182997f8f938c145537225097497515403e4213df8536fd5b95cad0ff7`。 |
| P2P、候选、DQ publication、容量和换源检查 | PASS，Slurm job `38075`，ada-A100 双卡 | `checks/results.json`：primitives、DQ publication、满容量、溢出、顺序换源全通过；32 个矩阵容量查询有效。P2P 双向 `access=1/native_atomics=1`；候选受控竞争通过；双向 transport 各 256 epochs、延迟 ACK 恢复、提交前检查 256 次均通过；DQ publication 400 cases 通过。 |
| 完整 audit | PASS（对上述 Slurm 原始输出执行 CPU-only audit） | `audit/audit.json`：320 formal、384 total、19 个共享 core 文件一致；最大 bucket 使用 2,177,477 / 16,776,704（13.0%），无计数回绕。首轮检查的 `WIDE_ORACLE` 行数断言从每个双卡查询一行修为每 GPU 一行；后续 audit 的 `L2_CAPACITY` 断言按每 GPU 两个队列日志修正。第一次失败输出仍保存在 `checks_reset_count_failed/`；修正后 Slurm checks 为 `valid=true`，audit 对同一原始记录输出 `AUDIT_PASS`。 |

GPU 作业内均先检查 `nvidia-smi` 与拓扑；两张卡均为空闲后才执行。matrix 为正式性能结果；容量和换源检查是单独运行，audit 不把它们合并为性能样本。定向测试支持代码所述同步合同，但不构成所有编译分支与执行轨迹的形式化证明。

## 性能数字与论文短段落

| 比值 | 最终结果 |
|---|---:|
| 同原图：M1-original / M2-original（八图几何平均） | **0.899×** |
| 同 G+：M1-shortcut / M2-shortcut（八图几何平均） | **0.921×** |
| USA 同 G+：M1-shortcut / M2-shortcut | **1.068×** |
| 同 G+：ADDS(1 GPU) / M2-shortcut | **1.643×** |
| shortcut 对双卡 solve time 的影响：M2-original / M2-shortcut（八图几何平均） | **1.302×** |
| shortcut 对单卡 solve time 的影响：M1-original / M1-shortcut（八图几何平均） | **1.270×** |

历史 ADDS 比较来自 job 37646、固定官方二进制 SHA256 `511e88f56762914829a055aea0215a6c8e3c0d1dfc1c078f5f2247a559638ec2`。本轮矩阵的原图、G+、oracle 哈希及 source/cut 均与归档 8/8 相同；矩阵实际主机也是 A100 80GB。旧批次同为 A100、两轮交错、每轮 1 次预热和 5 次正式测量，因此沿用其 ADDS G+ 数值并与本轮 M2-shortcut 重算。两边均是 solve-only，但 ADDS 的参数选择单独计时而 MLMQ 计入窗口/通信路径，正文保留此边界和 job 37646 来源。

建议替换正文的英文短段落（明确为 solve-only，不称端到端加速）：

> On identical shortcut-augmented inputs, the two-GPU configuration achieves a geometric-mean speedup of **1.643x** over single-GPU ADDS. Relative to one-GPU MLMQ, same-input speedups are **0.899x** on G and **0.921x** on G+; USA reaches **1.068x** on G+, the only G+ graph with a ratio above one. Shortcuts improve the two-GPU solve time by **1.302x** and the single-GPU solve time by **1.270x**. These solve-only results demonstrate a working two-GPU extension while showing that additional speedup remains workload dependent.

## 已知适用范围

正式性能构建使用冻结论文宏配置。`SEED_BARRIER`、`GLOBAL_ROUND_ASYNC_RX_FRONTIER`、`L3_RX_EXPRESS` 等未启用的可选协议分支没有纳入本轮运行时矩阵。其中 dormant `SEED_BARRIER` 路径仍有 gated plain P2P store 到 `peer_node_data`，其与 `seed_ready` 的发布/获取链没有纳入本轮同步结论；不能把本报告的保证外推到该分支。ghost event 标记已使用 device-scope release/claim，但默认 `GHOST_DEPTH=0`，因此也没有由本轮图回归触发。若要对这些可选协议作独立正确性声明，需要各自启用后补相应回归。

## 后续复测：移除同步补丁（2026-09-23）

按用户要求，本次将运行时源码文件还原到同步补丁提交 `567e47f` 的父版本，并移除 `SSSP/l3/l3_sync.cuh`。

| 项目 | 结果 |
|---|---|
| Slurm | job `38082`，`ada-A100`，`2×A100 80GB`，driver `570.133.20`，NVLink `NV12`，`COMPLETED` / `0:0`；运行前 `nvidia-smi` 未发现计算进程 |
| 配对矩阵 | `MATRIX_PASS`；64 个配置进程、384 条查询、320 个正式样本，全部 `rc=0` 且逐查询校验有效 |
| 输入一致性 | 与上方带补丁 job `38059` 的 8 张图之原图、G+、oracle 哈希全部相同 |
| 无补丁二进制 SHA256 | 单卡 `b9c365892b7208dd59feb68f6062feb3f662a730ec047c8a750e2280bae359f8`；双卡 `21a5b887152053fd2f5e52801d3f2f52dae7b5a01a69766dabf512b047fcbf91` |
| G+ 同图单卡/双卡几何平均 | `0.959×` |
| USA G+ M1-shortcut / M2-shortcut | `75.324 ms / 69.103 ms = 1.090×`；两轮比值分别约 `1.095×` 和 `1.059×` |

与带补丁 job `38059` 的 USA G+ `1.068×` 相比，无补丁复测为 `1.090×`；与更早 job `37327` 的 `1.105×` 接近。它们是不同 Slurm 批次，不能据此把全部时差归因于同步补丁。该矩阵验证的是回退版本的输出和性能，不验证其同步安全性。原始日志、manifest、二进制和逐图结果保存在 `tmp/l3_no_sync_rerun_20260923/`。
