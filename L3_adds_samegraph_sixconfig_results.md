# L3 补充：ADDS(1 GPU) 同协议补跑与六配置对照（2026-09-22）

本轮任务：在**作业 37327 容量修复后冻结版本**的同一 A100 平台、同一冻结图/源点、同一计时边界与重复协议下，补跑 **ADDS(1 GPU, G)** 与 **ADDS(1 GPU, G+)**，与已有 MLMQ 四配置统一整理为六配置对照表和绘图数据。

## 1. 固定版本与证据

- MLMQ 四配置数据直接引用作业 37327 主表（`tmp/l3_supplement_20260921_v2/matrix/records.json`），未重跑；双卡二进制 `12c97d84…`、单卡 `07baa744…`、平台与协议见 `L3_supplementary_experiments_results.md` 第 2 节。
- ADDS 使用冻结官方 v4 二进制 `tmp/adds_official_v4/adds`，SHA256 `511e88f56762914829a055aea0215a6c8e3c0d1dfc1c078f5f2247a559638ec2`（harness `ADDS/official_bench.cu` = `3c0cd71d…`，与 v4 构建记录一致；nvcc 12.4.131 / sm_80，与 37327 相同工具链）。官方 kernel 与参数选择未改动。
- 图与源点使用 37327 导出并在运行前逐文件复核哈希的冻结件：`tmp/l3_supplement_20260921_v2/matrix/graphs.json`（原图 `landmark206_matrix/<G>/landmark.gr`，增强图 `tmp/l3_supplement_20260921_v2/graphs/<G>/augmented.gr`）。
- 平台：ada-A100，A100 80GB ×2（ADDS 为单卡进程，只用 GPU0）；本轮分配内检查 GPU 空闲。

## 2. 协议对齐与证据边界

| 项 | ADDS(1) 本轮 | MLMQ 37327 |
|---|---|---|
| 图/源点 | 冻结 graphs.json，逐哈希复核 | 同一冻结件 |
| 轮次 | 2 轮交错，第 2 轮反序 | 同 |
| 每进程 | 1 预热 + 5 正式 | 同 |
| 每格样本 | 2 进程 × 5 正式 = 10，取中位数，保留慢值 | 同 |
| 正确性 | 每查询逐顶点对照进程内 int64 Dijkstra 参考（official harness） | 逐顶点 WIDE_ORACLE |
| 计时边界 | `solve_ms` = driver_kernel 启动→cudaDeviceSynchronize；参数 profiling 在 `parameter_ms` 单列，不入 solve | `solve_ms` = 双卡就绪 host barrier→最后 GPU 同步 |
| 预处理 | 不入 solve（`BENCH_SETUP` 另行披露） | 不入 solve |

必须披露的边界：ADDS 的 solve 不含其 original parameter 选择（与 v3/v4 基线一致的单列 `parameter_ms`，实测 <0.1 ms 量级）；MLMQ solve 含其窗口/通信路径。两者同为 solve-only 口径，但阶段构成不同，对照时按各自边界解读，不把某一边的阶段悄悄并入另一边。

## 3. 执行结果

作业 **37646**（partition=a100，`--gres=gpu:a100:1`，8 CPU / 48G）：16 进程、96 查询（16 预热 + 80 正式），全部 `correct=1`、`RUN_RC=0`、审计 valid。入口 `scripts/multigpu/run_adds_samegraph_matrix.py`，日志 `tmp/l3_adds_sameprotocol_20260922/adds_matrix/`。

### 六配置对照表（ms，10 正式样本中位数）

| 图 | ADDS(1,G) | ADDS(1,G+) | M1-orig | M1-sc | M2-orig | M2-sc |
|---|---:|---:|---:|---:|---:|---:|
| NY | 10.6575 | 9.7094 | 6.4630 | 6.0098 | 7.1297 | 6.3752 |
| BAY | 13.1008 | 11.4271 | 5.1241 | 4.1287 | 5.7296 | 4.7419 |
| COL | 24.9613 | 18.4571 | 8.6291 | 5.7925 | 9.7735 | 6.9372 |
| FLA | 54.6454 | 44.9952 | 20.2768 | 17.3081 | 21.4211 | 17.6975 |
| CAL | 37.1897 | 22.9162 | 25.0761 | 17.1468 | 27.6907 | 18.0617 |
| E | 39.7378 | 30.3407 | 29.1492 | 23.4167 | 30.5892 | 24.2286 |
| W | 55.2349 | 38.6489 | 38.2061 | 29.1672 | 43.2860 | 30.0120 |
| USA | 136.3699 | 95.2021 | 96.6751 | 75.3454 | 112.1410 | 68.2066 |
| **几何均值** | **34.7369** | **26.3968** | **18.8059** | **14.7123** | **20.8096** | **15.4960** |

分轮中位数稳定性：ADDS 两轮差异普遍 <1%（如 USA-G 136.41/136.13、BAY-G 13.099/13.103），无单轮异常。

### 关键比值（逐图几何均值）

| 比值 | 含义 | GM |
|---|---|---:|
| ADDS(1,G) / M1-original | 官方单卡 vs MLMQ 无 L3 单卡，同原图 | **1.8471×** |
| ADDS(1,G+) / M1-shortcut | 官方单卡 vs MLMQ 无 L3 单卡，同增强图 | **1.7942×** |
| ADDS(1,G) / M2-shortcut | 官方单卡原图 vs MLMQ 双卡增强图 | **2.2417×** |
| ADDS(1,G+) / M2-shortcut | 同增强图单/双卡 | **1.7035×** |
| ADDS(1,G) / ADDS(1,G+) | ADDS 自身 shortcut 收益 | **1.3160×** |

乘法分解自洽：`ADDS_shortcut_gain × ADDS(1,G+)/M2-shortcut = 1.3160 × 1.7035 = 2.2417 = ADDS(1,G)/M2-shortcut`。

### 与旧三方数据的可比性说明

历史三方预览（`provenance_l3_road_bars_threeway.json`，adds_over_l3=2.2008、adds_over_nol3=1.7481）与本轮口径不同：旧批是同批次交错三进程但图视图约定不同（adds1 用原图、l3cut_2 用增强图），且 ADDS 协议为 2 warmup。本轮把 ADDS 补进 37327 的冻结图与 1 warmup 协议后，`ADDS(1,G)/M1-original=1.8471` 与旧 adds_over_nol3=1.748 同向且量级一致；`ADDS(1,G)/M2-shortcut=2.2417` 与旧 adds_over_l3=2.2008 一致。**两组各自成立，不能跨批拼接。**

## 4. 结论边界

1. 在同图同源同协议下，官方 ADDS 单卡在本八图上慢于 MLMQ 单卡（GM ≈ 1.79–1.85×）和 MLMQ 双卡（GM ≈ 1.70–2.24×）；该结论只覆盖这八个固定源点，不外推。
2. ADDS 自身 shortcut 收益 1.3160× 与 MLMQ 单卡 shortcut 收益 1.2782× 方向一致，说明图增强收益不依赖特定求解器，量级相近。
3. 计时边界差异（ADDS 的 parameter 阶段单列）按第 2 节披露；不能把六配置表解读为逐阶段成本分解。
4. ADDS 为单卡进程（gpu_count=1），未占用第二张卡；第二张卡在本轮分配中闲置，与 37327 的双卡分配（2 卡同跑一个查询）不同，属预期配置差异。

## 5. 复现入口与产物

| 内容 | 路径 |
|---|---|
| ADDS 运行脚本 | `scripts/multigpu/run_adds_samegraph_matrix.py` |
| sbatch | `tmp/l3_adds_sameprotocol_20260922/submit_adds_matrix.sh` |
| 原始日志/JSON | `tmp/l3_adds_sameprotocol_20260922/adds_matrix/`（16 组 `.log/.json` + `records.json`） |
| 六配置汇总 | `tmp/l3_adds_sameprotocol_20260922/six_config_summary.{json,csv}` |
| 绘图数据 | `tmp/l3_adds_sameprotocol_20260922/plot_data_six_config.json` |
| MLMQ 侧证据 | `tmp/l3_supplement_20260921_v2/matrix/`（作业 37327，未重跑） |
