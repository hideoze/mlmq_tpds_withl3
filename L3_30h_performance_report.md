# L3 30 小时性能执行报告

## 当前结论

正式执行已经收口。冻结 clean SHA
`9dd69fa02777853537e8068e90996b0ee7cc4186` 的 Job A 38304 与独立 Job B 38305
均以 `rc=0`、`status=COMPLETE` 完成，且源码、构建、输入和采样完整性门均通过。
原始正式证据根目录为
`/mnt/709/data3/home/Dingzhong/l3_30h_formal/9dd69fa02777853537e8068e90996b0ee7cc4186/`。
最终分类为：**execution PASS / measurement valid / target false /
`VALID_BELOW_TARGET`**。这表示实现与正式测量有效，但固定 solve-only 门槛
`S >= 1.20` 未达到；不得把执行通过写成性能目标通过。

| 正式主测 | 输入与源点（逻辑 ID -> 冻结布局 ID） | T1 median / IQR / MAD (ms) | T2 median / IQR / MAD (ms) | round 0 / round 1 `S` | combined `S` | 1.20 target |
|---|---|---:|---:|---:|---:|---|
| Job A 38304 | USA G+，`0 -> 11973673` | 73.3605045 / 0.5703575 / 0.3674960 | 64.6717355 / 1.0505103 / 0.6058680 | 1.1177602878 / 1.1374969373 | 1.1343518762 | false |
| 独立 Job B 38305 | USA G+，`0 -> 11973673` | 73.3405565 / 0.6623133 / 0.3687345 | 64.5799375 / 1.0865965 / 0.5707030 | 1.1392450291 / 1.1306786575 | 1.1356554270 | false |

两批正式主测均为 `sampling=formal`，每批 T1/T2 各 10 个、合计 20 个计时样本完整，
`formal_integrity_valid=true`。所有原样 solve/query-wall 样本、四个进程记录和逐轮
统计分别保存在 Job A 的 [records](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobA/01_primary/records.json)、
[samples](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobA/01_primary/samples.json)、
[summary](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobA/01_primary/summary.json)
及 Job B 的 [records](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobB/01_primary_confirmation/records.json)、
[samples](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobB/01_primary_confirmation/samples.json)、
[summary](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobB/01_primary_confirmation/summary.json)。
合并 query-wall 比值分别为 2.0904831244 与 2.1020434941，但只作为次级边界报告，
不能替代上表 solve-only 验收。

Job A 的终检全部有效：4/4 small fixtures、顺序换源/reset 和 8/8 图正确性检查均
通过，`problems=[]`。numeric checked-add 是正确性专用诊断，single/dual 共
10/10 项通过；该派生构建明确 `performance_claim_allowed=false`，其中计时不进入
性能结论。两个冻结附加 USA 源点的逐源结果为：

| 稳健性源点 | 逻辑 ID -> 冻结布局 ID | T1 median (ms) | T2 median (ms) | round 0 / round 1 `S` | combined `S` |
|---|---|---:|---:|---:|---:|
| additional 1 | `7982448 -> 18266241` | 73.3442000 | 99.8804085 | 0.7077949713 / 0.7424814402 | 0.7343201845 |
| additional 2 | `15964897 -> 6146689` | 60.3638890 | 55.3409575 | 1.0915499763 / 1.0904844026 | 1.0907633645 |

两项测量均有效但未达目标，原样逐轮数组位于 additional 1 的
[summary.json](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobA/04_usa_sources/sources/fixed_additional_source_1/summary.json)
和 additional 2 的
[summary.json](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobA/04_usa_sources/sources/fixed_additional_source_2/summary.json)。

八图逐图/逐源 solve-only 结果如下；G 与 G+ 的 source 均是同一冻结布局 ID：

| 图 | source | G T1 / T2 (ms) | G `S` | G+ T1 / T2 (ms) | G+ `S` |
|---|---:|---:|---:|---:|---:|
| NY | 132173 | 6.4820350 / 7.2509915 | 0.8939515375 | 6.0012980 / 6.5845435 | 0.9114220295 |
| BAY | 160635 | 5.0066155 / 5.7178005 | 0.8756191301 | 4.1111360 / 4.8006820 | 0.8563649915 |
| COL | 217833 | 8.4261215 / 10.0421045 | 0.8390792488 | 5.8036630 / 7.0449735 | 0.8238019632 |
| FLA | 0 | 20.6379750 / 21.9721555 | 0.9392785792 | 17.3261735 / 18.1160940 | 0.9563967542 |
| CAL | 945407 | 24.3467280 / 26.9236450 | 0.9042879595 | 16.9820485 / 18.3103230 | 0.9274576150 |
| E | 1799311 | 28.3792305 / 33.3588815 | 0.8507248812 | 22.7811780 / 24.7179085 | 0.9216466676 |
| W | 3131052 | 38.1140805 / 44.9916625 | 0.8471365222 | 29.4475200 / 30.0333170 | 0.9804950948 |
| USA | 11973673 | 94.5793005 / 98.1853925 | 0.9632726222 | 74.4119290 / 68.6201200 | 1.0844039474 |

八图 G/G+ 回归 16/16 case 有效，共 64 个进程、384 个查询（含预热）、
320 个计时查询；G 的 solve 几何平均为 `0.8881843984`，G+ 为
`0.9298583563`，16 项中达到 1.20 的数量为 0。该结果与主测共同说明：当前 L3
实现已通过执行、正确性和证据完整性验收，但在冻结工作负载上属于有效低于目标，
而不是性能验收通过。

附加源点与八图由正式 Job A 外层在同一 clean SHA 下分别 fresh-build 并冻结
independent-single/dual pair，前后核验 worktree、pair 与输入完整性；每个 case 均
执行两轮反序、每进程 1 次预热 + 5 次计时。内层通用 `run_l3_30h.py` 为复用外层
刚构建的冻结 pair，按接口原样记录 `sampling="exploratory"`。因此这些结果准确
称为“外层 formal-equivalent gates 下的 full-sample clean-SHA 扩展回归”，不把
内层 runner mode 重标为 formal，也不用于 primary target selection。逐 case 原样
数组和 source 见 [summary.json](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobA/05_eight_graph/summary.json)
与 [case_index.json](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobA/05_eight_graph/case_index.json)。

前五次 clean-SHA Job A 和 dirty 探针继续作为隔离的失败/探索证据保留；它们没有
被用于补齐 Job 38304/38305 的任何样本，也没有与最终批次拼接。当前仓库不含论文
`.tex` 或可编译论文工程，因此论文正文插入与 TeX 编译状态为 **NOT_RUN
(manuscript source absent)**；本报告只给出可回填结论。

## 历史过程与因果定位（保留）

截至 Job 38292，阶段 0、阶段 1、Route A、Route B1 与 Route B2 已完成并进入
正式验收，但当时前五次 clean-SHA Job A 均未形成完整可接受批次。前两次用于定位
源码归档门和 BNUM=16 容量问题，后续批次证明 BNUM=8 主测和终检可完整运行，并
继续暴露了三项互相独立的正式编排问题：终检工具路径、fixture 源码物化和 numeric
诊断的阻塞 launch 环境。上述状态已由 Job 38304/38305 的完整通过取代，但保留于此
作为失败账本和因果证据。
Route C 因作业 38122 的固定搜索范围内未找到可用 NVSHMEM 安装而按止损条件停止；
这不是机器范围的“绝对未安装”声明。compact-candidate 只有小幅探索信号且仍低于
目标，不进入当前重冻结候选；该候选仍是 USA、delta=400000、blocks=107、
W512、固定 25000-cycle 的 control 配置，但 L2 几何改为 BNUM=8。

`c0bc9ca` / Job 38227 的 dual/single pair 编译成功，但在任何 GPU 采样前，
source provenance gate 将 tar 中 Git 不跟踪的目录权限和普通文件非执行权限差异
误判为源码差异，工作流停止。比较边界修正为 Git 的路径、类型、内容及可执行位
语义后，`7cd563c` / Job 38238 通过了源码与构建来源门。Job 38238 的 round 0
完成 single/dual 各 1 次预热和 5 次正式正确样本；round 1 的 dual 进程完成预热
后，第二个查询在两张 GPU 上均出现 `CUDA error 719: unspecified launch failure`，
进程 `rc=2`。因此整批 `measurement_valid=false`、`target_met=null`；round 0 的
`73.053410 / 65.569749 = 1.114133` 只是失败批次诊断值，不能作为正式性能结论。

`ead2cdb` / Job 38271 的 BNUM=8 `01_primary` 随后完整通过：single 与 dual 各取得
10 个正式正确样本，两个反序 round 均有效，容量/守恒/no-wrap 门通过。合并 solve
中位数为 `73.9061515 / 64.3800475 = 1.1479667`，所以 solve 口径的 1.20 目标
未达到；query-wall 中位数为 `154.050660 / 72.8607125 = 2.114317`，但它只是次级
口径，不能替代 solve 验收。Job A 随后在 `02_final_checks` 执行任何终检前失败：
pair 已权威记录 `/usr/bin/nvcc`，编排却仍传入计算节点不可解析的
`/usr/local/cuda/bin/nvcc`。因此该目录作为失败工作流证据保留；`03_numeric_add`、
`04_usa_sources`、`05_eight_graph` 和 Job B 均为 `NOT_RUN`，不得用该批的主测结果
与后续批次拼接成完整验收。

`a539b8a` / Job 38280 删除过时 nvcc 路径后再次完整通过 `01_primary`：合并 solve
中位数为 `73.7722515 / 64.3301410 = 1.1467758`，两轮分别为 `1.1503776` 和
`1.1684188`，仍是“有效但未达 1.20”。`02_final_checks` 已成功按 pair 记录的
`/usr/bin/nvcc` 重建归档 dual 二进制，随后在执行任何 fixture 前编译失败：fixture
源文件位于工作树并通过相对路径包含工作树 `SSSP`，同时命令显式包含归档
`core/include`，使同一 `common.h` 以两个路径进入编译单元并产生重复定义。该失败
是终检源码物化错误，不是 GPU 算法失败。`03_numeric_add`、`04_usa_sources`、
`05_eight_graph` 和 Job B 再次为 `NOT_RUN`，Job 38280 目录原样保留。

`feb2d92` / Job 38292 的 `01_primary` 同样有效，合并 solve 中位数为
`73.1693135 / 65.8522925 = 1.1111126`，两轮分别为 `1.1147943` 和
`1.1003592`，目标未达。该批首次完整通过 `02_final_checks`：4/4 small fixtures、
顺序换源/reset 和 8/8 图正确性全部有效，`problems=[]`。`03_numeric_add` 的
single/dual 派生二进制也成功构建，但 runner 强制 `CUDA_LAUNCH_BLOCKING=1`，
使依赖多 stream 协作的持久 kernel 在第一个 `atmosmodm_single` 上等待后续尚未
启动的 kernel。该运行 600 秒超时、`rc=124`、无 BENCH 样本；runner 随后进入
dual 时 Job 38292 被定向取消以释放两张 A100，因此 orchestration.status 保留在
`RUNNING/03_numeric_add`，不得解释为完成。移除该环境变量后，同一派生 single
二进制在一张 A100 上以 `rc=0`、oracle correct 和 solve `7.100645 ms` 完成，形成
直接因果复核；此定向复核不是性能证据。`04_usa_sources`、`05_eight_graph` 和
Job B 为 `NOT_RUN`。

同一取消作业后的第二项定向复核进一步确认 dual 诊断构建合同：清除
`CUDA_LAUNCH_BLOCKING` 后，原始派生 dual 二进制在 W512 启动时报 CUDA 701
（kernel 请求资源过多）；只为该正确性诊断派生构建增加
`--ptxas-options=-maxrregcount=96` 后，相同 W512、BNUM=8 和算法宏在两张 A100 上
以 `rc=0` 完成，wide oracle correct、final audit `mismatches=0/residual=0`，且
no-wrap conservation 全部通过，solve 为 `7.957710 ms`。这两个定向运行都不是
正式性能证据；寄存器上限不得加入正式 single/dual pair，也不得用其计时形成性能
结论。

静态源码与正式二进制 SASS 证据高置信指向逐桶 DQ no-wrap guard：累计
`write_reserve` 越过物理逐桶容量时会执行 device `trap`。队列地址虽取模，但
当前协议没有证明安全复用所需的连续 retire frontier 或 generation。失败进程的
预热查询中，GPU1 单桶写入已达到 `9,105,045 / 16,776,704`；后续查询可能因
工作量重尾波动越过上限。CUDA 719 与该活跃显式 trap 一致，但因没有取得 trap
PC，此处只称“高置信定位”，不称指令级最终证实。

30 小时 GPU 窗口从 Slurm 首次实际授予两张 A100 的
`2026-09-23 15:43:29 +08:00` 起算。下文 Stage 1 性能数均是
dirty-worktree、`07df3ad` 回退同步运行时上的探索样本（每进程 1 次预热 + 3 次
正式）；它们可以用于淘汰或选候选，不能替代恢复同步后的 clean-SHA 两轮反序
正式结果。Job 38227、38238、38271、38280 和 38292 的失败/取消目录均原样保留，
后续不得覆盖。

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
| 同步/容量实现收口 | 已完成 | BNUM=16 正式运行暴露高置信 no-wrap 容量 trap；保留 fail-stop，并将 dual/single 同步改为 BNUM=8；最终 Job A/B 正确且无回绕 |
| clean-SHA 定向验证 | 已完成 | 旧 Job 38227--38292 分别暴露并隔离归档、容量、工具路径、混合源码树和诊断 launch 问题；`9dd69fa...` 的 Job 38304 全链通过 |
| 正式验收 | 执行 PASS；有效但低于目标 | Job A 38304 与独立 Job B 38305 均 COMPLETE、measurement valid；solve 分别 `1.1343518762x` 与 `1.1356554270x`，所以 `target_met=false`、`VALID_BELOW_TARGET` |
| 数值与扩展回归 | 已完成 | numeric 10/10 正确性诊断通过；附加 USA 两源点有效；八图 G/G+ 16/16 valid，320 个计时查询，0/16 达到 1.20；内层 exploratory、外层 formal-equivalent gates |
| 论文正文插入/TeX 编译 | NOT_RUN | 当前仓库缺少论文 `.tex` 与可编译论文工程；报告提供回填文本，但不虚构编译或版面证据 |

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
最终 USA G+ 另以 `usa_augmented_dataset_manifest.json` 全量核对。当时
`atmosmodm`、`rmat22` 与 USA 三个冻结源点的每次活跃候选加法仍待最终 clean pair
的 checked-add 派生构建关闭；Job 38304 已完成 10/10 正确性诊断并将该冻结查询集
的 numeric gate 置为 PASS。该诊断禁止用于性能主张。

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

## 第一次冻结失败、容量定位与 BNUM=8 重冻结

第一次 clean-SHA 候选恢复提交 `567e47f` 的完整同步修复，并只叠加本轮固定窗口参数化、有效
配置日志和查询后容量门禁。默认 BULK 活跃路径的 candidate/mark、payload/READY、
RX/L2/ACK、DQ publication、retained/requeue、worker recovery 与终止链已经逐项
静态审计；字段、发布点、观察点和作用域记录在
`evidence/l3_30h_20260923/final/sync_contract.md`。结论不外推到未启用的
`SEED_BARRIER`、tile-loan、owner-commit 或其他实验分支。

正式构建对每个逐桶权威 DQ `write_reserve` RMW 启用
`DQ_COUNTER_OVERFLOW_GUARD`。其当前语义是“累计写预留不得发生物理环绕”，
不只是 32 位整数 overflow：旧值与 int64 新值必须不超过逐桶 `total_size`，否则
置 sticky flag 并 fail-stop。由于当前队列没有连续 retire frontier/generation，
不能把门槛简单放宽到 `INT_MAX` 后宣称 modulo 槽位可安全复用。后续
read/completion 计数仍由唯一成功预留约束，并由终态精确守恒交叉验证；independent
single 使用相同共享-core guard。

当前重冻结候选将 dual 和 independent single 同时固定为 `BNUM=8`、
`BUCKET_MAX=4`、`l2_batch_size=8`。2,147,483,647-byte 预算对应 268,435,455
条八字节记录；每桶容量由 BNUM=16 的 16,776,704 增至 33,553,920，总可用桶区
为 268,431,360，剩余 4,095 条记录不属于任何桶。该改动不是纯容量扩大：manager
warp 数从 16 降到 8，共享元数据、可见 delta 桶跨度、远距离项 clamp 和工作顺序
均改变。因此它是新算法配置，旧 BNUM=16 时间不能复用。

dirty 预冻结探针 Job 38254 运行 3 个全新双卡进程、每进程 1 次预热 + 5 次记录
查询，共 18 次查询全部 `rc=0`、`WIDE_ORACLE correct=1`、L2 读写/完成/guarded
write 守恒，且 `overflow_detected=0`、`no_wrap=1`；观测最大单桶写入为
`3,316,286 / 33,553,920`（9.88%）。此前 Job 38251 及随后一次未取得 job ID
的直接重试均因计算节点不可见登录节点 `/tmp` 中的二进制而以 `rc=127` 结束，
算法没有执行，也不计入上述 18 次查询。dirty dual/single pair 均实际编入三项
几何宏并编译成功；57/57 CPU 合同测试通过。这些
仅是重冻结工程门，不是 clean-SHA 验收或正式性能证据。Job 38254 的紧凑摘要、
日志哈希和原始保留路径记录在
`evidence/l3_30h_20260923/final/bnum8_exploratory_probe.json`。

`mlmq_benchmark::finish` 在计时终点后作为两卡 host barrier；两卡都完成
`cudaDeviceSynchronize` 后才读取 `L2_FINAL`。最终门强制逐桶 drain、guarded
writes/总读写/完成守恒、精确 `INT_MAX`-byte budget BNUM=8 容量元组、32 位配置、
`overflow_detected=0` 和 `no_wrap=1`。canonical compile argv 另强制 BNUM、
BUCKET_MAX 和 batch，正式 input contract 同时冻结这些字段。并发发布测试仍不能
由终态守恒替代。

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

## 正式收口与文档状态

原“下一步”中的 numeric launch 合同修正、完整 Job A、独立 Job B、附加源点与八图
回归均已在 clean SHA `9dd69fa02777853537e8068e90996b0ee7cc4186` 完成。最终
证据只取 Job 38304/38305；Job 38227--38292 和 dirty 探针继续位于独立目录，作为
失败与因果账本，不参与正式样本计数。

正式证据已归档到
[`evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/`](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/)，
其中包含独立复算的 [`formal_summary.json`](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/formal_summary.json)、
[`completion_matrix.json`](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/completion_matrix.json)、
[`SHA256SUMS`](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/SHA256SUMS)、
单栏 [`PDF`](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/figures/l3_30h_final.pdf) /
[`PNG`](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/figures/l3_30h_final.png)
以及 [`paper_update.md`](evidence/l3_30h_20260923/formal/9dd69fa02777853537e8068e90996b0ee7cc4186/paper_update.md)。
当前仓库缺少论文源码，因此正文插入、TeX 编译和版面检查保持
`NOT_RUN (manuscript source absent)`；若后续提供论文工程，应使用上述冻结图与
`VALID_BELOW_TARGET` 段落完成这三项，不能把尚未执行的文档检查写成 PASS。
