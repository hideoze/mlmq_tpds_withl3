# L3 补充实验清单：源码核查、执行设计与论文使用边界

核查日期：2026-09-21。对应原文：[L3_supplementary_experiments.md](L3_supplementary_experiments.md)。

本文回答“这份补充实验清单如何依据当前实现落到可验证的实验，以及哪些内容可以用于论文”。本轮完成源码核查和已有日志的离线整理，不把建议实验写成已完成结果。未启动新的 GPU 作业；原清单、求解器和上一份答复保持原样。

## 0. 总体判断：保留范围，但需要六项具体修正

原清单把工作限制在双 GPU 扩展、优先同图归因与关键正确性，范围合理。根据源码，应作以下修正后再执行：

1. **冻结增强图即可利用现有 `.gr` 读取能力，不必先发明新图格式；关键是关闭再次增强，保持邻接顺序，并对照原图参考距离。** 直接让当前开启 shortcut 的双卡程序读取 `G+`，会再次运行构造，可能得到另一个输入。
2. **现有 `-s` 多源入口不能用于 L3 跨查询 reset 验证。** 它按源点把独立查询分派给两卡，每卡拥有完整图，逻辑 `n_gpu=1`；当前 shortcut 构建甚至直接拒绝多源入口。
3. **RX 和本卡终止检查由同一个 manager warp 执行。** “延迟 RX 提交，同时让同一 manager 发起终止”不是当前实现能发生的并发窗口，测试设计必须遵守真实角色顺序。
4. **现有 `BENCH correct=1` 使用的 MLMQ CPU reference 也是 32 位距离。** 数值边界测试需要另外的 64 位或更宽参考；不能将已有标志直接解释为宽整数核验。
5. **诊断字段名称不能直接作为论文指标定义。** `candidate_updates` 是 peer-cache 原子胜出数；candidate 真正严格改进数对应 `signal`。`tx_batches` 在发布之前增加，`waves` 只保留前 32 批，现有计数也不等于完整的自然重试统计。
6. **已有日志足以先补成本披露，但 setup 与每次 query 必须分开。** 本轮已经整理成可复算的表，不需要为了这一步新增 GPU 运行。

### 核查版本和证据层级

分支 `para-frame3-multisource`，HEAD `b842b8a`。工作区有既有未提交改动，因此以归档文件和二进制哈希补充定位。

- 当前性能版本：`L3-ROAD-CHAIN-WIN25-W512-CUT-20260916`。
- `tmp/base_today/l3road_w512_frozen/source.tgz` 的 74 个源码成员，本轮再次核对均与工作区对应文件相同。
- `tmp/base_today/prod_nogate512/mlmq` SHA256：`9731384959c0e39b8f860b5f37677f8b03272cb31fffe109f8923d4c1d2b43ac`。
- `tmp/base_today/nol3/mlmq` SHA256：`2eca02408fa819b061063f106bfa781c3a0e34a74555ec6e51f3a800bb344d5b`。
- 原补充实验清单 SHA256：`837c50e09b7448119ed580de2004409e9a527162b710be18ef37b74e04b05aef`。

文中“源码已支持”不等于“该实验已运行”；“已有日志验证”不等于“本轮新 GPU 复验”；“建议增加的钩子/字段”也不是现有接口。后面分别标明。

## 1. 四配置对照：怎样真正固定图表示和本地实现

### 1.1 这四组是必要且足够紧凑的主要性能实验

令 `G` 为现有 landmark 布局、尚未增加 shortcut 的图；固定双卡 owner 切分 `π` 和构造源点 `s0`：

\[
G^+=F(G,\pi,s_0).
\]

| 配置 | 设备数 | 实际 CSR | host 构造行为 |
|---|---:|---|---|
| M1-original | 1 | `G` | 不增加 shortcut |
| M1-shortcut | 1 | 冻结的 `G+` | 不重新按单卡归属构造 |
| M2-original | 2 | `G` | 关闭 chain augmentation，保留同一 L3 协议 |
| M2-shortcut | 2 | 与 M1-shortcut 一致的 `G+` | 不再次增强，使用固定 `π` |

当前入口的行为见 `SSSP/main.cu:469`：开启 `L3_CHAIN_SHORTCUTS` 后，`n_gpu>1` 会执行构造；`:590` 再用增强图分区。仅改变 `-n 1/2` 不能直接得到完整四组。

**M2-original 应保留 boundary index、窗口、direct RX、retained TX、终止恢复及 W512，只关闭图增强。** 不能拿早期“无 shortcut、无 boundary index、另一套窗口”的旧双卡版本充当它，否则改变了多个因素。

### 1.2 推荐的最小实现路线：host 导出一次，现有 loader 读取

现有 `SSSP/csr_graph.cu:185` 的 `readFromGR()` 已读取 row ends、destinations 和整数权重；`:302` 也有 `writeToGR()`。因此原清单的“增加读取冻结 CSR 入口”可以收紧成：

1. 使用当前 `l3_chain_shortcut_view::build`，按已选双卡 cut 和 source 生成 `G+`。
2. 以现有 `.gr` 格式导出一次；重新读回，逐数组核对 row offsets、目标、权重。
3. 单卡与双卡都读取这个冻结文件；两个消费者均跳过再次构造。
4. 将原 `G` 的独立宽整数参考结果用于两者的正确性检查。

导出工具尚未在本轮新增或执行。可以复用现有构造器和序列化逻辑，但必须验证文件边界、奇数边数时的 padding，以及读回后的数组一致性，不能仅凭存在 `writeToGR` 函数就宣布导出路径已通过。

`L3_CHAIN_SHORTCUTS` 的当前主要接线位于主机入口和配置约束；普通松弛内核通过输入 CSR 使用新增边。因此关闭自动构造、读取事先增强的 CSR 是与现有结构相容的实验路径。仍应比较实际编译参数和目标内核，不能未经检查宣称两个构建的 GPU 指令完全相同。

另一条可行路线是为 host 测试适配器添加“输入已经增强”的模式。这应是明确的新实验接口，本文不提供不存在的命令行开关。两条路线选一条即可，避免同时维护两套语义。

### 1.3 “同一图”必须包含邻接顺序

当前构造器把 shortcut 放在对应 CSR 行的原边之前，然后保留原边顺序：`SSSP/l3/l3_chain_shortcuts.h:99`、`:102`。邻接顺序会影响异步任务生成的先后，因此不能只核查边集合相同。

冻结清单至少包含：

```text
原 G 文件哈希、G+ 文件哈希
row offsets / destinations / weights 的规范化内容哈希
顶点编号映射哈希、n、m、m+、shortcut 数
构造器源码哈希、strict/loose 模式、构造源点 s0
双卡 cut 的整数顶点位置、两侧 owner 区间
实际查询源点 s、距离类型和 INF 定义
```

如 padding 或尾部保留字节未规范化，文件哈希可能不同而有效 CSR 相同；同时记录内容哈希可以解释此差异。但正式冻结后，消费者优先共享同一份文件，避免歧义。

只比较 eligible/shortcuts/longest_path 等统计不够。历史 `run_chain_no_l3_ablation.py:71` 主要比较构造统计，不能据此替代当前四组所需的逐数组一致性。

### 1.4 正确性参考必须来自 G，不能只来自 G+

如果 M1-shortcut 和 M2-shortcut 都以错误构造的 `G+` 为输入，而 CPU 也仅在 `G+` 上求解，CPU/GPU 一致仍可能掩盖图变换错误。

建议对每个固定源点先计算 `Dijkstra64(G,s)`，再检查：

\[
D_{1,G}=D_{1,G^+}=D_{2,G}=D_{2,G^+}=Dijkstra64(G,s).
\]

小图可同时检查增强图的所有源点；道路图保留当前固定源点即可满足主对照范围。各数组保留独立的不可达标记，不把无法表示的有限距离转换为 `INT_MAX` 后当成不可达。

### 1.5 固定本地实现不是要求单卡运行多余的 L3 开销

应对齐的是 L0–L2 算法、有效队列参数、距离类型、每卡工作几何和共同的正确性修复；单卡可以继续使用独立无 L3 实现。不能为了“同源码”直接以多卡程序 `-n 1` 替换现有无 L3 基线，却不说明其中残留的管理开销。

推荐每个设备数量对应一套明确的 solver 构建，每套构建分别运行 `G` 与 `G+`；对不同设备数量的实现差异做限定范围的源码和参数核对。若单卡与双卡的本地队列版本或修复不同，先统一，或明确降格为系统配置对比。二进制哈希本身不能证明本地实现相同。

当前已确认的主要参数：

| 参数 | 应固定的值/语义 |
|---|---|
| 本地队列 | `L1SLF_L2DQ` |
| delta | 200000；当前有效路径通过 `-d` 设置 `g_delta_override` |
| work blocks | 每卡 107 |
| 工作线程 | `MLMQ_WORKER_THREADS=512`，16 个 work warps；另加本地管理 warp，launch 为 544 threads/block |
| 典型 DQ 配置 | 16 桶、4 个并发扫描桶、batch=8；归档最终 setup 值 |
| shortcut | strict 模式；固定构造 source/cut；消费者不再次增强 |
| L3 | 固定当前 direct RX、retained TX、boundary、window、ACK 和 recovery 设置 |
| 诊断 | 正式计时关闭新交错钩子和热路径诊断；不把故障版耗时纳入主表 |

`sssp_run_adaptive()` 当前使用 `setup.init_setup()`，`init_setup_adaptive(info)` 被注释；覆盖队列类型和 delta 的位置见 `SSSP/sssp_run.cu:12018`。因此应打印最终 setup，而不只记图特征或函数名称。

### 1.6 固定源点和分区的具体表

| 图 | source（当前重排后 0-based） | GPU0 比例 | cut 顶点 |
|---|---:|---:|---:|
| NY | 132173 | 60% | 158607 |
| BAY | 160635 | 50% | 160635 |
| COL | 217833 | 55% | 239616 |
| FLA | 0 | 50% | 535188 |
| CAL | 945407 | 50% | 945407 |
| E | 1799311 | 50% | 1799311 |
| W | 3131052 | 40% | 2504841 |
| USA | 11973673 | 60% | 14368408 |

source 来自 `tmp/landmark206_matrix/<graph>/layout.json`；cut 从当前同批日志核对。M2-original 和 M2-shortcut 使用相同 cut；M1-shortcut 复用由这个 cut 构造的 `G+`。这些比例是已经选定的逐图配置，不应写成自动分区算法。

### 1.7 最小正式批次与统计

保留原清单的两轮反向交错，每个进程 1 预热+5 正式：

- 8 图 × 4 配置 × 2 轮 = **64 个进程记录**；
- **320 个正式查询**，加 64 个预热，共 **384 个查询**；
- 其中真双卡查询 192 个，正式双卡查询 160 个。

第一轮按固定四配置顺序，第二轮同时反转图顺序与配置顺序。两个 GPU 在同一 Slurm 分配内独占用于该批实验；单卡配置求解时不让另一张卡同时执行其他测量。提交时保留显式 `--mem`，设备检查和实验都在目标节点完成。

每次运行从清理过的 `MLMQ_*`/`BENCH_*` 环境开始，再显式设置所有值，尤其为两种 M2 配置都设置 cut。当前三方脚本从 `os.environ` 复制环境，只对特定图覆盖 cut；不能原样扩为四配置后依赖继承环境。已有 `usa_recovery_confirmation_runner.py:75` 的清理方式可以参考，但清理后必须补回本次 cut。

每图每配置合并 10 正式样本取 median；离散度推荐 **IQR，固定 inclusive 线性插值定义**，并保留原始样本。它是离散度，不是 95% 置信区间。反向交错提供批次控制，但同一进程的五次查询不是五个独立的硬件会话，不宜据此夸大统计显著性。

使用未舍入的中位数计算原清单四个比值和逐图几何平均。除原式外，可在正文需要时补 `T(2,G)/T(2,G+)`，表示固定双卡条件下的图增强收益；不必另加一幅图。

失败、超时、缺失或错误样本不能删除后以剩余样本补齐十次。将该单元格标记为未完成/失败，保存原批；修复后冻结新版本并采集新的完整对照。实验成立的标准是公平和可复核，不是八图全加速或全部超过 1.1×。

## 2. 定向正确性实验：五类用例应怎样接到真实执行路径

### 2.1 先区分已有测试覆盖

| 已有材料 | 实际覆盖 | 不能据此声称的内容 |
|---|---|---|
| `test_chain_shortcuts.cpp` | strict host 构造及 64 位参考；上一轮实际 213 fixtures PASS | 新的 GPU 数值/并发检查已完成 |
| `test_l3_collect.cu` | 预填 candidate/mark 后收集，检查容量边界和剩余状态 | 多个 producer 与 TX 的受控交错已覆盖 |
| `test_l3_reset.cu` | 两设备上 typed fill 的 guard、长度、尾部和重复填充 | 完整 L3 查询的 cache、ACK、token、队列 reset 已覆盖 |
| `test_l3_activation.cu` | `experimental/l3_activation_215.cuh` 的激活合并原型 | 当前普通 direct RX 的端到端行为已验证 |
| fault512 归档 | 有限 publish/claim/ready/ACK 扰动和隐藏通知，记录正常终止 | 真实占满两槽、所有迟到交错、当前逐图 cut 都已覆盖 |
| 当前三方性能日志 | 固定源点重复，真实双卡，已有结果正确标记 | 同进程换源且继续使用双卡 L3 已覆盖 |

本轮没有重新执行这些 GPU 测试，也没有重复运行上一轮的 213-fixture host 测试；这里是源码与已有材料的覆盖核查。

特别是 fault512 的 NY/W/USA 日志采用**50% cut**：NY=132173、W=3131052、USA=11973673，而正式性能配置分别使用 60%、40%、60%。该历史证据仍有价值，但不能称为当前所有分区配置的故障复验。

### 2.2 用例 E1：同一目标连续改进与 candidate 领取交错

**钩子位置建议（尚未实现）：** `l3_record_candidate` 的值更新/mark 发布，及 TX 清 mark 成功与 `atomicExch(remote_cand, INF)` 之间。当前全扫分支位于 `SSSP/sssp_run.cu:9449`、`:9457`，协作收集在 `SSSP/l3/l3_collect.cuh`。

为同一目标构造有效候选 10→4→2，至少分别触发：

1. 更小值在 TX 清 mark 之前产生；
2. 更小值在清 mark 后、取 candidate 前产生；
3. 更小值在 candidate 取走后产生；
4. 同一个 mark word 的另一个 bit 变化，使 CAS 竞争或剩余 bit 需要保留；
5. peer cache 已胜出而 candidate 发布尚未完成，验证 worker 在途责任不能被提前确认。

每种时序独立重置状态。最终检查原图的宽整数参考距离，并记录被领取的值、后续 candidate/mark、journal、发布 epoch 和 owner 收到的最小值。**允许合并和冗余发送，不要求 10、4、2 各发送一次，也不要求严格 exactly-once。** 要求的是最优必要值最终被覆盖。

普通候选数组位于发送 GPU；其 producer/TX 竞争本身主要发生在同一设备。但最终集成用例必须将对应结果实际经 P2P 送到另一个 owner，不能仅让两张卡各自执行同一个本地单元测试。

测试图应有无法被其他边绕过的关键远端目标，否则最终距离正确可能来自另一条传播路径。受控钩子需记录确实进入目标窗口；未触发时标 `NOT_TRIGGERED`，不能计 PASS。

受控钩子的额外同步也可能掩盖原实现的弱内存排序问题。因此它验证的是指定交错下的责任交接，仍需与实际操作的内存模型论证和无钩子运行证据分别报告。

### 2.3 用例 E2：两槽占满、retained journal 和延迟 ACK

当前每方向有两个槽。只制造一次 `publish=false` 不能证明“两槽都占用时后续 journal 被保留”。建议按明确事件次序触发：

```text
产生批次 e1 -> 发布槽 1，暂不释放
产生批次 e2 -> 发布槽 0，暂不释放
产生第三批 e3 -> 两槽不可复用 -> retained_count > 0
保持一段有界等待 -> 核查 journal 内容和槽位 generation 未改变
释放接收/ACK -> e3 发布 -> owner 提交 -> 最终终止
```

必须分波产生候选；一次产生很多同目标更新可能被合并成一批，不能保证触发三个 epoch。建议在很小的真实双卡 fixture 上用事件钩子控制生成波次。

另设 ACK-only 子用例：让接收已经提交 L2 并进入 DONE，但暂时没有 ACK，确认仅看到 DONE 不足以允许覆盖旧 generation。现有 `L3_FAULT_INJECT_ACK_DELAY` 提供了可参考的有限延迟实现。

核查点：两槽的 state/generation/epoch/ACK、第三批的 retained 状态和内容、等待期间新增候选是否继续保留、释放后是否逐批推进。原始状态不得被测试钩子伪造为 DONE 以绕过协议。

保留生产的两槽配置即可完成主要用例。减为一槽可以作为额外诊断，但属于另一配置；也不必为了“缩小槽位”擅自修改 inbox 的地址步长。当前 `BULK_L3_BATCH=128` 是**每 lane**容量，总 journal 为 4096；协作收集的编译约束要求每 lane 至少 32，即总容量至少 1024。

### 2.4 用例 E3：距离已经改善，L2 尚未提交

接收路径 `SSSP/l3/l3_receive.cuh:98` 执行 owner `atomicMin`，`:153`/`:200` 提交 L2，`:242` 才 ACK。该函数由 manager warp 0 调用，而本卡终止状态机也由 warp 0 执行。

因此应把原清单中的测试分成两个真实窗口：

- **提交前窗口**：在 owner 已更新距离、RX journal 未提交时做有界暂停。观察者/对端确认该批仍未 ACK，本卡也未完成退出；释放后提交并传播。
- **冻结遇到新 inbox**：先让本卡处于 QUIESCING，再允许对端最后一个 READY 到达。观察 manager 先取消当前终止请求、恢复 workers，再进入 RX 提交路径。位置为 `SSSP/sssp_run.cu:7009`。

不能让暂停的 RX warp 等待“由自己下一轮才会设置的终止状态”，否则死锁由测试代码制造。观察和放行必须来自能独立推进的角色；钩子应 warp 一致、有超时，并避免占满全部 SM 的额外常驻内核。

**旧 key=10、新距离=4 子用例**需要保证旧任务仍在排队、新值到达且目标有必须传播的后继边。记录入队 key、过滤判断、展开时读取的 owner 距离和后继结果。当前源码不实施通用“已在队列所以不再激活”的去重，不能为了复现审阅反例向正式路径加入这样的行为。

若旧 key 被 `last_processed` 过滤，仍可能是合法行为；检查是否有更小 key 的任务或恢复责任覆盖，而不是要求旧任务必定执行。若旧任务执行，则当前边松弛读权威距离的事实应能在 trace 中核对。

### 2.5 用例 E4：最后一批与终止确认交错

建议覆盖两个不同安全点：

1. manager 的初步空闲观察之后，TX 在其安全点确认之前又发布一批；manager 必须在收齐 TX ACK 后检查**最新** published epoch，而不是沿用旧值。
2. 本卡 QUIESCING/准备 READY 时，对端产生最后一批必要更新；本卡取消旧确认并继续处理，最终重新建立完成条件。

源码定位：`SSSP/sssp_run.cu:8173`、`:8209`、`:8224`、`:8303`、`:8374`。

trace 中明确区分本地终止 token、传输 epoch 和 query ID。两卡的本地 token 无须相同；要求每个参与角色确认本卡当前 token，旧 token 不可用于本次完成。

冻结后发现权威 mark 非空还应有排空反馈证据；否则可能仅反复取消/重新冻结而不能前进。用一条反复跨 owner 的有向路径加不可达顶点，可以检查晚到更新和两侧交替空闲；shortcut 保持 owner-local，不能把跨卡传播本身消掉。

最终不要求所有槽位字段都为零：正常已消费槽位可以为 DONE，epoch/ACK 可以非零。要求没有未提交责任、没有未确认的最新发送、没有必要工作且所有内核退出。

### 2.6 用例 E5：同进程不同源点的 L3 reset

**不得直接使用现有 `-s` 和 `run_multisource_repeated.sh`。** `main.cu:209` 明确将每卡按逻辑 `n_gpu=1` 初始化完整图；`:235` 调用 `sssp_setup_independent`，它测试的是源点并行。当前 shortcut 构建在 `main.cu:396` 拒绝此入口。

建议增加测试专用的**顺序查询**适配路径，保留双卡固定 owner 划分和 peer 连接。以单个原图或一次冻结的 `G+`，在同一进程中运行：

```text
source A（GPU0） -> source B（GPU1） -> source I（孤立/小可达分量） -> source A
```

每个查询都同时调用两卡的普通双卡求解路径。上次两卡内核完全结束并收集结果后，才 reset 下一次；两卡 reset 和同步都完成后，才跨过 benchmark 的 pre-launch barrier。不能让一张卡开始新查询而对端尚在清除槽位。

当前 reset 核心已存在：

| 状态 | 现有位置 | 应检查的初始条件 |
|---|---|---|
| `node_data`、global exit | `sssp_re_init:10747` | 全 INF、退出标志 0；随后只按新源点激活 |
| `last_processed`、dirty/hints、local idle | `l3_host_context::reset_query:557` | INF/清零 |
| candidate、peer cache、mark/hints | `sssp_run.cu:714` | candidate/cache 为 INF，位图为 0 |
| inbox count/epoch/ACK/state/generation | `sssp_run.cu:640` | 控制字段回初始状态 |
| 终止 request/state/ACK slots | `sssp_run.cu:641` | 当前查询的确认状态不继承 |
| L2、L1 host/workspace 状态 | `sssp_run.cu:10978`、`mlmq.reinit_host` | 累计计数与读写位置重置 |
| signal、诊断/故障一次性状态 | `sssp_re_init` 后续分支 | 按当前构建逐查询重置 |
| kernel 私有 epoch、retained journal 等 | 每次重新启动内核 | 不继承上一 launch 的私有执行状态 |

旧 payload 内存可以保留，只要状态字段正确禁止读取旧 generation；不必将整块 payload 清零当作验收条件。用源点 I 使上一查询可达顶点在下一查询变为不可达，是检测残留距离和 cache 的有力用例。

固定 `G+` 在不同源点上仍保距；其构造 source 与 query source 可以不同，必须分别记录。先完成固定图的 reset 实验即可，不必同时加入逐源重构；后者还需要处理 CSR、分区、boundary index 和内存指针更新，是不同的验证范围。

### 2.7 五类用例统一的验收记录

每例至少保存 `case_id/query_id/source/cut/build_hash`、双方 GPU 参与证据、关键事件的发生顺序、宽整数原图参考、全部顶点比较、最终协议检查、退出码和超时状态。

建议状态为 `PASS / FAIL / TIMEOUT / NOT_TRIGGERED / OUT_OF_DOMAIN`。只有交错真实触发、全部顶点正确、协议检查通过且正常终止，才为 PASS。只出现两张卡的设备枚举，不足以证明两卡合作处理了同一次查询。

当前 `node_data_check` 确实会遍历顶点直到发现错误，但控制台只打印前 30 个距离。新的宽整数核验应在结果收集处完成，或导出完整结果向量；仅解析这 30 个数是不够的。

已有 fault512 runner 的 `check_fault_evidence()` 只在所有故障计数合计为零时拒绝，不能保证每卡每种预期故障都被触发。新验收器应按预先定义的“每个角色/每种事件”逐项检查，而不是仅检查某条含 `recovered` 的输出。

隐藏一次 RX 通知后最终正确，也不一定证明特定恢复扫描完成了修复：另一条普通激活可能覆盖它。NY 旧日志中就有某卡 `recovered_scans=0`。若要验证恢复机制本身，fixture 必须排除替代传播，并记录残余改进实际被发现和重新激活的事件。

## 3. 预处理和查询成本：本轮已经能交付什么

### 3.1 已完成的离线归档核对

本轮读取 `tmp/base_today/run_threeway_samebatch`，对应作业 33155，核对了：

- 48 个进程记录，原始日志与 JSON 中的正式 solve 样本相同；
- 240 个正式查询，连同预热共 288 个查询；
- 每个查询的算法、GPU 数、source 和已有正确标记；进程记录 rc 均为 0；
- 双卡日志中 192 条 `L2_FINAL`，均满足 reads=writes=completed；
- 两轮各图的构造/setup 记录，以及 10 正式样本的查询成本中位数和 IQR。

产物：

- [双卡成本明细 CSV](knowledgebase/l3_supplementary_review_20260921/existing_samebatch_costs.csv)
- [离线核查结果与原日志哈希](knowledgebase/l3_supplementary_review_20260921/existing_evidence_audit.json)
- [只读取已有日志的复算脚本](knowledgebase/l3_supplementary_review_20260921/collect_existing_evidence.py)

在仓库根目录执行 `python3 -B knowledgebase/l3_supplementary_review_20260921/collect_existing_evidence.py` 可重新生成上述 CSV 和核查 JSON；该脚本只更新本次离线分析产物，不调用 GPU 求解器。

这些是已有实验的重新整理。`correct=1` 沿用原程序的核验含义，本轮没有运行新的宽整数 GPU 比较；最终 L2 计数相等也不单独证明容量安全。

### 3.2 可直接作为成本披露依据的表

以下均为毫秒。构造列有**两个进程观测**；query 各列分别为**合并 10 个正式样本的中位数**。这里报告的是完整双卡配置。

| 图 | shortcut 构造 r0 / r1 | prepare | solve | solve IQR | collection | query wall |
|---|---:|---:|---:|---:|---:|---:|
| NY | 15.185 / 15.242 | 2.613 | 6.426 | 0.241 | 0.155 | 9.311 |
| BAY | 19.021 / 19.092 | 2.720 | 4.709 | 0.132 | 0.170 | 7.688 |
| COL | 29.739 / 30.081 | 2.648 | 7.020 | 0.743 | 0.192 | 9.935 |
| FLA | 67.467 / 67.071 | 2.633 | 17.298 | 1.257 | 0.515 | 21.027 |
| CAL | 131.442 / 129.218 | 2.651 | 18.042 | 0.644 | 0.622 | 21.464 |
| E | 243.885 / 243.546 | 2.627 | 24.634 | 3.482 | 1.240 | 28.596 |
| W | 443.904 / 440.942 | 2.682 | 29.896 | 0.286 | 1.758 | 34.594 |
| USA | 1701.669 / 1722.180 | 2.797 | 67.972 | 1.414 | 9.072 | 80.063 |

还有约 0.04–0.05 ms 的 `post_solve` 中位开销，完整值见 CSV。**独立取中位数后的 prepare、solve、post、collection 不必恰好相加等于 query wall 中位数。** 不能把这个统计差异当作计时错误或强行平账。

### 3.3 哪些时间互相包含

源码位置：`SSSP/main.cu:308`、`:469`、`:745`、`:755`、`:774`、`:808`；`SSSP/benchmark.h:36`、`:46`。

| 字段 | 实际边界 | 是否已包含在其他字段 |
|---|---|---|
| `L3_CHAIN_SETUP build_ms` | host shortcut 构造区间 | 在该进程 `setup_wall_ms` 内 |
| `L3_BOUNDARY_INDEX build_ms` | 各卡 boundary index 的 host 构造/准备区间 | 在 setup 内；不是新增的独立总时间 |
| `BENCH_SETUP setup_wall_ms` | 从 main 起始时刻到重复查询循环之前 | 包含 input、CPU reference、shortcut、分区及此前 CUDA/setup 开销 |
| `initialization_other_ms` | `setup_wall-input-reference` 的剩余量 | 包含 shortcut/index 等，不等同于它们之外的纯净 setup |
| `prepare_ms` | 本 query 开始到所有 GPU 准备完毕的 barrier 起点 | 在 query wall 内；包括 reset、同步、主机线程准备及首次查询可能的工作区分配 |
| `solve_ms` | barrier 起点到最后一张 GPU 同步完成后的 finish | 在 query wall 内，覆盖 launch、计算、通信、恢复、终止和同步 |
| `post_solve_ms` | solve 结束到开始收集 | 在 query wall 内 |
| `collection_ms` | 按 owner 区间收集结果到 host | 在 query wall 内 |
| `query_wall_ms` | query 开始到结果收集完成后 | 不含循环前 setup；也未计后面的逐顶点比较和全部诊断输出 |

逐样本有近似分解：

\[
T_{query}=T_{prepare}+T_{solve}+T_{post}+T_{collection}+\epsilon.
\]

本轮核对的 96 个双卡查询中，分解差值最大约 0.000146 ms；时间戳调用之间的小间隙及输出精度会形成残差。不要给四个字段都加上 shortcut 构造时间，也不要把 `setup_wall` 与其组成部分重复累加。

两轮 setup wall 的观测如下，**包括 CPU reference 和进程初始化，不是纯算法预处理成本**：

| 图 | r0 setup wall/ms | r1 setup wall/ms |
|---|---:|---:|
| NY | 2495.314 | 374.608 |
| BAY | 387.679 | 389.258 |
| COL | 416.286 | 414.066 |
| FLA | 551.362 | 580.938 |
| CAL | 757.981 | 756.269 |
| E | 1177.457 | 1183.802 |
| W | 1887.962 | 1885.724 |
| USA | 6769.703 | 6757.526 |

NY 两轮差异明显，不能直接将其中差值归因于图构造或 L3；现有粗粒度日志不足以拆解其原因。表中公开两次观测比把它们包装为“十次稳定 setup 中位数”更准确。

### 3.4 当前成本披露的边界

这份日志可以补“已测构造成本、prepare/solve/result 边界”，但还缺少：既有 landmark 重排成本的本批重测、完整 host/GPU 峰值内存、所有 setup 子阶段的互斥分解，以及固定 `G+` 跨不同源点查询的实际复用收益。

ADDS 的参数 profiling 在其 solve 之前，另输出 `parameter_ms`；`complete_solve_ms` 已包含 parameter+solve。它与 MLMQ 的 prepare 不是同一个实现阶段。对照时应按所声明的边界分别披露，不能将某一边额外阶段悄悄并入另一边的 solve。

对于当前限定的 solve-time 论文主张，**这批已有成本表可以先用于披露**，不必自动扩成端到端评测。若新的前置修复或导出适配改变了相关路径，则需重新记录新版本成本。

### 3.5 摊销应使用增量成本和真实复用范围

如果未来要讨论处理 `N` 个查询，应比较：

\[
T_{total,c}(N)=T_{setup,c}+\sum_{i=1}^{N}T_{query,c,i}.
\]

若两个配置共享一部分布局成本，回本取决于**额外 setup 差值**，不是任一方所有 setup 的绝对值。在查询节省为正且复用协议成立的理想模型下：

\[
N_{break-even}\approx
\frac{T_{setup,new}-T_{setup,base}}
{\bar T_{query,base}-\bar T_{query,new}}.
\]

这只是设计公式。本轮没有给出回本次数，因为目前缺少同一固定图复用协议下的完整增量成本；用 USA 的某次 host 构造时间除以另一批 solve 差值，不能写成实测回本。

## 4. 四项前置核查：什么才算关闭

### 4.1 Candidate/mark 和 P2P：建立按字段的顺序合同

建议先形成一个小表，分别列出 candidate→mark、mark 领取→candidate exchange、payload→READY、READY→payload 读取、L2 commit→ACK、ACK→slot reuse，以及 token/ACK/READY 的双方访问。

每项记录：数据在哪张卡、谁写谁读、原子与普通访问的组合、所需顺序、实现的 fence/同步/作用域、平台能力。不能只列一句“用了 atomic 和 system fence”。

CUDA 12.4 将普通无后缀原子描述为 device scope、relaxed ordering；因此 fence 和原子作用域应分别核查。[CUDA 12.4 原子说明](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#atomic-functions)

`main.cu:631` 现有初始化检查 peer access；建议在目标节点对两个方向额外记录 `cudaDeviceGetP2PAttribute(..., cudaDevP2PAttrNativeAtomicSupported, src, dst)`。这个能力查询表示链路是否支持 native atomic，但本身不能证明具体代码已建立所需同步关系。[CUDA 12.4 P2P 属性定义](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__TYPES.html)

**关闭产物**应是代码对应的顺序论证、目标平台能力记录，以及受控和无钩子验证。测试通过与内存模型证明是互补证据，不能互相替代。若需修改访问方式，冻结修改后的版本，再做四配置性能实验。

### 4.2 L2：先区分存储复用安全和工作完成计数

`cu_delta_queue.cuh:460` 预留写位置，`:467` 对桶容量取模，`:489` 总是返回成功。当前没有满检测。读出后复制到私有输入，再增加 `bucket_read_done`；全部求解责任完成则通过另一个 `read_done` 退休。

因此至少有三个不同概念：

- 预留但尚未发布的条目；
- 尚未安全读出的存储槽位；
- 已出队但仍在执行的工作责任。

**不能仅凭某时刻 `write_reserve-bucket_read_done` 小于容量，就认定所有取模槽位都可安全复用。** 多个读者可以非按序完成，累计数量未必等于连续释放前沿；`block_write_done` 的代次/清零也必须一致。

最小核查顺序：

1. 打印实际 `sizeof(NODE_TYPE)`、分配字节数、bucketNum、每桶 `total_size`、计数器类型，核对 2 GiB 宏经 `int max_size` 传递的范围问题。
2. 采用独立宽计数或阈值检查确认原计数器没有回绕；不要仅从最终非负值推断。
3. 若能证明每查询每桶累计预留始终小于该桶物理容量，则本查询根本不发生环形覆盖，这是比一般环形复用证明简单的充分条件。
4. 若会发生 wrap，则需要验证具体槽位代次与读取完成关系，或实现真实容量管理；仅测峰值 outstanding 不够。

本轮同批日志中的最大“每卡每查询累计 L2 写数”为 USA 的 **7,723,591**（含预热检查）。这是后续容量核查的有用尺度，但原日志未打印实际每桶容量，最终计数也不能排除所有回绕问题，因此**尚不能据此宣布容量项关闭**。

`L2_FINAL` 目前核查每桶读写和全局完成守恒，但打印的是合计，不是峰值和槽位历史。保留它作为必要证据，同时增加前述针对性容量依据即可；不必在已证无 wrap 的输入范围内先重写整个通用队列。

如果最后选择“满则等待”，需要重查该等待是否依赖已经冻结的 workers 或正等待 ACK 的对端。不能只增加一个自旋循环就认为进展问题已解决。

### 4.3 数值范围：现有 CPU reference 也要纳入审核

`main.cu:92` 的参考解使用 `VALUE_TYPE`，当前为 int；`:123` 的 CPU 加法与 GPU 的距离加边权一样没有在此处使用 64 位保护。故“CPU/GPU 一致”不能单独排除共同的越界算术。

补测应采用独立 `int64_t`/更宽或 Python 整数 Dijkstra，并明确：

- 有限可表示距离必须小于 `INT_MAX`，与无穷哨兵区分；
- 非负权、shortcut 路径和、**所有实际尝试的**有限 distance+weight 都需满足范围；
- 不能仅以最终最短距离很小作为保证，较差中间路径仍可能被松弛；
- DQ 中 `first_pos*delta`、`key-base`、位置和计数运算也需要范围检查，见 `cu_delta_queue.cuh:429`、`:440`。

低权重小图可以用保守上界建立安全域；复杂道路图可增加诊断版宽整数检查，记录尝试加法的最大值及越界次数。诊断运行本身仍不是所有输入/调度的普遍证明；论文应限定输入数值域，或在实现中提供明确的检查/饱和/更宽类型合同。

支持域内边界用例要求正确；超域用例应按事先声明的接口拒绝或被测试框架标记 `OUT_OF_DOMAIN`，不能将溢出后的错误结果算作算法支持。

### 4.4 管理角色进展：W512 需要它自己的资源证据

当前每卡先 launch manage kernel，再在另一 nonblocking stream launch work kernel；每卡 107 个 work blocks，W512 的实际 block 是 544 threads。代码中的一项 occupancy 检查只覆盖 320/384 分支，未覆盖 W512。

源码定位：`SSSP/sssp_run.cu:11144`、`:11148`、`:11181`、`:11225`。

正式归档应记录实际 SM 数、两个 kernel 的寄存器和 shared memory 用量、launch 几何及并发执行条件。在诊断版中确认管理和工作角色都有推进记录。可使用各角色本地时间差或主机/CUDA trace；不要跨 GPU 或任意不同 SM 直接相减 `clock64()` 当成全局时间。

已测正常退出支持这些具体运行的进展。它不能被表述为“只要先 launch manager，任何硬件上都保证无死锁”。

## 5. 选做实验：源码现有计数能支持哪些主张

### 5.1 聚合指标必须先修正定义

计数位置见 `SSSP/l3/l3_diagnostics.cuh` 和 `SSSP/sssp_run.cu:2105`、`:2156`、`:9700`、`:11676`。

| 需要的指标 | 当前字段 | 实际含义/注意事项 |
|---|---|---|
| 远端松弛尝试 `R` | `remote_attempts` | 进入远端分支的次数；普通有效 peer 配置下使用 |
| cache 原子胜出 `C` | `candidate_updates` | 在 `new_dist < cache_old` 后增加，**早于**真正 candidate 原子更新 |
| candidate 严格改进 `U` | `signal` / `g_bulk_mark_signal` | `l3_record_candidate` 严格胜出后增加；比 `C` 更准确对应该指标 |
| cache 预读拒绝数 | `cache_filtered` | 仅统计可选 `BULK_CACHE_PREFILTER` 的早退；该开关默认关闭，不是全部 cache 拒绝数 |
| 提取条目 `X` | `extracted` | 本轮从 candidate 领取到 journal 的条目数；不是在此处确认发布成功 |
| 非空收集批次 `B` | `tx_batches` | 发布之前增加，不能不加条件命名为成功发送批数 |
| owner 接收条目 `Y` | `received` | 接收完成统计，可按方向核对发送侧排空后的责任守恒 |
| owner 严格改进 `W` | `improved` | owner 上严格胜出并进入普通接收激活的计数；故障抑制路径需另标 |
| 批次样本 | `waves[32]` | 仅前 32 批的 count/min/max，不能代表全程分布 |
| 人为发布失败 | `injected_publish_retries` | 仅指定 fault 注入次数，不是自然槽位忙重试次数 |

一个具体交错解释 `C != U`：producer A 以 10 降低 cache 后暂缓；producer B 以 4 再降低 cache 并更新 candidate；A 随后写 candidate 10 不会胜出。此时两次 cache 胜出，只发生一次 candidate 严格改进。因此不能按字段名称把两者合并。

可以报告以下定义明确的比值，分母为零时写 N/A：

\[
f_{cache}=1-C/R,\quad
f_{coalesce}=1-Y/U,\quad
f_{effective}=W/Y.
\]

这些式子应限定普通配置、查询已经完整排空且双向计数按发送/接收对应汇总。`f_coalesce` 描述事件到接收记录的压缩，不是直接测得的字节带宽下降；协议控制流量、不同记录格式和重复发送等因素应另行说明。若出现与预期不符的负值/守恒差异，保留原数并检查定义和构建，不做截断来美化比例。

若需要精确的成功发布批数/条目数，在初次 publish 成功和 retained publish 成功两个分支分别记录，避免重试重复计入。若需要完整批大小分布，添加 histogram 或足够容量且有 overflow 检查的 trace；不要把前 32 批绘成全程统计。

这些热路径计数会扰动调度，甚至改变合并数量。正式耗时采用关闭新增诊断的构建，诊断构建用来解释数量级和机制；二者分别归档。不能把不同执行的角色时间直接拼成一个无重叠的 solve 时间分解。

### 5.2 聚合“减少记录”和聚合“贡献加速”是不同结论

若正文仅解释先聚合再传输的结构，给出定义和少量计数足够。若声称聚合贡献了某个速度收益，还需要固定 `G+`、双卡、cut、worker 和其余协议的受控对照。

当前 `BULK_NO_CACHE=true` 只关闭前置 cache 过滤，**不会关闭 `remote_cand` 的 atomic-min 合并**，因此不能将它命名为“无聚合版本”。构造真正的无合并传输基线还会涉及流量、容量和正确性变化，属于额外实现工作；若不是论文保留的核心主张，可以暂缓。

代表图建议先选 NY/W/USA，分别覆盖小图固定开销、中等规模和最大输入；这是后续选择建议，不是已经完成的机制实验。

### 5.3 其他源点：先明确实验问题

两种协议回答不同问题：

- **固定增强图复用**：固定 `G+=F(G,π,s0)`，在未用于调参的 `s1,s2,...` 上求解。检验固定图及分区下的跨源行为，适合复用场景。
- **逐源重构**：每个 `si` 构造自己的 `G+i=F(G,π,si)`。同一源点的 M1/M2 共享这份图，并记录各自构造成本，不能称为零重构开销。

主实验仍保留当前八图固定源点。可选扩展优先在两个 owner 和不可达/小可达区域各选源点；source 选择规则在运行前固定，不能根据最终加速挑选。顺序复用不同源点时必须使用 §2.6 的真实 L3 查询路径。

### 5.4 窗口敏感性：改变 max 不等于比较固定窗口

当前 mode=2、min=max=25000，因此普通窗口固定在 25000。若只把 `L3_WINDOW_MAX_CYCLES` 改为 50000 或 100000，mode=2 的 budget 可以变化，且 PROBE 可以提前触发；此时比较的是**动态策略上限**，不是固定 25k/50k/100k 窗口。

源码定位：`SSSP/l3/l3_window.h:30`、`:66`；`SSSP/sssp.cuh:769`、`:825`。

按论文主张选择其一：

| 问题 | 合法实验设计 | 应使用的名称 |
|---|---|---|
| 当前策略对上限是否敏感 | 固定 mode=2，max=25k/50k/100k | 窗口上限敏感性；同时报告实际 budget/PROBE 行为 |
| 固定轮询间隔如何影响性能 | mode=1，分别固定 max=25k/50k/100k，另保留当前 mode=2/25k 对照 | 固定普通扫描窗口比较；终止反馈排空仍需一致 |
| 低于25k是否更好 | 需修改下限及相关规则并重新验证 | 新配置/新策略，不能冒称现有参数扫描 |

延迟指标应先定义起止事件。例如 candidate 严格改进→取入 journal 是发送卡本地延迟；发布→owner 提交跨卡，不能直接相减两个 GPU 的 `clock64()`。没有统一时间基准时，可测发送端观察完整 ACK 的往返区间，或采用校准后的 trace，并明确它不是纯链路延迟。

若正文只报告已测参数而不声称自适应、最优或低延迟硬界，这项实验可暂缓。

## 6. 现有脚本的复用与归档要求

### 6.1 哪些脚本可借鉴，哪些不能原样执行

| 文件 | 可复用部分 | 必须调整/检查 |
|---|---|---|
| `run_l3_threeway.py` | 两轮反向、warmup/formal 解析、已有输出结构 | 增为四配置；清理继承环境；检查预热、GPU 数、source、cut、哈希；禁止覆盖历史目录 |
| `run_chain_no_l3_ablation.py` | 独立单卡 adapter 的验证思路 | 硬编码旧二进制、半切分和三正式样本；缺 M2-original；不能作为当前四配置直接重跑 |
| `run_l3fault512.py` | 有限故障类别和 fixture 来源 | 按角色核验触发；显式 cut；不能只检查所有故障合计非零；超时 bytes/text 处理也需统一 |
| `test_l3_reset.cu` | typed fill 底层检查 | 不代替完整 L3 换源查询；保留为单元测试即可 |
| `run_multisource_repeated.sh` | 不适合作为此次 reset 主测试 | `-s` 的源点并行语义绕过 L3 |
| `usa_recovery_confirmation_runner.py` | 环境记录、资源/拓扑记录、保留返回码 | 使用清理功能后重设本次 cut 等参数；不要直接继承不相符的旧统计假设 |

当前 `run_l3_threeway.py` 的正式通过判定主要检查正式样本数量和 correct/algorithm；新 runner 应同时核对预热是否成功、实际 `gpu_count/source` 和 config。失败预热不能被正式样本成功掩盖。

### 6.2 推荐产物结构

以下是后续实验的目录设计建议，本轮没有创建其 GPU 运行内容：

```text
experiment_<version>_<job>/
  manifest.json               # 源码/二进制/图/布局/参数/环境
  platform/                   # 分配设备、UUID、驱动、拓扑、P2P、launch 资源
  inputs/                     # G 与 G+ 的哈希、映射、构造和读回验证
  correctness/                # E1-E5 case、trace、wide reference、退出状态
  performance/                # 四配置逐进程原始输出与逐样本 records
  costs/                      # 两次 setup 观察与逐query成本，明确包含关系
  diagnostics/                # 可选，单独构建和计数定义
  analysis/                   # 聚合脚本、四配置表、ratio 和 dispersion
```

manifest 至少记录：source snapshot/hash、完整编译命令或预处理宏清单、binary hash、实际 GPU 架构、所有相关环境变量、图与增强图内容哈希、cut/source、queue/worker、计时起止点、统计定义、超时阈值和错误状态。

当前冻结 `macros.txt` 不完整，缺少 direct RX/retained TX 等必要配置；不能把它作为新实验唯一的构建清单。Git HEAD 同样不足以代表这份有未提交改动的源码。

### 6.3 完成顺序与每阶段停止条件

| 阶段 | 具体产出 | 进入下一阶段的条件 |
|---|---|---|
| A：定义与前置核查 | 顺序合同、容量/数值支持域、实际 W512 资源记录 | 关键边界已说明；需要的修复完成并冻结，不能将待核查项写成通过 |
| B：输入适配 | 固定 G+，读回哈希一致，与 G 的参考保距 | 两种 GPU 数量确实消费相同 CSR，未发生二次增强 |
| C：E1–E5 | 宽参考、触发 trace、退出和最终责任检查 | 目标交错已触发且通过；未触发/超时单列 |
| D：八图四配置 | 64 进程、320 正式样本及全量输出 | 全部单元格有明确状态，不删减速/慢值/失败 |
| E：成本与论文统计 | 新版本成本表、同图比值、图与 provenance 一致 | 比较对象、批次和统计口径统一 |
| F：选做机制 | 与保留主张相对应的计数或消融 | 只完成论文实际需要的部分 |

既有日志整理可以与 A/B 的分析并行完成，本轮已经做完这项独立工作。前置核查“关闭”可以是明确且有证据的适用域，不自动意味着必须重写一个适用于任意图的通用队列；但不能用一条未经验证的假设冒充关闭。

## 7. 如何将结果写回论文

### 7.1 正文只需要少量呈现

建议新增一张四配置表/消融图，一张成本补充表，以及一段简短正确性覆盖说明。完整源码钩子、事件轨迹、宽参考结果、平台与参数记录放实验材料。

主表应同时给出四个绝对时间和关键同图比值。若 `S_same_graph` 接近 1 或小于 1，而 `S_complete>1`，应说明组合配置主要得益于图增强及其与调度的相互作用，不能继续说 L3 独立加速明显。反之，若同图比值稳定大于 1，则可在已测双卡平台和固定输入范围内说明双卡扩展收益，仍不能自动归因某个通信子机制。

现有 33155 三方数据仍然是有效的完整配置结果，但不是四配置实验。当前图表 provenance 的旧/新批次混用问题也应一起修正，不能只替换正文数字。

### 7.2 已有日志允许现在就写的成本说明

以下是依据当前记录的新拟英文文本：

> The reported solve interval excludes graph loading, host-side chain augmentation, query preparation, and result collection. In the archived two-GPU runs, chain augmentation was performed once per process before repeated queries; its two recorded construction times were 1.702 s and 1.722 s for USA-road. The median solve interval for that graph was 67.972 ms across ten measured runs. These measurements describe separate construction and solve costs and do not establish an end-to-end speedup or a preprocessing break-even point.

这段可以作为已有成本证据，但若最终论文采用新版本的四配置数据，应同步更新成本批次。

### 7.3 四配置完成后可采用的实验方法段

下面描述的是**拟执行协议**，实验完成前不得写成已完成过去时。为便于后续使用，给出完成后可改用的模板：

> We separate graph augmentation from device-count effects using four configurations: one and two GPUs on the reordered input graph, and one and two GPUs on the same frozen augmented CSR. The augmented graph is constructed once using the fixed two-GPU partition and construction source, and is not augmented again by either consumer. Local queue settings, per-GPU worker geometry, and common correctness fixes are aligned across configurations. Each configuration is measured in two reversed interleaved rounds with one warm-up and five measured queries per round. We pool the ten measured samples per graph and configuration and report the median and interquartile range, retaining all slow samples.

其中“common correctness fixes are aligned”必须由实际源码对照支撑；无法对齐时改为披露具体差异，不保留这句话掩盖差异。

### 7.4 定向验证完成后可采用的覆盖说明

> The supplementary tests exercise candidate-claim interleavings, occupied inbox slots and delayed acknowledgements, receive-to-activation handoff, late messages during quiescence, and sequential source changes within the same two-GPU L3 process. Each accepted case includes evidence that the intended event occurred, comparison of all vertex distances against an independent wide-integer reference on the original graph, and normal termination. These tests complement the ordering and capacity contracts rather than replace them.

当前只有部分历史测试和本轮离线核查，尚不能将上述整段作为已完成的事实写入论文。

## 8. 本轮已完成与仍待执行

**本轮已完成：** 当前源码与冻结包一致性复核；原清单逐项实现映射；五类交错的可执行设计；multi-source、计数语义、旧 fault cut 等差异识别；现有三方日志的成本表和离线核查归档；本文及复算材料。

**仍待执行：** 前置合同的最终关闭及必要修复、冻结 G+ 的适配与读回验证、E1–E5 的新真实双卡运行、当前版本八图四配置正式实验，以及论文选择保留的机制实验。

因此，这份补充清单可以继续采用，但应按本文修正入口、触发条件、参考解和计数定义。其验收目标是解释清楚**相同图上的双卡表现与完整配置收益之间的关系，并以实际执行路径验证关键责任交接**，而不是把所有图调到某个预设加速门槛。

## 附录：关键源码入口

| 核查事项 | 源码链接 |
|---|---|
| 主机增强入口 | [main.cu](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/main.cu:469) |
| 现有多源是独立查询分派 | [run_multi_source_batch](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/main.cu:209) |
| 32 位 CPU reference 加法 | [sssp_sequential](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/main.cu:123) |
| `.gr` 读取和序列化 | [csr_graph.cu](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/csr_graph.cu:185) |
| 普通 query reset | [reset_query](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/sssp_run.cu:557) |
| candidate/cache 计数位置 | [relax_dst](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/sssp_run.cu:2102) |
| retained journal 重试 | [TX 重试](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/sssp_run.cu:9105) |
| 接收后 L2 提交和 ACK | [l3_receive.cuh](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/l3/l3_receive.cuh:88) |
| 新 inbox 取消终止 | [manager 接收入口](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/sssp_run.cu:7009) |
| DQ 预留、取模和发布 | [cu_delta_queue.cuh](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/core/cu_delta_queue/cu_delta_queue.cuh:427) |
| 最终 L2 计数含义 | [report_final_l2_counts](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/sssp_run.cu:10907) |
| 窗口固定/动态选择 | [l3_window.h](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/l3/l3_window.h:30) |
| 热路径诊断结构 | [l3_diagnostics.cuh](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/SSSP/l3/l3_diagnostics.cuh:5) |
| 三方批次执行与结果判定 | [run_l3_threeway.py](/mnt/709/data3/home/Dingzhong/src/mlmq_ppopp26/scripts/multigpu/run_l3_threeway.py:37) |
