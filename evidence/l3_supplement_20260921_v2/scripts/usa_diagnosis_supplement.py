#!/usr/bin/env python3
"""Create offline U1 progress and U2 interpretation supplements."""
import argparse
import json
from pathlib import Path
from statistics import median


AUDIT = "FINAL_AUDIT mismatches=0 residual_edges=0 cross_residual_edges=0"


def fields(line, prefix):
    return dict(token.split("=", 1) for token in line.split()[1:]
                if "=" in token) if line.startswith(prefix) else None


def formal_progress(path):
    pending = []
    records = []
    lines = path.read_text().splitlines()
    for line in lines:
        if line.startswith("L3_PROGRESS "):
            pending.append(fields(line, "L3_PROGRESS "))
        elif line.startswith("BENCH "):
            bench = fields(line, "BENCH ")
            if bench.get("warmup") == "0":
                if len(pending) != int(bench["gpu_count"]):
                    raise ValueError("progress/bench grouping mismatch: " + str(path))
                if any(line != AUDIT for line in lines if line.startswith("FINAL_AUDIT ")):
                    raise ValueError("nonzero audit: " + str(path))
                for row in sorted(pending, key=lambda item: int(item["gpu"])):
                    span = int(row["span"])
                    records.append({
                        "pair": int(path.stem[4:7]),
                        "gpu": int(row["gpu"]),
                        "span_cycles": span,
                        "first_rx_cycles": int(row["first_rx"]),
                        "first_rx_lifetime": int(row["first_rx"]) / span,
                        "last_rx_cycles": int(row["last_rx"]),
                        "first_winner_cycles": int(row["first_winner"]),
                        "last_commit_cycles": int(row["last_commit"]),
                        "post_commit_lifetime": (span - int(row["last_commit"])) / span,
                        "rx_cycles": int(row["rx_cycles"]),
                        "rx_lifetime": int(row["rx_cycles"]) / span,
                        "backstop_cycles": int(row["backstop_cycles"]),
                        "backstop_lifetime": int(row["backstop_cycles"]) / span,
                        "backstop_calls": int(row["backstop_calls"]),
                        "batches": int(row["batches"]),
                        "items": int(row["items"]),
                        "winners": int(row["winners"]),
                    })
            pending = []
    if len(records) != 2:
        raise ValueError("expected two formal GPU records: " + str(path))
    return records


def summary(rows, key):
    values = [row[key] for row in rows]
    return {"median": median(values), "min": min(values), "max": max(values),
            "values": values}


def pct(value):
    return "%.2f%%" % (100.0 * value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    u1 = root / "U1_clean_vs_progress_usa"
    rows = []
    for path in sorted(u1.glob("pair*_candidate.log")):
        rows.extend(formal_progress(path))
    if len(rows) != 6 or {row["pair"] for row in rows} != {0, 1, 2}:
        raise ValueError("expected exactly three formal pairs")
    raw = {"source": "U1_clean_vs_progress_usa/*_candidate.log",
           "warmup_excluded": True, "formal_queries": 3, "rows": rows}
    (root / "u1_progress_raw_fields.json").write_text(json.dumps(raw, indent=2) + "\n")

    explanations = """# U1 L3_PROGRESS 原始字段解释

数据来自 USA `U1_clean_vs_progress_usa` 的 progress candidate 日志，只保留三次 `warmup=0` 正式查询；每次查询保留 GPU0/GPU1 各一行。所有周期是本卡同一个 manager warp 内的 `clock64()` 差分。

| 字段 | 含义 | 本报告中的处理 | 不能据此推出 |
|---|---|---|---|
| `span` | 本卡 manager 从开始到结束的生命周期周期数 | 作为本卡比例分母 | 不能跨 GPU 比较绝对周期，也不能当作墙钟时间 |
| `first_rx` | 首次成功领取远端 inbox 的相对周期偏移 | `first_rx/span` | 不是纯通信延迟，包含源卡本地推进时间 |
| `rx_cycles` | 成功领取后到完成接收服务/ACK 的累计本卡周期 | `rx_cycles/span` | 不是跨卡传输耗时，也不是全 GPU 接收总耗时 |
| `backstop_cycles` | 本卡协作 backstop 扫描及等待注入 helper 完成的累计周期 | `backstop_cycles/span` | 不能全部称为浪费，可能包含必要恢复责任 |
| `backstop_calls` | backstop 调用次数 | 直接汇总 | 不能单独表示每次调用耗时或关键路径占比 |
| `last_commit` | 本卡最后一次 winner 提交的相对周期偏移 | `(span-last_commit)/span` | 末尾仍可能有本地松弛、排空、恢复和终止工作 |
| `batches/items/winners` | 领取批次数、领取记录数、winner 提交数 | 作为原始上下文保留 | `items` 含候选记录，不能称为唯一有效边或唯一有效工作 |

比例只在同一张卡、同一次查询内有意义。报告的中位数和范围只描述三次正式样本，不能替代跨卡校准后的墙钟关键路径分析。
"""
    (root / "u1_progress_raw_field_explanation.md").write_text(explanations)

    by_gpu = {}
    keys = ("first_rx_lifetime", "rx_lifetime", "backstop_lifetime",
            "backstop_calls", "post_commit_lifetime")
    for gpu in (0, 1):
        gpu_rows = [row for row in rows if row["gpu"] == gpu]
        by_gpu[str(gpu)] = {
            key: summary(gpu_rows, key) for key in keys
        }
    supplement = {
        "kind": "usa_diagnosis_offline_supplement",
        "u1": {"formal_rows": rows, "summary_by_gpu": by_gpu,
                "warmup_excluded": True},
        "u2_interpretation": {
            "recovery2_independently_confirmed": False,
            "event_screen_ratio": 1.0923912637601425,
            "recovery2_screen_ratio": 1.091659218535563,
            "difference_percentage_points":
                (1.0923912637601425 - 1.091659218535563) * 100.0,
            "selected_for_confirmation": "event",
            "selection_rule": "maximum 3-pair USA screening median ratio",
            "limitation": "event and recovery2 screening ratios differ by only 0.073 percentage points; recovery2 had 3/3 wins and CI above 1 in screening, but was not independently confirmed",
            "event_confirmation_ratio": 1.0125811491448355,
            "event_confirmation_ci95": [0.7930717396393295, 1.0906905696801124],
            "interpretation": "event confirmation failure does not reject recovery2; recovery2 remains an unconfirmed candidate and its evidence is stronger at screening level than event by wins/CI, subject to the small screening sample",
        },
    }
    (root / "report_supplement.json").write_text(json.dumps(supplement, indent=2) + "\n")

    lines = [
        "# USA U0-U2 离线交付补充报告",
        "",
        "本文件只补充已有 U0-U2 结果，不新增 GPU 实验，不改变测试选择，不调参；U1 统计排除全部预热。",
        "",
        "## U1 progress 正式查询",
        "",
        "原始记录见 `u1_progress_raw_fields.json`，字段语义见 `u1_progress_raw_field_explanation.md`。三次正式查询分别为 pair000、pair001、pair002，每次包含 GPU0/GPU1 各一条 progress 记录。",
        "",
        "| GPU | 指标 | 中位数 | 范围（最小-最大） | 三次原始比例 |",
        "|---:|---|---:|---:|---|",
    ]
    names = {
        "first_rx_lifetime": "first_rx / lifetime",
        "rx_lifetime": "rx_cycles / lifetime",
        "backstop_lifetime": "backstop_cycles / lifetime",
        "backstop_calls": "backstop_calls",
        "post_commit_lifetime": "last_commit 后比例",
    }
    for gpu in (0, 1):
        for key in keys:
            item = by_gpu[str(gpu)][key]
            if key.endswith("lifetime"):
                values = ", ".join(pct(value) for value in item["values"])
                med = pct(item["median"])
                span = pct(item["min"]) + " - " + pct(item["max"])
            else:
                values = ", ".join("%.0f" % value for value in item["values"])
                med = "%.1f" % item["median"]
                span = "%.0f - %.0f" % (item["min"], item["max"])
            lines.append("| %d | %s | %s | %s | %s |" %
                         (gpu, names[key], med, span, values))
    lines += [
        "",
        "`first_rx/lifetime` 表示首次成功领取远端 inbox 在本卡 manager 生命周期中的相对位置；它包含源卡本地推进时间，不能解释为消息延迟。`rx_cycles/lifetime` 是本卡同一 manager warp 记录的接收服务比例；`backstop_cycles/lifetime` 包括协作 backstop 及等待 helper 完成的累计本卡周期，不能把全部尾部都算作浪费。`last_commit 后比例` 为 `(span-last_commit)/span`，其中仍可能包含必要的本地松弛、队列排空、恢复和终止工作。",
        "",
        "这些比例不能跨 GPU 相减或相加为双卡墙钟时间；`items` 是候选记录数，不能称为唯一有效边数。",
        "",
        "## recovery2 与 event 选择局限",
        "",
        "recovery2 没有进行独立的 10 对确认。event 的 10 对确认失败，只能说明 event gate 未获得稳定确认，不能据此否定 recovery2。",
        "",
        "筛选阶段 event 比值为 `1.092391`，recovery2 为 `1.091659`，只相差 `0.073` 个百分点。按设计中的“中位数最大”规则选择 event 做唯一确认在流程上是正确的，但这个差距很小，选择并不代表 event 的证据明显更强。recovery2 筛选中 3/3 对 candidate 胜出，且 3 对 bootstrap CI 为 `[1.0334, 1.1299]`；event 为 2/3 胜出，CI 为 `[0.9042, 1.7417]`。就筛选证据强度而言，recovery2 更有支持，但它仍然是未独立确认的候选，不能升级为已确认优化。",
        "",
        "event 唯一确认结果仍为 `1.0126`，CI `[0.7931, 1.0907]`，所以本轮没有改变任何候选的默认状态，也没有继续调参。",
        "",
        "## 交付边界",
        "",
        "本补充没有执行 U3/U4，没有新增诊断插桩，没有修改 `84_usa_causal_plan.md`、`SSSP/core`、历史日志或原始实验文件。",
    ]
    (root / "report_supplement.md").write_text("\n".join(lines) + "\n")
    print("wrote", root / "report_supplement.md")
    print("wrote", root / "u1_progress_raw_fields.json")
    print("wrote", root / "u1_progress_raw_field_explanation.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
