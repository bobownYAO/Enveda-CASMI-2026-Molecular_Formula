"""Independently verify saved ranks and write the Chinese experiment report."""
import json
from pathlib import Path
import numpy as np
from casmi.formula.data import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports/baseline-sampling"


def main():
    protocol = json.loads((OUT / "sampling.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (OUT / "records.jsonl").read_text().splitlines()]
    records = {r["molecule_id"]: r for r in rows}
    selected = set().union(*map(set, protocol["rounds"].values()))
    assert len(rows) == len(records) == len(selected) and set(records) == selected
    for seed, ids in protocol["rounds"].items():
        assert len(ids) == len(set(ids)) == 100
        ranks = [records[k]["rank"] for k in ids]
        expected = summary["rounds"][seed]
        assert abs(sum(0 if k is None else 1/k for k in ranks)/100 - expected["mrr_full"]) < 1e-12
        assert abs(sum(1/k for k in ranks if k is not None and k <= 25)/100 - expected["mrr25"]) < 1e-12
        for k in (1, 5, 10, 25):
            assert abs(sum(r is not None and r <= k for r in ranks)/100 - expected[f"top{k}"]) < 1e-12
    for row in rows:
        assert row["rank"] is None or 1 <= row["rank"] <= row["candidate_count"] <= 10000
        assert row["status"] != "resource_limit" or row["rank"] is None
    lines = ["# 基线分子式多轮采样评估", "", "## 评估口径", "",
        "- 从质控后留出集的 26,951 个结构均匀采样；每轮 100 个，种子 101–105，轮内无重复、轮间独立采样。共 500 次评估，493 个独立结构。",
        "- 使用每个结构全部通过质控的谱，模块内部去除完全重复谱。抽样结构与训练集、验证集均零交叉。",
        "- 使用 baseline-v1：支持谱数量优先，其次碎片解释强度，最后质量误差。没有加载学习模型，也未根据本次结果修改配置。",
        f"- 配置版本：`{protocol['config_version']}`，前体容差 10 ppm，候选上限 10,000，单次搜索结果资源上限 250,000。",
        "- 完整 MRR = 所有分子的 1/k 平均值；真值未进入保留候选集或搜索失败均为 0。这里的完整指最多 10,000 个保留候选，不代表未经截断的无限候选列表。",
        "- MRR@25：排名超过 25 同样计 0。Top-K 是正确分子式进入前 K 名的比例；候选召回率是进入保留候选集的比例。", "", "## 逐轮结果", "",
        "| 种子 | 数量 | 完整 MRR | MRR@25 | Top-1 | Top-5 | Top-10 | Top-25 | 候选召回率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, s in list(summary["rounds"].items()) + [("合并（轮间重复保留）", summary["pooled_rounds"]), ("独立分子去重", summary["unique_molecules"])]:
        lines.append(f"| {label} | {s['molecules']} | {s['mrr_full']:.6f} | {s['mrr25']:.6f} | {s['top1']:.2%} | {s['top5']:.2%} | {s['top10']:.2%} | {s['top25']:.2%} | {s['candidate_recall']:.2%} |")
    s = summary["unique_molecules"]
    lo, hi = summary["unique_mrr_full_bootstrap_95ci"]
    ranks = [r["rank"] for r in rows if r["rank"] is not None]
    lines += ["", "## 诊断与适用范围", "",
        f"独立分子的状态统计：`{json.dumps(s['statuses'])}`。搜索失败保持在分母中。真值被召回时的排名中位数为 {np.median(ranks):.1f}。",
        f"完整 MRR 的结构级 bootstrap 95% 区间：[{lo:.6f}, {hi:.6f}]（固定种子 2026，10,000 次重采样）。这是当前样本的统计不确定性，不涵盖数据分布变化。",
        f"推理阶段墙钟耗时 {summary['elapsed_seconds']:.1f} 秒，4 个工作进程；每个独立分子的平均任务耗时 {s['mean_seconds']:.2f} 秒。",
        f"输入谱数量中位数 {np.median([r['input_spectra'] for r in rows]):.1f}、最大值 {max(r['input_spectra'] for r in rows)}；公开测试每个分子仅 1–9 张谱，两者多谱数量分布不同。", "",
        "| 中性质量（Da） | 独立分子数 | 完整 MRR | MRR@25 | Top-25 | 候选召回率 | 搜索资源失败数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for label, s in summary["mass_strata"].items():
        lines.append(f"| {label} | {s['molecules']} | {s['mrr_full']:.6f} | {s['mrr25']:.6f} | {s['top25']:.2%} | {s['candidate_recall']:.2%} | {s['statuses'].get('resource_limit', 0)} |")
    lines += ["", "本结果针对已通过预处理质控、受支持的中性分子式与加合物。原始数据中因标签冲突、域外元素、带电标签、质量误差或无效输入被排除的记录不在抽样总体内。因此不能将本结果直接当作原始数据全覆盖准确率或竞赛结构预测得分。", "",
        "结论：当前基线候选召回率较高，但真值常排在较后位置，默认 Top-25 输出表现较弱。候选生成能覆盖真值，不等于基线能将真值排在前列。", "", "## 复现", "", "```powershell", ".venv/Scripts/python.exe scripts/sample_baseline.py", ".venv/Scripts/python.exe scripts/report_baseline_sampling.py", "```", "", "sampling.json 保存各轮 ID 与配置；records.jsonl 保存逐分子真值排名、状态与耗时；summary.json 保存汇总数值。复跑复用已完成的逐分子记录。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "verification.json", {"unique_records": len(rows), "rounds": 5, "metrics_recomputed": True, "all_sample_ids_accounted_for": True})
    print("Verified 5 rounds and all individual records; report.md written.")


if __name__ == "__main__":
    main()
