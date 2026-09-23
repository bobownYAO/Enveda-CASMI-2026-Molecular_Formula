"""Recompute experiment summaries and verify serialized-model API ranks."""

import json

import polars as pl
from casmi.formula import FormulaPredictor
from casmi.formula.data import write_json
from train_multisource_ranker import DATA, MODEL, OLD, OUT, weighted


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    selection = json.loads((DATA / "selection.json").read_text())
    records = json.loads((OUT / "test-records.json").read_text())
    training = json.loads((MODEL / "training_report.json").read_text())
    priors = selection["priors"]
    for name, field in [
        ("multisource", "rank"),
        ("enveda", "old_rank"),
        ("baseline", "baseline_rank"),
    ]:
        assert weighted(records, priors, field) == summary[name]
    assert len(records) == 1500 == len({r["molecule_id"] for r in records})
    sets = {
        s: set().union(*map(set, sources.values()))
        for s, sources in selection["selected"].items()
    }
    assert not sets["train"] & sets["validation"] and not sets["test"] & (
        sets["train"] | sets["validation"]
    )
    frame = pl.read_parquet(DATA / "test/part-000.parquet")
    predictor = FormulaPredictor.load(MODEL)
    assert predictor.config.version == FormulaPredictor.load(OLD).config.version
    checks = []
    for source in ["enveda-180", "gnps", "pluskal_ms2"]:
        r = min(
            (
                r
                for r in records
                if r["source"] == source and r["candidate_count"] and r["mass"] < 400
            ),
            key=lambda r: r["spectra"],
        )
        result = predictor.predict(
            frame.filter(pl.col("molecule_id") == r["molecule_id"]).to_dicts(),
            molecule_id=r["molecule_id"],
            top_k=10000,
        )
        rank = next(
            (c.rank for c in result.candidates if c.formula == r["formula"]), None
        )
        assert rank == r["rank"]
        checks.append(dict(molecule_id=r["molecule_id"], rank=rank))
    write_json(
        OUT / "verification.json",
        dict(
            metrics_recomputed=True,
            structure_overlap=0,
            public_api=checks,
            **{
                k: v
                for k, v in summary["verification"].items()
                if k != "structure_overlap"
            },
        ),
    )
    old, new = summary["enveda"]["weighted"], summary["multisource"]["weighted"]
    ci = summary["paired_delta_95ci"]["weighted_total"]
    resource_failures = sum(r.get("status") == "resource_limit" for r in records)
    weighted_resource_failures = sum(
        priors[s]
        * d["resource_failures"]
        / summary["multisource"]["by_source"][s]["molecules"]
        for s, d in summary["diagnostic_strata"].items()
    )
    lines = [
        "# 多来源分布加权训练对照",
        "",
        f"加权平均 1/k：原 Enveda 模型 **{old['mrr_full']:.6f}**，多来源模型 **{new['mrr_full']:.6f}**，差值 **{new['mrr_full'] - old['mrr_full']:+.6f}**，配对 95% 区间 [{ci[0]:+.6f}, {ci[1]:+.6f}]。",
        f"加权 Top-1：{old['top1']:.2%} → {new['top1']:.2%}；候选召回率两者均为 {new['candidate_recall']:.2%}。",
        "",
        "本轮 10 个非 Enveda-180 来源的平均 1/k 均提升；Enveda-180 从 0.9054 小幅回落至 0.8972，配对差值区间包含 0。结果支持原模型的训练分布覆盖不足是外库低分的重要原因之一。",
        f"测试中 {sum(r['rank'] is None for r in records)} 组未召回真值，其中 {resource_failures} 组触发搜索资源上限；按来源权重分别占 {1 - new['candidate_recall']:.2%} 和 {weighted_resource_failures:.2%}。这一部分无法通过重排修复。",
        "例如 Spectraverse 只有 49.3% 的候选召回率，其中 36/75 组触发搜索上限；在已召回子集上，新模型平均 1/k 为 0.8506。因此其端到端低分不能直接归因于谱质量差，应优先检查大分子候选搜索覆盖。",
        "",
        "## 实验口径",
        "",
        "- 固定原有候选生成、10 ppm 前体容差、元素边界和 74 个特征；只改变训练来源与查询权重。两个模型在完全相同的候选集上打分。",
        "- 11 个来源，3,000 个训练组、550 个验证组、1,500 个测试组。分组单位为来源—结构，同一结构跨来源始终属于相同划分；同结构同来源的所有谱一起输入。",
        "- 来源权重仅使用清洗后训练划分中的独立来源—结构数量；不按谱数量加权，测试标签不影响权重。训练/验证/测试最低每来源 30/15/75 组，不足则全取；采样偏差通过来源权重校正。该总分代表可用训练库的来源分布，并非竞赛自然产物分布。",
        "- 训练每个查询的权重与来源占比/该来源抽样数成正比。同查询所有候选使用相同权重；无真值的组不进入拟合，但始终计入验证与测试分母。",
        "- 沿用既有标签冲突、元素/加合物及 20 ppm 质量误差清洗；峰高于前体+2 Da 则移除。不使用跨来源统一原始基峰强度阈值，缺失元数据保留。结果针对清洗后可用范围，不能等同整个原始库覆盖率。",
        "- 与原模型相同：31 叶、学习率 0.05、最多 1,000 轮、早停 50。仅验证集加权 MRR@25 决定轮数；未用测试集调参。原模型不覆盖。",
        f"- 实际可拟合训练组 {training['actual_groups']}/3000；最佳迭代 {training['best_iteration']}；测试包含 {summary['verification']['unique_test_structures']} 个独立结构。测试可能与历史评测重叠，但与训练、验证结构零交叉。",
        "- 主得分为平均 1/k，任何候选遗漏、无候选或资源失败均记 0；额外报告 MRR@25。配对区间使用 3,000 次按结构聚类的 Poisson bootstrap，跨来源同一结构共用权重，固定来源占比。区间描述测试抽样不确定性，不包含重复训练的波动。",
        "",
        "## 各来源测试结果",
        "",
        "| 来源 | 总体权重 | 训练抽样/可拟合 | 测试数 | 原模型 1/k | 多来源 1/k | 差值 | 差值 95% CI | 原 Top-1 | 新 Top-1 | 候选召回 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for s in priors:
        a, b = summary["enveda"]["by_source"][s], summary["multisource"]["by_source"][s]
        lo, hi = summary["paired_delta_95ci"][s]
        lines.append(
            f"| {s} | {priors[s]:.2%} | {len(selection['selected']['train'][s])}/{training['actual_by_source'].get(s, 0)} | {b['molecules']} | {a['mrr_full']:.4f} | {b['mrr_full']:.4f} | {b['mrr_full'] - a['mrr_full']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {a['top1']:.1%} | {b['top1']:.1%} | {b['candidate_recall']:.1%} |"
        )
    lines += [
        "",
        "## 总体指标",
        "",
        "| 指标 | 原 Enveda 模型 | 多来源模型 | 化学基线 |",
        "|---|---:|---:|---:|",
    ]
    for metric in [
        "mrr_full",
        "mrr25",
        "top1",
        "top5",
        "top10",
        "top25",
        "candidate_recall",
    ]:
        lines.append(
            "| "
            + metric
            + " | "
            + " | ".join(
                f"{summary[m]['weighted'][metric]:.6f}"
                for m in ["enveda", "multisource", "baseline"]
            )
            + " |"
        )
    lines += [
        "",
        "未加权样本均值受到小来源过采样影响，不作为总体结论：",
        "",
        f"原模型 {summary['enveda']['sample_micro']['mrr_full']:.6f}，多来源模型 {summary['multisource']['sample_micro']['mrr_full']:.6f}。",
        "",
        "## 排序与候选问题的拆分",
        "",
        "| 来源 | 仅已召回样本：原/新 1/k | 237–465 Da 且全部谱≤10 ppm：样本数 | 匹配子集：原/新 1/k | 资源失败数 | 全部谱超10 ppm数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s, d in summary["diagnostic_strata"].items():
        a, b = d["recalled"], d["matched_mass_accuracy"]
        lines.append(
            f"| {s} | {a['old_rank']['mrr_full']:.4f}/{a['rank']['mrr_full']:.4f} | {b['rank']['molecules']} | {b['old_rank']['mrr_full']:.4f}/{b['rank']['mrr_full']:.4f} | {d['resource_failures']} | {d['all_spectra_outside_10ppm']} |"
        )
    lines += [
        "",
        "候选固定后，新模型的配对提升是排序训练分布影响的证据。未召回的候选不可能通过 LightGBM 排序修复。资源失败属于当前算法搜索边界，不应归咎于谱质量。超 10 ppm、缺失元数据等只是质量/采集差异线索，不能凭来源得分断言数据质量优劣。",
        "",
        "该实验保持训练抽样预算相同，但实际可拟合数、来源构成和分子式覆盖随之改变；不构成对仪器质量的严格因果实验。小来源区间较宽，不能将单次轻微涨跌解释为稳定差异。",
        "",
        "## 复现与产物",
        "",
        "```powershell",
        ".venv/Scripts/python.exe scripts/train_multisource_ranker.py",
        ".venv/Scripts/python.exe scripts/report_multisource_ranker.py",
        "```",
        "",
        f"- 新模型：`{MODEL}`",
        f"- 对照模型：`{OLD}`",
        f"- 抽样、权重、清洗规则：`{DATA / 'selection.json'}`",
        f"- 逐组预测：`{OUT / 'test-records.json'}`",
        f"- 指标、配对区间：`{OUT / 'summary.json'}`",
        "- 验证了原模型哈希未变、结构划分零交叉、指标独立重算及新模型加载后的公开 API 排名与缓存一致。",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("REPORT_VERIFIED", OUT / "report.md")


if __name__ == "__main__":
    main()
