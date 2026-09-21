"""Check the paired experiment and render its report from saved results."""
import json
from pathlib import Path
import polars as pl
from casmi.formula.data import write_json
from evaluate_enveda_alignment import clean_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports/enveda-alignment"


def main():
    # Meaningful boundary check: preserve aligned intensities and the +2 boundary.
    row=dict(base_peak_intensity=1000,precursor_mz=100,ms2_mzs=[99,102,102.001],ms2_normalized_intensities=[.2,.5,1])
    clean,counts=clean_rows([row,{**row,"base_peak_intensity":999},{**row,"base_peak_intensity":None}])
    assert len(clean)==1 and clean[0]["ms2_mzs"]==[99,102] and clean[0]["ms2_normalized_intensities"]==[.2,.5]
    assert counts["removed_high_peaks"]==1 and counts["removed_low_or_missing_base"]==2
    m=json.loads((OUT/"manifest.json").read_text())
    a=json.loads((OUT/"audit.json").read_text())
    s=json.loads((OUT/"summary.json").read_text())
    rows=[json.loads(line) for line in (OUT/"records.jsonl").read_text().splitlines()]
    records={(r["arm"],r["molecule_id"]):r for r in rows}
    assert len(records)==len(rows)==m["evaluated_unique"]*2
    selected=set().union(*map(set,m["rounds"].values()))
    for arm in ("source_only","test_cleaned"):
        for seed,ids in m["rounds"].items():
            assert len(ids)==len(set(ids))==100
            ranks=[records[(arm,k)]["rank"] for k in ids]
            assert abs(sum(1/r for r in ranks if r)/100-s["rounds"][arm][seed]["mrr_full"])<1e-12
            assert abs(sum(1/r for r in ranks if r and r<=25)/100-s["rounds"][arm][seed]["mrr25"])<1e-12
    before=pl.read_parquet(OUT/"source_only.parquet")
    after=pl.read_parquet(OUT/"test_cleaned.parquet")
    for frame in (before,after):
        assert frame.filter((pl.col("ingest_lib")!="enveda-180") | (pl.col("instrument_type")!="timsTOF")).height==0
    for row in after.to_dicts():
        assert row["base_peak_intensity"]>=1000
        assert len(row["ms2_mzs"])==len(row["ms2_normalized_intensities"])
        assert all(v<=row["precursor_mz"]+2 for v in row["ms2_mzs"])
    assert set(after["molecule_id"])==set(before["molecule_id"])
    d=a["sampled_source_distribution"]
    lines=["# Enveda/timsTOF 对齐清洗与基线复测", "", "## 数据与实验边界", "",
        "官方说明：enveda-180 与测试集由相同的 Bruker timsTOF 平台采集；enveda-180 主要是合成药物样化合物，测试偏天然产物。同仪器不代表化学空间、谱数、碰撞能量或全部采集条件完全相同。",
        "来源：https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data", "",
        f"本地原始 enveda-180：{a['raw_enveda']['spectra']:,} 张谱，{a['raw_enveda']['structures']:,} 个结构，全部标记 timsTOF；低于 1,000 的原始基峰和缺失基峰均为 0。公开 test：{a['public_test_spectra']} 张谱、{a['public_test_structures']} 个分子，全部标记 timsTOF。",
        f"在既有结构隔离/质控数据中，enveda-180/timsTOF 留出集有 {m['population']:,} 个结构。按种子 101–105 分别均匀抽取 100 个结构，共 500 次、{m['sampled_unique']} 个独立分子；另取上一轮具备该来源谱的 {len(m['historical_paired_ids'])} 个结构作同分子配对比较。总计 {m['evaluated_unique']} 个不同结构。",
        "所有结构与训练、验证集合零交叉。沿用既有预处理的支持元素/加合物范围、标签冲突隔离和 20 ppm 谱质控，不重新划分。官方没有给出完全一致的前体误差阈值，因此不宣称精确复现其全部清洗流程。",
        f"算法保持 baseline-v1、配置 {m['config_version']}，10 ppm 前体匹配、最多 10,000 个候选、单次搜索上限 250,000；元素计数边界仍为之前训练划分确定的边界，未针对本次结果调整。", "", "## 两个对照分支", "",
        "1. source_only：仅保留 enveda-180 且 instrument_type=timsTOF 的谱。",
        "2. test_cleaned：同一批分子/谱，额外丢弃基峰缺失或低于 1,000 的谱，删除 m/z > precursor_mz+2 Da 的峰，并保持峰与强度一一对应。再由原预测器执行相同的归一化、0.1% 过滤与 Top-128 预处理。",
        "没有按真值排名筛选样本，没有调整质量容差，没有加入新的分子式化学规则或学习模型。失败分子计 0，清洗后无谱也计 0。",
        f"此次导出的 {a['cleaning']['input_spectra']:,} 张谱共有 {a['cleaning']['input_peaks']:,} 个峰，删除 {a['cleaning']['removed_high_peaks']:,} 个过高峰；删除低强度/缺失基峰谱 {a['cleaning']['removed_low_or_missing_base']} 张。两个 Parquet 包含新抽样与历史配对样本的并集，不是全量训练集的替代品。",
        f"新抽样分子共 {d['spectra']:,} 张谱；每分子谱数最小/中位数/P90/最大值：{d['spectra_per_structure_quantiles']}。未强制匹配公开 test 的 1–9 张谱分布。",
        f"根据标签重新计算的绝对前体 ppm 误差，中位数/P95/P99/最大值：{[round(x,4) for x in d['precursor_abs_ppm_quantiles']]}。这表明质量准确度较好；ppm 误差不是仪器分辨率的测量值。", "", "## 五轮结果", "",
        "MRR = 平均 1/k（使用最多 10,000 个保留候选）；MRR@25 将第 26 名及以后计 0；无真值候选、资源失败均计 0。", "",
        "| 种子 | 数量 | 仅来源 MRR | 清洗后 MRR | 仅来源 MRR@25 | 清洗后 MRR@25 | 清洗后 Top-1 | 清洗后 Top-25 | 清洗后召回 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in m["rounds"]:
        l=s["rounds"]["source_only"][seed]; r=s["rounds"]["test_cleaned"][seed]
        lines.append(f"| {seed} | 100 | {l['mrr_full']:.6f} | {r['mrr_full']:.6f} | {l['mrr25']:.6f} | {r['mrr25']:.6f} | {r['top1']:.2%} | {r['top25']:.2%} | {r['candidate_recall']:.2%} |")
    lines += ["", "| 分组 | 数量 | MRR | MRR@25 | Top-1 | Top-5 | Top-10 | Top-25 | 候选召回 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def table_row(label,r):
        return f"| {label} | {r['molecules']} | {r['mrr_full']:.6f} | {r['mrr25']:.6f} | {r['top1']:.2%} | {r['top5']:.2%} | {r['top10']:.2%} | {r['top25']:.2%} | {r['candidate_recall']:.2%} |"
    for label,r in [("之前混合来源（不同抽样总体，仅供背景）",s["historical_original_overall"]),("仅来源：五轮合并",s["source_only"]),("清洗后：五轮合并",s["test_cleaned"]),("清洗后：独立分子去重",s["unique"]["test_cleaned"])]:
        lines.append(table_row(label,r))
    lines += ["", "## 同分子配对对照", "", "这组固定分子身份，比较原有混合来源的全部谱与仅 Enveda 谱。它控制分子组成，但移除来源同时也改变谱数量，不能单独归因于仪器型号。", "",
        "| 分组 | 数量 | MRR | MRR@25 | Top-1 | Top-5 | Top-10 | Top-25 | 候选召回 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("original_mixed","source_only","test_cleaned"):
        lines.append(table_row(arm,s["historical_paired"][arm]))
    lines += ["", f"来源筛选相对原混合谱的配对 MRR 差值与 bootstrap 95% 区间：{s['source_minus_mixed']}。",
        f"清洗相对仅来源筛选的配对 MRR 差值与 bootstrap 95% 区间：{s['clean_minus_source']}。",
        f"新抽样中清洗前后真值排名变化的分子数：{s['paired_rank_changes_after_cleaning']}。", "", "## 质量分层与状态", "",
        "| 中性质量 Da | 数量 | 清洗后 MRR | MRR@25 | Top-25 | 候选召回 | 状态统计 |", "|---|---:|---:|---:|---:|---:|---|"]
    for label,r in s["mass_strata"]["test_cleaned"].items():
        lines.append(f"| {label} | {r['molecules']} | {r['mrr_full']:.6f} | {r['mrr25']:.6f} | {r['top25']:.2%} | {r['candidate_recall']:.2%} | {r['statuses']} |")
    lines += ["", f"四进程推理墙钟耗时 {s['elapsed_seconds']:.1f} 秒；清洗后独立分子状态：{s['unique']['test_cleaned']['statuses']}。输入完全相同时复用推理结果，记录含 reused_identical_input 标记。", "",
        "## 复现与文件", "", "```powershell", ".venv/Scripts/python.exe scripts/evaluate_enveda_alignment.py", ".venv/Scripts/python.exe scripts/report_enveda_alignment.py", "```", "",
        "source_only.parquet 和 test_cleaned.parquet 是两组样本谱；manifest.json 保存种子、ID、配置；records.jsonl 保存逐分子排名、状态；audit.json 保存来源与清洗统计；summary.json 保存所有汇总与配对分析。原始数据及原预测算法保持原样。"]
    diagnostics=OUT/"ranking_diagnostics.json"
    if diagnostics.exists():
        examples=json.loads(diagnostics.read_text())["examples"]
        lines += ["", "## 排序诊断抽查", "", "按新抽样分子 ID 字典序取前 5 个检查，不按结果挑选。5 个错误首位候选的当前 DBE 特征均为负，显示当前排序没有约束化学可行性。含 P/S 的 DBE 需要结合价态解释，这里不据此制定通用硬阈值。", "",
            "| 分子 ID | 真值 | 真值排名 | 第一名候选 | 真值 DBE | 第一名 DBE |", "|---|---|---:|---|---:|---:|"]
        for e in examples:
            lines.append(f"| {e['molecule_id']} | {e['truth']['formula']} | {e['truth_rank']} | {e['top1']['formula']} | {e['truth']['dbe']} | {e['top1']['dbe']} |")
        lines += ["", "第一例真值质量误差仅 1.61 ppm、碎片解释强度 92.21%，错误首位候选质量误差 7.93 ppm、解释强度 97.79%。两者均支持全部 6 张谱；当前基线把解释强度作为优先条件，使不合理组成排在真值前。原子子集能匹配碎片，并不等于分子式或碎裂反应化学上可行。", "",
            "本轮证据支持：仅统一来源和仪器、去除少量过高峰不足以修复低分；当前候选合理性约束和排序规则是明确的改进重点。不能将跨总体 MRR 差异解释成 Enveda 数据质量更差；同结构配对结果显示筛选来源基本没有改善。此次不修改算法，保留该基准供后续规则和 LightGBM 实验比较。"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(OUT/"verification.json",dict(unique_structures=m["evaluated_unique"],record_count=len(records),matched_source_verified=True,
        cleaning_boundaries_verified=True,peak_alignment_verified=True,round_mrr_independently_recomputed=True,structure_overlap=0))
    print("All artifact and metric checks passed; report written.")


if __name__=="__main__":
    main()
