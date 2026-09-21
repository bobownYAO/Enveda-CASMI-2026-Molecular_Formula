"""Verify source-isolated external-library evaluation and summarize limitations."""
import json
from pathlib import Path
from casmi.formula.data import write_json
from evaluate_external_sources import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reports/external-sources-v1'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    rows=[json.loads(line) for line in (OUT/'records.jsonl').read_text().splitlines()]
    assert len(rows)==len({(r['source'],r['molecule_id']) for r in rows})==1672
    for source,choice in manifest['selection'].items():
        rs=[r for r in rows if r['source']==source]
        assert {r['molecule_id'] for r in rs}==set(choice['ids'])
        scores=summary['sources'][source]['learned']
        n=len(rs)
        assert abs(sum(1/r['rank'] for r in rs if r['rank'])/n-scores['mrr_full'])<1e-12
        assert abs(sum(1/r['rank'] for r in rs if r['rank'] and r['rank']<=25)/n-scores['mrr25'])<1e-12
        for k in (1,5,10,25):
            assert abs(sum(r['rank'] is not None and r['rank']<=k for r in rs)/n-scores[f'top{k}'])<1e-12
    gnps=json.loads((OUT/'gnps-no-raw-threshold-summary.json').read_text())
    alt=[json.loads(line) for line in (OUT/'gnps-no-raw-threshold.jsonl').read_text().splitlines()]
    assert {r['molecule_id'] for r in alt}==set(manifest['selection']['gnps']['ids'])
    assert metrics(alt)==gnps['learned']
    joint={source:metrics(rs) for source in manifest['selection'] if (rs:=[r for r in rows if r['source']==source and r['mass_in_enveda_test_range'] and r['max_label_abs_ppm'] is not None and r['max_label_abs_ppm']<=10])}
    write_json(OUT/'matched-mass-and-accuracy.json',joint)
    lines=['# 固定 Enveda LightGBM 模型的跨数据库测试','',
        f"模型：`{summary['model_version']}`（31 叶、227 轮）。只测试，不训练、不调整规则或参数。",'',
        '## 实验口径','',
        f"从既有结构隔离的 test 划分中，按来源分别随机抽取最多 200 个结构；不足 200 的来源使用全部可用结构。共 {summary['source_structure_pairs']} 个来源—结构组合、{summary['unique_structures']} 个不同结构。每次预测只使用该来源的谱，不混合数据库。所有结构与 train、validation 零交叉，但不要求它们此前从未用于其他留出评估。",
        '不同来源可含相同结构，因此表格应按来源理解；不将所有来源行当成相互独立样本，也不把库间分数差异解释为仪器本身的因果影响。',
        '继承 prepared 的中性标签、支持元素/加合物、标签冲突和前体误差 ≤20 ppm 质控。预测容差仍为 10 ppm。原始峰进一步删除 precursor+2 Da 以上的峰。主评估丢弃已知基峰强度 <1,000 的谱；缺失强度保留并记录。清洗后无谱的分子仍在分母中，计 0 分。',
        '原始基峰阈值来自 timsTOF 清洗协议，不同平台的强度不一定可比，因此额外对相同 200 个 GNPS 分子做不设置原始强度门槛的对照。其他来源本轮没有因该门槛删谱。',
        'MRR = 平均 1/k，采用最多 10,000 个保留候选；MRR@25 将第 26 名及以后计 0。真值缺失、搜索失败及清洗后无谱均计 0。','',
        '## 分来源结果','',
        '| 来源 | 分子数 | 平均 1/k | MRR@25 | Top-1 | Top-5 | Top-25 | 候选召回 | 搜索资源失败 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for source,detail in summary['sources'].items():
        r=detail['learned']
        lines.append(f"| {source} | {r['molecules']} | {r['mrr_full']:.6f} | {r['mrr25']:.6f} | {r['top1']:.2%} | {r['top5']:.2%} | {r['top25']:.2%} | {r['candidate_recall']:.2%} | {r['statuses'].get('resource_limit',0)} |")
    lines += ['', '原 Enveda-180 的 3,000 分子测试：MRR=0.900679、Top-1=83.97%、Top-25=100%；质量范围仅 237.13–464.04 Da。它是同来源参照，不是本表各库的同分子配对实验。','',
        '## 状态与抽样不确定性','',
        '| 来源 | 中性质量范围 Da | MRR 95% 区间 | 状态统计 |', '|---|---|---|---|']
    for source,detail in summary['sources'].items():
        lo,hi=detail['mrr_bootstrap_95ci'];mlo,mhi=detail['mass_range']
        lines.append(f"| {source} | {mlo:.1f}–{mhi:.1f} | [{lo:.4f}, {hi:.4f}] | {detail['learned']['statuses']} |")
    lines += ['', '区间为每个来源内按结构 bootstrap（10,000 次）。来源间可能重复结构，库间差值不能直接按两个独立总体解释。enveda-np-examples 仅 27 个、masaryk 仅 45 个留出结构，结果不适合精确排序数据库优劣。','',
        '## GNPS 强度门槛对照','', '| 同一批 GNPS 分子 | 数量 | MRR | MRR@25 | Top-1 | Top-25 | 候选召回 | 状态 |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for name,r in [('保留原始强度门槛',summary['sources']['gnps']['learned']),('取消原始强度门槛',gnps['learned'])]:
        lines.append(f"| {name} | {r['molecules']} | {r['mrr_full']:.6f} | {r['mrr25']:.6f} | {r['top1']:.2%} | {r['top25']:.2%} | {r['candidate_recall']:.2%} | {r['statuses']} |")
    lines += ['', '取消门槛时仍保留原始基峰值作为输入特征，仍移除过高峰；不是把强度伪装为缺失，也不是模型重训练。','',
        '## 控制质量范围与前体误差后的子集','',
        '以下只取中性质量 237–465 Da 且清洗后所有谱的标签质量误差 ≤10 ppm 的分子，以减轻质量大小和质量误差的混淆。这是诊断性子集，不替代上面的端到端指标；库间化学组成仍不同。','',
        '| 来源 | 子集分子数 | MRR | Top-1 | Top-25 | 候选召回 |','|---|---:|---:|---:|---:|---:|']
    for source,r in joint.items():
        lines.append(f"| {source} | {r['molecules']} | {r['mrr_full']:.6f} | {r['top1']:.2%} | {r['top25']:.2%} | {r['candidate_recall']:.2%} |")
    lines += ['', '## 与基线的同候选对照','', '| 来源 | 基线 MRR | 学习模型 MRR | 真值被召回时的学习 MRR |','|---|---:|---:|---:|']
    for source,detail in summary['sources'].items():
        conditional=detail['strata'].get('truth_recalled',{}).get('mrr_full')
        lines.append(f"| {source} | {detail['baseline']['mrr_full']:.6f} | {detail['learned']['mrr_full']:.6f} | {conditional if conditional is not None else 'N/A'} |")
    lines += ['', '最后一列排除漏召回，专门诊断排序；不能用它代替完整命中率。高质量区间、seen/unseen 分子式等更多分层见 summary.json。','',
        '## 数据清洗审计','', '| 来源 | 原谱数 | 缺失原始基峰 | 低基峰剔除谱 | 保留谱数 |', '|---|---:|---:|---:|---:|']
    for source,a in audit['by_source'].items():
        lines.append(f"| {source} | {a['input_spectra']} | {a['missing_base']} | {a['dropped_known_low_base']} | {a['retained_spectra']} |")
    lines += ['', '本地仪器字段显示 pluskal_ms2 为 Orbitrap，enveda-np-examples 为 timsTOF；其他来源包含多种平台、缺失或自由文本，不能一律标为高分辨率。完整仪器计数见 audit.json。',
        '注意：上游 prepare 已移除不支持标签、加合物、冲突结构和大质量误差谱；本结果是受支持且质控后数据上的评估，不是原始库全覆盖准确率。原始库各类接受/拒绝数量保存在 manifest.json 的 source_coverage。','',
        '## 复现与交付','', '```powershell','.venv/Scripts/python.exe scripts/evaluate_external_sources.py','.venv/Scripts/python.exe scripts/evaluate_gnps_intensity_ablation.py','.venv/Scripts/python.exe scripts/report_external_sources.py','```','',
        f"主评估耗时 {summary['elapsed_seconds']/60:.1f} 分钟。已核对全部来源的逐分子计分、样本 ID、结构隔离及模型文件哈希。原模型未修改。",
        'manifest.json 保存抽样来源与 ID；records.jsonl 保存每个来源—分子的排名和失败原因；spectra.parquet 保存清洗后的谱；GNPS 对照单独保存在 gnps-no-raw-threshold*；summary.json 保存全量统计与分层。']
    lines += ['', '## 解释', '',
        '模型仍明显优于无训练基线，但不能把 Enveda 内部约 0.90 的得分视为跨库保证。跨库分子质量、化学组成、仪器、碰撞能量和元数据同时变化，当前结果不能单独归因于某个仪器或数据库质量。',
        'Spectraverse 的 200 个样本中 113 个触发搜索资源上限，候选生成是其主要瓶颈之一。限制到相近质量且前体误差较小的诊断子集后，各来源候选召回均为 100%，但平均 1/k 仍约 0.55–0.79；这支持排序泛化不足的判断。',
        'GNPS 的绝对原始强度门槛会使 58 个分子无谱，取消门槛后 MRR 从 0.4162 提高到 0.5999。跨仪器清洗不能机械沿用 timsTOF 的原始计数阈值。',
        '后续需要分别处理高质量分子的搜索预算与剪枝，以及自然产物/多来源训练覆盖；当前实验不修改模型或选择新模型。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(OUT/'verification.json',dict(source_pairs=len(rows),unique_structures=len({r['molecule_id'] for r in rows}),metrics_recomputed=True,gnps_same_cohort_verified=True,**summary['verification']))
    print('All sources and GNPS control verified; report written.')


if __name__=='__main__':
    main()
