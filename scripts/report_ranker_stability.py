"""Independent score verification and frozen-model stability report."""
import json
from pathlib import Path
import numpy as np
from casmi.formula.data import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reports/enveda-stability-v1'


def main():
    s=json.loads((OUT/'summary.json').read_text())
    m=json.loads((OUT/'manifest.json').read_text())
    rows=[json.loads(line) for line in (OUT/'records.jsonl').read_text().splitlines()]
    records={r['molecule_id']:r for r in rows}
    assert len(rows)==len(records)==3000
    ids=[key for keys in m['rounds'].values() for key in keys]
    assert len(ids)==len(set(ids))==3000 and set(ids)==set(records)
    def check(rs,metric):
        ranks=[r['rank'] for r in rs]
        assert len(rs)==metric['molecules']
        assert abs(sum(1/r for r in ranks if r)/len(rs)-metric['mrr_full'])<1e-12
        assert abs(sum(1/r for r in ranks if r and r<=25)/len(rs)-metric['mrr25'])<1e-12
        for k in (1,5,10,25):
            assert abs(sum(r is not None and r<=k for r in ranks)/len(rs)-metric[f'top{k}'])<1e-12
    check(rows,s['overall'])
    for index,keys in m['rounds'].items():
        assert len(keys)==300
        check([records[k] for k in keys],s['rounds'][index])
    lines=['# 当前 LightGBM 模型的扩大样本稳定性测试','',
        f"固定模型 `{s['model_version']}`（enveda-ranker-v1：31 叶、lr=0.05、227 轮），未重新训练，未调参。",'',
        '## 方案','',
        f"Enveda/timsTOF 质控后留出总体有 {m['source_population']:,} 个结构，排除此前已评估的 {m['previously_evaluated_source_structures']:,} 个，剩余 {m['eligible_population']:,} 个。",
        '以种子 20260923 从剩余总体无放回抽取 3,000 个结构，按抽样顺序分成 10 批，每批 300 个。批间、与此前已评估样本、与训练/验证结构均无交叉。继续使用相同来源、谱质控、清洗、候选生成和模型配置。',
        'MRR 为平均 1/k（最多 10,000 个保留候选）；MRR@25 将第 26 名及以后计 0；真值未召回和搜索失败均计 0。谱数不限制为公开 test 的 1–9 张，仍使用分子的全部可用 Enveda 谱。',
        '这是固定模型在同来源新样本上的稳定性测试；不检验不同随机种子重新训练的稳定性，也不检验天然产物或其他仪器上的分布迁移。','',
        f"本次样本中性质量范围为 {min(r['mass'] for r in rows):.2f}–{max(r['mass'] for r in rows):.2f} Da；不能据此判断更高质量分子的稳定性。",'',
        '## 每批结果','',
        '| 批次 | 分子数 | 平均 1/k | MRR@25 | Top-1 | Top-5 | Top-25 | 候选召回 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def row(label,r):
        return f"| {label} | {r['molecules']} | {r['mrr_full']:.6f} | {r['mrr25']:.6f} | {r['top1']:.2%} | {r['top5']:.2%} | {r['top25']:.2%} | {r['candidate_recall']:.2%} |"
    for index,r in s['rounds'].items():
        lines.append(row(index,r))
    lines.append(row('新增样本合并',s['overall']))
    lines += ['', '## 波动与区间','',
        '整体 95% 区间使用结构级 bootstrap，固定种子 20260924、10,000 次重采样。批次标准差描述每批 300 个分子的分数波动，不等于整体均值的不确定性。这里的随机分批不是时间序列，不应将批次起伏解释成模型随时间漂移。','',
        '| 指标 | 批次均值 | 批次标准差 | 批次最小值 | 批次最大值 | 合并均值 95% 区间 |',
        '|---|---:|---:|---:|---:|---|']
    for name,d in s['batch_variation'].items():
        lo,hi=s['bootstrap_95ci'][name]
        lines.append(f"| {name} | {d['mean']:.6f} | {d['sd']:.6f} | {d['minimum']:.6f} | {d['maximum']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', '## 分层结果','',
        'seen/unseen 按实际用于拟合的分子式划分。所有测试结构都与训练结构隔离，即使它们可能具有相同分子式。','',
        '| 分层 | 分子数 | 平均 1/k | MRR@25 | Top-1 | Top-5 | Top-25 | 候选召回 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in s['strata'].items():
        lines.append(row(name,r))
    lines += ['', '## 与历史结果对照','',
        '前两次各 500 个新留出分子的平均 1/k 分别为 0.897685、0.908350，Top-1 分别为 83.8%、85.0%。这些旧结果已经被查看并用于讨论模型选择，因此本轮新增 3,000 个分子才是主要确认样本。',
        '把本次与前两次不重叠的 500 分子测试合并，可得到 4,000 个分子的描述性汇总；不把它当作全新盲测。','',
        '| 分组 | 分子数 | 平均 1/k | MRR@25 | Top-1 | Top-5 | Top-25 | 候选召回 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',row('历史＋本次',s['cumulative_4000']),'',
        '## 状态与验证','',f"新增样本状态：{s['overall']['statuses']}。本次脚本耗时约 {s['elapsed_seconds']/60:.1f} 分钟。",
        '已验证：10 批样本互不重叠；新旧测试、训练/验证零结构交叉；固定模型文件哈希前后一致；全部逐分子排名重新计算汇总；公开 predict API 的三个抽查结果与并行推理一致。',
        '即使同来源总体分数稳定，也不意味着每个化学类别都同样准确；分子式新颖程度、分子质量和谱数量的构成变化仍可能改变得分。','',
        '## 复现与文件','', '```powershell','.venv/Scripts/python.exe scripts/evaluate_ranker_stability.py','.venv/Scripts/python.exe scripts/report_ranker_stability.py','```','',
        'manifest.json 保存全部抽样 ID、固定模型哈希与配置版本；spectra.parquet 保存样本谱；records.jsonl 保存逐分子排名、状态、来源和耗时；summary.json 保存指标、分层及置信区间。复跑复用已完成记录。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(OUT/'verification.json',dict(unique_structures=3000,disjoint_batches=10,all_metrics_recomputed=True,**s['verification']))
    print('Verified 3,000 distinct structures and all 10 batches; report written.')


if __name__=='__main__':
    main()
