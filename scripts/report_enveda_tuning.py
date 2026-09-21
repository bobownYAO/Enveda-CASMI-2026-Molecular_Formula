"""Recompute parameter-search results and produce a reviewable report."""
import json
from pathlib import Path
from casmi.formula import FormulaPredictor
from casmi.formula.data import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/reports/enveda-tuning-v1"


def main():
    trials=json.loads((OUT/"trials.json").read_text())
    best=json.loads((OUT/"selected.json").read_text())
    result=json.loads((OUT/"new-test-summary.json").read_text())
    selection=json.loads((ROOT/"artifacts/prepared-enveda-tuning-v1/selection.json").read_text())
    assert len(trials)==8
    expected=sorted(trials,key=lambda r:(-r['validation']['mrr25'],-r['validation']['top1'],r['best_iteration'],r['name']))[0]
    assert expected['name']==best['name']
    for name in ('original','tuned'):
        records=json.loads((OUT/f'new-test-{name}.json').read_text())
        assert len(records)==len({r['molecule_id'] for r in records})==500
        assert {r['molecule_id'] for r in records}==set(selection['ids'])
        ranks=[r['rank'] for r in records]
        assert abs(sum(1/r for r in ranks if r)/500-result[name]['mrr_full'])<1e-12
        assert abs(sum(1/r for r in ranks if r and r<=25)/500-result[name]['mrr25'])<1e-12
        for k in (1,5,10,25):
            assert abs(sum(r is not None and r<=k for r in ranks)/500-result[name][f'top{k}'])<1e-12
    predictor=FormulaPredictor.load(ROOT/'artifacts/formula/enveda-ranker-tuned-v1')
    lines=['# Enveda LightGBM 参数对照实验','',
        '固定 3,000 个训练结构、500 个验证结构和既有候选特征，只调整训练参数。原模型作为对照，另试 7 组参数。未改动元素范围、质量容差、候选生成、清洗流程或训练负例。',
        '每组使用种子 42、早停 50 轮；最多 1,000 轮，低学习率组最多 1,600 轮。按验证 MRR@25 选择；并列时依次比较 Top-1、更少树、名称。测试结果不参与选择。',
        '原模型验证分数已重新计算并与旧报告一致；新排名计算方法与原稳定排序方法在全部 500 个验证分子上逐一对照一致。不同 min_data_in_leaf 试验使用独立 Dataset，避免预过滤特征在试验间残留。',
        '[LightGBM 参数定义与 Dataset 预过滤说明](https://lightgbm.readthedocs.io/en/stable/Parameters.html#feature_pre_filter)','',
        '## 验证集全部结果','',
        '| 组合 | 主要改动 | 最佳轮数 | MRR@25 | Top-1 | Top-5 | Top-25 |',
        '|---|---|---:|---:|---:|---:|---:|']
    labels=dict(original31='原模型：31 叶、lr=0.05',leaves15='15 叶',leaves63='63 叶',slow31='lr=0.025',
        regularized31='叶最小样本 50、L2=1',regularized63='63 叶、叶最小样本 50、L2=1',
        subsample31='特征/行采样 0.8、L2=1',top10rank31='LambdaRank 截断位置 10')
    for t in trials:
        v=t['validation']
        lines.append(f"| {t['name']} | {labels[t['name']]} | {t['best_iteration']} | {v['mrr25']:.6f} | {v['top1']:.2%} | {v['top5']:.2%} | {v['top25']:.2%} |")
    lines += ['',f"验证选出的组合为 **{best['name']}**，模型版本 `{predictor.model_version}`。",'',
        '## 全新留出集','',
        '使用种子 20260921 从既有 Enveda/timsTOF 留出集合中均匀抽取 500 个结构，排除此前所有已评估的 Enveda 对齐样本和上一轮新测试样本，且与训练、验证结构零交叉。来源和清洗规则不变。只在选定组合后，对原模型与选中模型进行配对评估。',
        'MRR 为平均 1/k；完整列表最多 10,000 个候选。MRR@25 将第 26 名及以后计 0。候选未召回和搜索失败均计 0。','',
        '| 模型 | 完整 MRR | MRR@25 | Top-1 | Top-5 | Top-10 | Top-25 | 候选召回 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('original','tuned'):
        r=result[name]
        lines.append(f"| {name} | {r['mrr_full']:.6f} | {r['mrr25']:.6f} | {r['top1']:.2%} | {r['top5']:.2%} | {r['top10']:.2%} | {r['top25']:.2%} | {r['candidate_recall']:.2%} |")
    lines += ['', '## 配对变化与不确定性','',
        '95% 区间采用同结构配对 bootstrap（10,000 次，固定种子 20260922）；它仅反映当前样本的抽样不确定性，不涵盖仪器或化学空间迁移。','']
    for name in ('mrr25','top1'):
        d=result[name+'_paired_difference']
        lines.append(f"- {name}：新减旧 = {d['mean']:.6f}，95% 区间 [{d['bootstrap_95ci'][0]:.6f}, {d['bootstrap_95ci'][1]:.6f}]；改善 {d['improved']} 个，变差 {d['worsened']} 个。")
    diff=result['mrr25_paired_difference']
    if diff['bootstrap_95ci'][0]>0:
        interpretation='新留出集支持此次参数调整带来 MRR 改善，但仍应在更大样本及天然产物数据上确认适用范围。'
    elif diff['mean']>0:
        interpretation='新留出集的 MRR 点估计有所提升，但配对区间包含 0，尚不足以确认稳定收益；不要把验证集提升视作已证实的泛化提升。'
    else:
        interpretation='验证集选中组合没有在新留出集提高 MRR，本次调参未证实泛化收益。原模型仍是更稳妥的当前参照。'
    lines += ['',interpretation,'',
        '没有根据新测试结果追加搜索或改选其他组合。所有试验模型、参数和结果均保留；原模型未覆盖。', '', '## 最佳组合参数','', '```json',json.dumps(best['parameters'],indent=2),'```','',
        '## 使用与复现','', '```python','from casmi.formula import FormulaPredictor','predictor = FormulaPredictor.load("artifacts/formula/enveda-ranker-tuned-v1")','```','',
        '```powershell','.venv/Scripts/python.exe scripts/tune_enveda_ranker.py','.venv/Scripts/python.exe scripts/report_enveda_tuning.py','```','',
        '复跑复用已完成试验；protocol.json 固定搜索方案，trials.json 记录每组结果，selected.json 记录验证选择，new-test-*.json 记录逐分子排名和配对汇总。新模型公开 API 与缓存排名已抽查三个新留出分子并一致。',
        '本实验仍是 Enveda 合成药物样化合物上的 3,000 分子训练；不是新增数据训练，也不是竞赛结构预测或天然产物泛化测试。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(OUT/'verification.json',dict(trials=8,validation_selection_verified=True,new_test_metrics_recomputed=True,model_loaded=True,api_checks=result['api_checks'],test_structure_overlap=0))
    print('All 8 trials, selection and new-test metrics verified; report written.')


if __name__=='__main__':
    main()
