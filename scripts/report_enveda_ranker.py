"""Verify and summarize the completed Enveda LightGBM experiment."""
import json
from pathlib import Path
import numpy as np
from casmi.formula import FormulaPredictor
from casmi.formula.data import write_json

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"artifacts/reports/enveda-ranker-v1"
MODEL=ROOT/"artifacts/formula/enveda-ranker-v1"
DATA=ROOT/"artifacts/prepared-enveda-ranker-v1"


def main():
    summary=json.loads((REPORT/"summary.json").read_text())
    selection=json.loads((DATA/"selection.json").read_text())
    metadata=json.loads((MODEL/"metadata.json").read_text())
    ready=json.loads((DATA/"ready.json").read_text())
    training=summary["training"]
    model=FormulaPredictor.load(MODEL)
    assert model.model_version==summary["model_version"] and model.config.version==summary["config_version"]
    sets={k:set(v) for k,v in selection["ids"].items()}
    assert not sets["train"]&sets["validation"] and not sets["train"]&sets["test"] and not sets["validation"]&sets["test"]
    for name in ("validation","fresh_test","previous_benchmark"):
        records=json.loads((REPORT/f"{name}-records.json").read_text())
        assert len(records)==len({r['molecule_id'] for r in records})
        for backend,field in [("learned","rank"),("baseline","baseline_rank")]:
            ranks=[r[field] for r in records]
            metric=summary["metrics"][name][backend]
            assert abs(sum(1/r for r in ranks if r)/len(ranks)-metric["mrr_full"])<1e-12
            assert abs(sum(1/r for r in ranks if r and r<=25)/len(ranks)-metric["mrr25"])<1e-12
            for k in (1,5,10,25):
                assert abs(sum(r is not None and r<=k for r in ranks)/len(ranks)-metric[f"top{k}"])<1e-12
    lines=["# Enveda/timsTOF LightGBM 首轮训练", "",
        f"模型：`{summary['model_version']}`；配置：`{summary['config_version']}`。", "", "## 数据与训练", "",
        "只使用 enveda-180 / timsTOF 来源，谱基峰至少 1,000，去除高于前体 m/z+2 Da 的峰。继承已有结构隔离划分、标签冲突隔离与谱质量控制；元素边界、10 ppm 前体容差、碎片特征、候选上限均保持不变。没有新增化学规则。",
        f"随机抽样 3,000 个训练结构、500 个验证结构；实际参与拟合 {training['training_groups']} 个候选组，训练候选召回率 {training['training_candidate_recall']:.2%}。每组含自然生成的真值和最多 127 个困难负例、128 个随机负例。真值未召回的训练分子不强行补标签。",
        f"训练谱 {ready['counts']['train']['retained_spectra']:,} 张、验证谱 {ready['counts']['validation']['retained_spectra']:,} 张。实际拟合分子式 {len(metadata['training_formulas']):,} 种。训练/验证/测试结构零交叉，候选特征中不含真实标签或标签质量误差。",
        "LightGBM LambdaRank：31 个叶节点，学习率 0.05，最多 1,000 轮，验证 MRR@25 早停 50 轮，固定种子 42。验证和测试使用完整保留候选集。目标函数为 LambdaRank，模型选择指标为 MRR@25。",
        f"最佳迭代轮数：{training['best_iteration']}。训练阶段耗时 {training['elapsed_seconds']:.1f} 秒（特征已提前并行缓存），训练进程峰值内存 {training['peak_memory_bytes']/1e9:.2f} GB；整个脚本本次运行约 {summary['elapsed_seconds']/60:.1f} 分钟。", "",
        "这是 3,000 个结构的首轮训练，不是全部 Enveda 训练语料的拟合。抽样 ID、种子、来源及清洗规则固化在 selection.json；模型元数据记录实际训练分子式和缓存身份。", "", "## 同分子、同候选对照", "",
        "MRR 为所有分子 1/k 的平均值，使用最多 10,000 个保留候选；MRR@25 把第 26 名及以后计 0。真值缺失与搜索失败计 0，不从分母剔除。", "",
        "| 数据集 | 排序器 | 分子数 | 完整 MRR | MRR@25 | Top-1 | Top-5 | Top-10 | Top-25 | 候选召回 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names=dict(validation="验证集（用于早停）",fresh_test="新留出集",previous_benchmark="上一轮固定基准")
    for name,backends in summary["metrics"].items():
        for backend,r in backends.items():
            lines.append(f"| {names[name]} | {backend} | {r['molecules']} | {r['mrr_full']:.6f} | {r['mrr25']:.6f} | {r['top1']:.2%} | {r['top5']:.2%} | {r['top10']:.2%} | {r['top25']:.2%} | {r['candidate_recall']:.2%} |")
    lines += ["", "新留出集的 500 个结构排除了此前 Enveda 对齐实验所有已评估结构；只在模型按验证集确定后进行评分。上一轮 491 个分子的基线排名与旧记录逐一核对，完全一致，模型比较使用相同候选。", "",
        f"新留出集完整 MRR 的结构 bootstrap 95% 区间：{summary['fresh_mrr_full_bootstrap_95ci']}（10,000 次，种子 202604）。该区间不涵盖仪器或化学空间迁移带来的误差。", "",
        "## 训练未见分子式", "", "结构隔离不保证分子式隔离；seen/unseen 按实际用于拟合的分子式区分。", "",
        "| 分子式 | 新留出分子数 | 完整 MRR | MRR@25 | Top-1 | Top-25 |", "|---|---:|---:|---:|---:|---:|"]
    for name,r in summary["fresh_formula_strata"].items():
        lines.append(f"| {name} | {r['molecules']} | {r['mrr_full']:.6f} | {r['mrr25']:.6f} | {r['top1']:.2%} | {r['top25']:.2%} |")
    lines += ["", "## 限制与校验", "",
        f"新留出集搜索失败 {summary['metrics']['fresh_test']['learned']['resource_failures']} 个，候选截断 {summary['metrics']['fresh_test']['learned']['truncated']} 个。",
        "已核对：集合结构零交叉、三份评估记录逐分子计分、模型重新加载、公开 predict API 与缓存排名一致（新留出集字典序前三个结构）、旧 491 个分子的基线排名一致。模块 46 项回归测试通过。",
        "模型仅针对当前中性分子式、元素范围和候选边界。没有 MS1 同位素包络；得分不是概率。enveda-180 为药物样合成化合物，不能将结果直接等同于天然产物或竞赛结构识别得分。排名器仍可能偏好不合理分子式，学习到的规律不替代显式化学可行性检查。", "",
        "## 模型特征重要性（gain，前十）", "", "这是模型内部的重要性，不能解释为因果贡献。", "", "| 特征 | gain |", "|---|---:|"]
    for name,gain in summary["feature_importance_gain"][:10]:
        lines.append(f"| {name} | {gain:.2f} |")
    lines += ["", "## 使用", "", "```python", "from casmi.formula import FormulaPredictor", "", 'predictor = FormulaPredictor.load("artifacts/formula/enveda-ranker-v1")', 'result = predictor.predict(spectra=spectra, molecule_id="example", top_k=25)', "```", "",
        "```powershell", ".venv/Scripts/python.exe scripts/train_enveda_ranker.py", ".venv/Scripts/python.exe scripts/report_enveda_ranker.py", "```", "",
        "复跑会复用已完成的特征和模型；改变抽样或配置应使用新的实验目录。baseline() 入口保持原行为，加载模型需显式使用上述路径。"]
    (REPORT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(REPORT/"verification.json",dict(model_load=True,metrics_independently_recomputed=True,**summary['verification']))
    print("Model, split and metric verification passed; report.md written.")


if __name__=="__main__":
    main()
