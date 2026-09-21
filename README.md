# CASMI：质谱推导分子式

输入同一分子的一个或多个 MS/MS 谱，输出排序后的中性分子式候选。
包含不需要训练的基线、可训练的 LightGBM 排序器，以及数据质控、准备、训练、评估和批量预测命令。
所有推理均可离线、CPU 运行。这里输出的是分子式，不是竞赛最终要求的 SMILES。

## 快速调用

本项目已创建 Python 3.11 环境 `.venv`。在项目根目录可直接运行
`.venv\Scripts\python.exe` 或 `.venv\Scripts\casmi.exe`。新环境安装方式：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Linux 使用 `.venv/bin/python` 与 `.venv/bin/casmi`。需要离线安装时，应事先为目标平台准备依赖 wheel；运行时不会下载数据或模型。

```python
from casmi.formula import FormulaConfig, FormulaPredictor

predictor = FormulaPredictor.baseline()
result = predictor.predict(
    spectra=[{
        "precursor_mz": 181.07066456,
        "adduct": "[M+H]+",
        "ms2_mzs": [163.06009988, 145.04953520],
        "ms2_normalized_intensities": [1.0, 0.4],
    }],
    molecule_id="example",
    top_k=25,
)
print(result.to_json())
for candidate in result.candidates:
    print(candidate.rank, candidate.formula, candidate.score)

# 指定自己的元素范围或质量容差。
predictor = FormulaPredictor.baseline(FormulaConfig.load("configs/formula.json"))
# 加载已训练模型。不存在或配置/特征版本不兼容时明确报错。
# predictor = FormulaPredictor.load("artifacts/formula/enveda-ranker-v1")
# results = predictor.predict_parquet("test.parquet", top_k=25)
```

本地已提供 Enveda/timsTOF 模型 `artifacts/formula/enveda-ranker-v1`，可直接加载。
使用 3,000 个结构训练、500 个结构验证；新的 500 个独立留出结构 Top-1 为 83.8%、
Top-5 为 98.0%、MRR@25 为 0.8977。训练时未见分子式的 284 个分子 Top-1 为 76.4%。
这是 Enveda 来源的首轮训练，尚未全量拟合，也未验证天然产物或其他仪器的同等效果。
46 项回归测试通过。训练协议、同候选基线对照和使用限制见
[Enveda 训练报告](artifacts/reports/enveda-ranker-v1/report.md)。
后续已比较原模型与 7 组新参数：验证集选中的 63 叶模型在另一批 500 个新留出分子上
未优于原模型（Top-1 84.2% 对 85.0%），因此仍保留 `enveda-ranker-v1` 作为当前推荐版本。
完整对照见[参数实验报告](artifacts/reports/enveda-tuning-v1/report.md)。
扩展到另外 10 个数据库来源后，冻结模型的平均 1/k 约为 0.24–0.73，明显低于
Enveda 内部约 0.90 的结果；部分高质量分子还会触发候选搜索资源上限。
来源分层、GNPS 强度门槛对照和范围限制见
[跨数据库报告](artifacts/reports/external-sources-v1/report.md)。
之前的 `artifacts/formula/pilot-model` 及其 32 分子试跑报告 `docs/validation.md` 保留作历史记录；
已有公开 test 预测文件来自此前模型，并未由本轮模型覆盖。

必填谱字段是上例的四项。可选字段为 `spectrum_id`、`molecule_id`、`ionization_mode`、
`collision_energy_ev`（列表）、`instrument_type`、`base_peak_intensity`。
离子模式不提供时从加合物推导；提供时必须一致。SMILES、分子式和标签质量误差不会进入推理特征。
多谱输入不能混入不同 `molecule_id`；精确重复记录不会重复增加证据。

文件入口按 `molecule_id` 分组。`predict_parquet(..., workers=4)` 可并行处理；
Windows 脚本需在 `if __name__ == "__main__":` 下启动多进程。默认单进程可直接用于 Notebook。
需要逐条消费结果时使用 `iter_predict_parquet()`。
文件预测入口会将输入文件及分组载入内存，适用于当前约 5 MB 的 test；
超大推理数据应由调用方分组后逐个调用 `predict()`。训练数据准备使用独立的分批读取流程。

## 命令行流程

以下示例在已激活 `.venv` 的终端执行；不激活也可用 `.venv\Scripts\casmi.exe` 替代 `casmi`。
原始数据保持在 `enveda-CASMI26-molecule-id-mass-spectra/`。

```text
casmi formula audit --input enveda-CASMI26-molecule-id-mass-spectra/train.parquet --output artifacts/reports/audit.json
casmi formula prepare --input enveda-CASMI26-molecule-id-mass-spectra/train.parquet --output artifacts/prepared
casmi formula predict --input enveda-CASMI26-molecule-id-mass-spectra/test.parquet --config artifacts/prepared/config.json --output artifacts/predictions/baseline.jsonl --workers 4
casmi formula train --input artifacts/prepared --output artifacts/formula/my-model --max-molecules 128 --max-validation-molecules 32
casmi formula evaluate --input artifacts/prepared --model artifacts/formula/my-model --split test --max-molecules 32 --output artifacts/reports/holdout.json
casmi formula predict --input enveda-CASMI26-molecule-id-mass-spectra/test.parquet --model artifacts/formula/my-model --output artifacts/predictions/learned.jsonl
```

`prepare` 和 `train` 要求新的空输出目录，防止混合不同运行的分片或模型。
已经准备好的 `artifacts/prepared` 可直接复用，不必重新处理原始数据。
去掉两个 `--max-...` 参数即可使用全部相应划分；全量拟合可能耗时很长，并需要更多内存。
候选特征按配置和输入指纹缓存，重跑会复用；特征矩阵落盘为 NumPy memmap，LightGBM 自身仍需内存。

配置使用 JSON。可以直接传 `configs/formula.json`，或者使用带 `formula` 对象、`input`、`output`、
`model` 的包装配置。命令行路径优先。训练使用 prepared 内固化的配置，模型推理使用模型内固化配置。
这些入口不允许通过配置暗中改变模型的处理规则。

## 输出约定

每个 `FormulaPrediction` 包含：

| 字段 | 含义 |
|---|---|
| `molecule_id` | 输入分子标识 |
| `status` | 本次推理状态 |
| `candidates` | 按排名排序、无重复的候选列表 |
| `warnings` | 证据不足、质量不一致、截断等提示 |
| `model_version` / `config_version` | 排序器与处理配置标识 |
| `diagnostics` | 候选数量、使用谱数与耗时 |

候选包含 `rank`、`formula`、`exact_mass`（Da）、`score`、`mass_errors_ppm` 和 `supporting_spectra`。
质量误差为 `(measured - theoretical) / theoretical × 10^6`，按谱 ID 返回。
基线分数是顺序编号；学习模型分数为原始排序分数。二者均不是概率，也不能跨分子比较。

状态：`ok`、`truncated`（正常排序但候选被裁剪）、`no_candidates`、
`invalid_input`、`unsupported_input`、`resource_limit`。
资源上限与 Top-K 不同：单次质量搜索超过 `max_search_results` 会返回明确失败，
不会把未完成枚举的结果冒充完整候选。批量 JSONL 每个分子一行，失败分子也保留记录。

## 方法与适用范围

1. 校验谱峰、合并完全相同 m/z、相对强度过滤、最多保留 128 峰并归一化。
2. 根据加合物将前体质量换算为中性质量区间，做元素组成枚举。整数剩余质量表仅用于保守剪枝，结果再以精确原子质量复核。
3. 多谱取候选并集，先按支持谱数、平均绝对质量误差、分子式字符串截取最多 10,000 个。
4. 用候选离子组成解释碎片，允许金属保留/丢失，并提取水、CO、CO2、NH3 丢失特征。
5. 聚合逐谱特征，执行基线排序或 LightGBM LambdaRank。

支持 C/H/N/O/P/S/F/Cl/Br/I 和 10 种加合物：
`[M+H]+`、`[M+NH4]+`、`[M-H2O+H]+`、`[M-2H2O+H]+`、`[M+Na]+`、`[M+K]+`、
`[M-H]-`、`[M-H2O-H]-`、`[M+CH2O2-H]-`、`[M+Cl]-`。

首版元素上限已按本地训练划分的最大元素计数上浮 20% 固化。新数据运行 `prepare` 会重新计算。
带电结构标签、其他元素和其他加合物不纳入训练；不能通过删除电荷符号将它们伪装成中性分子式。
没有独立 MS1 同位素包络；碎片匹配是原子组成可解释性特征，不代表已证明存在对应反应路径。
质量搜索可能产生化学上不合理的组成；本版将 DBE 和元素比例用于排序，不用未经验证的硬价态规则截断召回。
高质量峰在候选前体质量一致时枚举较小的中性丢失，再逐候选核对精确质量，减少搜索开销。
低信息谱、高质量分子、罕见元素组合和域外仪器数据需要重点评估。

## 训练与评估

按 `SHA256("42:" + inchikey14)` 固定划分约 80%/10%/10%，相同结构跨来源仍属于同一划分。
全量预处理隔离冲突结构，重新计算质量误差并排除绝对值 >20 ppm 的谱；按来源记录每个剔除原因。
元素范围只从保留的训练划分计算。验证和留出测试使用完整候选，漏召回、无候选与资源失败均计入分母。

训练每组最多使用真值、127 个基线高分负例和 128 个随机负例；真值未自然进入候选时不补入。
模型为 31 叶节点、学习率 0.05、最多 1,000 轮，验证 MRR@25 连续 50 轮未提升则早停。
只有验证 MRR@25 优于基线，报告才会推荐学习后端；代码不会自动替用户切换后端。

报告包含 Top-1/5/10/25、MRR@25、候选召回、截断/无候选/资源失败率、耗时与采样峰值 RSS，
并按极性、质量区间、来源、训练中已见/未见分子式分层。多来源/多极性分子可出现在多个分层内。
学习模型的已见/未见划分使用实际拟合过的分子式集合，基线则参照可用训练划分。
消融比较质量排序、单谱碎片基线、多谱碎片基线与学习排序。
公开 test 是流程验收样例；效果评估使用结构隔离的留出数据。分子式 MRR 不等于竞赛 SMILES MRR。

## 文件与验证

主要代码位于 `src/casmi/formula/`：`chemistry`/`candidates` 负责化学和枚举，
`spectra`/`features` 负责谱图与特征，`predictor` 是公共入口，`data`/`training` 负责数据与评估。
`configs/formula.json` 是已核实默认配置；运行产物统一在 `artifacts/`。

长期维护检查项、端到端数据流和推荐阅读顺序见
[`docs/maintenance-checklist.md`](docs/maintenance-checklist.md)。

```text
python -m pytest -q
python -m casmi formula --help
```

测试包含质量基准、精确 ppm 边界、枚举与穷举对照、正负/金属碎片原子守恒、重复谱、输入异常、
训练隔离、资源失败、模型加载一致性和 CLI/多进程一致性。实际运行结果见 `docs/validation.md`。
