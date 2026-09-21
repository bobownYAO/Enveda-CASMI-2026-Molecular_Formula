# CASMI 长期维护检查清单

这份清单是后来者理解和修改 CASMI 的长期入口。项目的主数据流是：

```text
原始 Parquet
  → audit（质量概览）
  → prepare（清洗、结构隔离、固定切分、分桶写入）
  → molecule grouping（按 molecule_id 聚合谱图）
  → spectrum normalization（校验、合峰、归一化、峰截断）
  → candidate enumeration（前体质量窗口中的中性分子式）
  → fragment features（碎片/中性丢失解释特征）
  → baseline 或 LightGBM ranking
  → prediction / evaluation
  → JSONL、模型元数据和报告
```

## 项目边界

- 当前输出是中性分子式候选，不是竞赛最终要求的 SMILES。
- 推理可以离线、CPU 运行；不会在运行时下载数据或模型。
- 训练和推理共用候选生成与特征提取逻辑，模型只负责候选排序。
- `artifacts/`、原始数据、缓存和虚拟环境是运行产物，不应手工与代码改动混合。
- 当前目录没有 Git 元数据。本项目暂不依赖提交历史追踪修改；如未来启用 Git，
  应继续忽略原始数据、`artifacts/`、`.venv/`、缓存和编译产物，只跟踪源码、测试和文档。

## 推荐阅读顺序

按“先看外部契约，再看算法，再看数据和验证”的顺序阅读：

1. `README.md`：目标、输入字段、命令和输出状态。
2. `pyproject.toml`、`configs/formula.json`：运行环境和默认边界。
3. `src/casmi/__main__.py`、`src/casmi/cli.py`：命令行入口和阶段分派。
4. `src/casmi/formula/__init__.py`、`types.py`、`config.py`：公共 API、结果协议和配置版本。
5. `src/casmi/formula/chemistry.py`：元素向量、精确质量、DBE 和加合物。
6. `src/casmi/formula/spectra.py`：输入校验、重复峰合并和谱图规范化。
7. `src/casmi/formula/candidates.py`：有界质量搜索、剪枝和资源上限。
8. `src/casmi/formula/features.py`：碎片匹配、中性丢失和跨谱聚合。
9. `src/casmi/formula/predictor.py`：候选生成、基线/学习排序和批量预测。
10. `src/casmi/formula/data.py`：原始数据审计、准备、切分和分片读取。
11. `src/casmi/formula/training.py`：缓存、负例采样、训练和分层评估。
12. `tests/`：按 `test_chemistry.py` → `test_prediction.py` → `test_data_training.py` 阅读行为边界。
13. `scripts/validate_artifacts.py`：真实产物的一致性验收。
14. `docs/validation.md`、`artifacts/reports/`：已完成运行的规模、指标和限制。

## 模块与验证入口

| 修改区域 | 首选验证 |
| --- | --- |
| `chemistry.py`、`candidates.py` | `python -m pytest -q tests/test_chemistry.py` |
| `spectra.py`、`features.py`、`predictor.py` | `python -m pytest -q tests/test_prediction.py` |
| `data.py`、`training.py`、CLI 配置 | `python -m pytest -q tests/test_data_training.py` |
| CLI 入口 | `python -m casmi formula --help` 和对应子命令 `--help` |
| 真实产物 | `python scripts/validate_artifacts.py` |
| 全部源码 | `python -m compileall -q src tests scripts examples` |
| 全部回归 | `python -m pytest -q` |

Windows 若默认临时目录权限不足，应在受控环境中为 Pytest 指定可写的
`--basetemp`，或使用项目配置的虚拟环境运行测试。

## 必须保持的不变量

- 推理特征不能使用分子式标签、SMILES 或原始 `precursor_error_ppm`。
- 同一个 `molecule_id` 的谱图才可以合并；重复谱只能贡献一次证据。
- 前体 ppm 边界使用理论质量作分母；修改公式时必须保留边界测试。
- 质量搜索的 `max_search_results` 是“搜索是否完成”的资源上限，不能等同于
  `max_candidates` 的展示/排序上限；资源失败不能冒充完整候选结果。
- 训练、验证和留出测试按结构标识固定切分，不能发生结构交叉。
- 元素范围只从保留的训练划分推导；验证和测试标签不能扩大推理搜索空间。
- `prepared/config.json` 是准备数据的配置来源；模型预测使用模型内固化配置。
- 模型加载必须同时匹配 artifact 版本、特征名称、特征数量和配置哈希。
- 学习模型的 seen/unseen 分层必须使用模型实际拟合过的分子式集合。
- 公开 test 用于接口和性能验收，不用于声称效果或竞赛成绩。

## 修改前检查

- [ ] 明确改动属于化学、谱图、候选、特征、预测、数据、训练还是 CLI 层。
- [ ] 阅读该层的模块注释、相邻测试和 `docs/validation.md` 相关限制。
- [ ] 确认是否会影响配置版本、特征名称、模型元数据或产物格式。
- [ ] 不直接编辑原始数据、prepared 分片、模型文件、预测 JSONL 或缓存。
- [ ] 若修改算法行为，先补能描述新行为的测试，再实现代码；纯注释改动不新增行为。

## 修改后检查

- [ ] 运行受影响模块的专项测试。
- [ ] 运行 `python -m compileall -q src tests scripts examples`。
- [ ] 运行 `python -m pytest -q`，记录通过数量和失败原因。
- [ ] 运行相关 CLI `--help`；涉及真实产物时运行验收脚本。
- [ ] 检查公共 API、错误状态、配置版本和输出字段没有非预期变化。
- [ ] 注释解释的是流程、原因和边界，而不是重复显而易见的语法。
- [ ] 注释中的数字、路径、版本和实验结论与 README/validation 记录一致。

## 常见排障入口

- 输入被拒绝：先看 `spectra.py` 的字段和数值校验，再看 `FormulaPrediction.status`。
- 候选为空：检查加合物、元素边界和前体质量窗口，不要先放宽资源上限。
- `resource_limit`：区分质量搜索和碎片搜索超限，不能把部分结果当完整列表。
- 模型无法加载：比较 `metadata.json` 的 artifact 版本、特征名称、配置版本和模型文件。
- 训练/评估结果异常：先检查 `manifest.json` 的切分、拒绝原因和候选召回，再看排序指标。
- 公开 test 指标变化：先确认是否只是配置/产物变化；公开 test 不提供效果标签。

