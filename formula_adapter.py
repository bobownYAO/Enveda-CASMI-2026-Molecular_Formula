"""质谱 → 分子式：单文件配置与调用入口（默认 multi）。

只需修改下方 CONFIG，模型权重无需复制到本文件。

输入
----
lightgbm_model: "multi" / "enveda-only" / 模型目录 / model.txt 的路径。
    自定义模型必须同时提供 model.txt 与 metadata.json；不接受裸 Booster，
    因为元素边界、特征顺序及预处理配置必须与训练一致。相对模型路径以本文件
    所在目录为基准；迁移部署时请配置模型的绝对路径，模型不打包进 Python 包。
spectra: 同一分子的一张或多张谱，list[dict]。每张谱必填：
    precursor_mz: 正浮点数，前体 m/z。
    adduct: 加合物，如 "[M+H]+" 或 "[M-H]-"。
    ms2_mzs: 正 m/z 数组。
    ms2_normalized_intensities: 等长、非负强度数组。
    可选：ionization_mode（须与加合物一致）、collision_energy_ev（列表）、
    instrument_type、base_peak_intensity、spectrum_id、molecule_id。
molecule_id: 可选分子标识；top_k: 正整数，默认 CONFIG["top_k"]。
    不需要 molecular_formula、SMILES 或 precursor_error_ppm 等监督标签。

输出
----
predict_formula / FormulaAdapter.predict 返回 dict，可直接 json.dumps：
    molecule_id: str
    status: "ok" / "truncated" / "no_candidates" / "invalid_input" /
            "unsupported_input" / "resource_limit"
    candidates: list[dict]，按 rank 排序，每项字段为：
        rank: int（从 1 起）；formula: str（中性分子式）；exact_mass: float（Da）；
        score: float（排序分数，不是概率，不跨模型比较）；
        mass_errors_ppm: dict[spectrum_id, float]；supporting_spectra: int。
    warnings: list[str]；model_version: str；config_version: str；diagnostics: dict。
    diagnostics 正常计算后包含 candidate_count、total_candidates、spectra_used、
    elapsed_seconds；输入/资源失败时可能为空，调用方须检查 status。
    模型文件缺失/不兼容、非法 top_k 等配置错误直接抛异常，不回退到其他模型。

两组模型均使用 31 叶、学习率 0.05，已保存的最佳轮数分别为 multi=263、
enveda-only=227；实际信息通过 adapter.model_info 获取。推理只加载已有权重，
修改此注释或训练参数不会重训模型。候选和谱图预处理全部使用模型元数据配置，
本层不额外清洗或改变原 FormulaPredictor 的输入语义。

调用示例
--------
from formula_adapter import FormulaAdapter, predict_formula
adapter = FormulaAdapter()  # 同一模型多次调用时复用此对象
result = adapter.predict(spectra, molecule_id="example")
result = predict_formula(spectra, lightgbm_model="enveda-only", top_k=10)
results = adapter.predict_parquet("test.parquet", output_path="predictions.jsonl")

Parquet 必须含 molecule_id，按该列自动分组；输出 list[dict]，JSONL 每分子一行。
输入文件及输出路径按调用方工作目录解析；已存在的输出文件不会被覆盖。
批量入口会将表和结果载入内存；超大数据请自行逐分子调用 predict。
Windows 使用 workers>1 时，调用代码须置于 if __name__ == "__main__": 内。
"""

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

from casmi.formula import FormulaPredictor


# ==================== 日常调用只需修改此处 ====================
CONFIG = {
    "lightgbm_model": "multi",  # 或 "enveda-only"，或自定义模型路径
    "top_k": 25,
    "workers": 1,  # Parquet 批量推理进程数
}

MODEL_PRESETS = {
    "multi": "artifacts/formula/multisource-ranker-v1",
    "enveda-only": "artifacts/formula/enveda-ranker-v1",
}
# =============================================================

_ADAPTER_DIR = Path(__file__).resolve().parent


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _model_directory(model: str | Path) -> Path:
    if not isinstance(model, (str, Path)) or not str(model).strip():
        raise TypeError("lightgbm_model must be multi, enveda-only, or a model path")
    path = Path(MODEL_PRESETS.get(str(model), model)).expanduser()
    if not path.is_absolute():
        path = _ADAPTER_DIR / path
    path = path.resolve()
    if path.name == "model.txt":
        path = path.parent
    for filename in ("model.txt", "metadata.json"):
        if not (path / filename).is_file():
            raise FileNotFoundError(
                f"Missing formula model file: {path / filename}. "
                "Use multi/enveda-only or a directory containing "
                "model.txt and metadata.json."
            )
    return path


class FormulaAdapter:
    """加载一次模型，复用同一对象进行单分子或文件级推理。"""

    def __init__(
        self,
        lightgbm_model: str | Path | None = None,
        *,
        top_k: int | None = None,
        workers: int | None = None,
    ):
        self.top_k = _positive_int(CONFIG["top_k"] if top_k is None else top_k, "top_k")
        self.workers = _positive_int(
            CONFIG["workers"] if workers is None else workers, "workers"
        )
        self.lightgbm_model = (
            CONFIG["lightgbm_model"] if lightgbm_model is None else lightgbm_model
        )
        self.model_path = _model_directory(self.lightgbm_model)
        self._predictor = FormulaPredictor.load(self.model_path)

    @property
    def model_info(self) -> dict[str, Any]:
        """返回实际加载模型的信息；仅供查询，不作为可变的训练参数入口。"""
        return {
            "lightgbm_model": str(self.lightgbm_model),
            "model_path": str(self.model_path),
            "model_version": self._predictor.model_version,
            "config_version": self._predictor.config.version,
            "num_trees": self._predictor.model.num_trees(),
            "lightgbm_parameters": dict(self._predictor.model.params),
            "formula_config": self._predictor.config.to_dict(),
        }

    def predict(
        self,
        spectra: Iterable[Mapping[str, Any]],
        *,
        molecule_id: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """输入同一分子的谱列表，输出字段见本文件顶部的输出协议。"""
        k = self.top_k if top_k is None else _positive_int(top_k, "top_k")
        return self._predictor.predict(
            spectra, molecule_id=molecule_id, top_k=k
        ).to_dict()

    def predict_parquet(
        self,
        input_path: str | Path,
        *,
        output_path: str | Path | None = None,
        top_k: int | None = None,
        workers: int | None = None,
    ) -> list[dict[str, Any]]:
        """按 molecule_id 分组预测；可选保存 JSONL，不覆盖已有文件。"""
        k = self.top_k if top_k is None else _positive_int(top_k, "top_k")
        count = self.workers if workers is None else _positive_int(workers, "workers")
        target = Path(output_path) if output_path is not None else None
        if target is not None and target.exists():
            raise FileExistsError(f"Output already exists: {target}")
        results = [
            r.to_dict()
            for r in self._predictor.iter_predict_parquet(input_path, k, count)
        ]
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                for result in results:
                    stream.write(
                        json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                    )
        return results


def predict_formula(
    spectra: Iterable[Mapping[str, Any]],
    *,
    lightgbm_model: str | Path | None = None,
    molecule_id: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """单次调用快捷入口；频繁预测请复用 FormulaAdapter，避免重复加载模型。"""
    return FormulaAdapter(lightgbm_model, top_k=top_k).predict(
        spectra, molecule_id=molecule_id
    )


if __name__ == "__main__":
    # 可直接运行本文件测试接口；真实业务在此替换谱图或从其他 Python 文件导入。
    example_spectra = [
        {
            "precursor_mz": 181.0706646,
            "adduct": "[M+H]+",
            "ms2_mzs": [163.0601, 145.0495],
            "ms2_normalized_intensities": [1.0, 0.4],
        }
    ]
    print(
        json.dumps(
            predict_formula(example_spectra, molecule_id="example"),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
