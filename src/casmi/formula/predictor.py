"""分子式候选生成、排序和批量预测的公共编排层。

预测器把谱图清洗、前体质量搜索、碎片特征和基线/学习排序串成一个调用流程，
同时把输入错误、无候选、截断和资源上限转换为可序列化的预测状态。
"""

from pathlib import Path
import json
import time
from collections.abc import Mapping

import numpy as np

from .candidates import FormulaEnumerator, SearchLimitExceeded
from .chemistry import MASSES, format_formula
from .config import FormulaConfig
from .features import FEATURE_NAMES, extract_features
from .spectra import UnsupportedInput, normalize_spectrum
from .types import FormulaCandidate, FormulaPrediction


class FormulaPredictor:
    """使用固定配置生成分子式候选，并可选加载 LightGBM 排序器。"""

    def __init__(self, config, model=None, model_version="baseline-v1"):
        """初始化候选搜索器和可选模型；不在构造阶段读取输入数据。"""
        self.config = config
        self.enumerator = FormulaEnumerator(
            config.element_limits, config.max_search_results
        )
        self.model = model
        self.model_version = model_version

    @classmethod
    def baseline(cls, config=None):
        """创建不需要训练的质量/碎片基线预测器。"""
        return cls(config or FormulaConfig())

    @classmethod
    def load(cls, path):
        """加载模型并验证产物版本、特征列和配置哈希的一致性。"""
        import lightgbm as lgb

        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if (
            metadata["artifact_version"] != 1
            or metadata["feature_names"] != FEATURE_NAMES
        ):
            raise ValueError("Incompatible formula model or feature schema")
        config = FormulaConfig(**metadata["config"])
        if config.version != metadata["config_version"]:
            raise ValueError("Model configuration checksum mismatch")
        model = lgb.Booster(model_file=str(path / "model.txt"))
        if model.num_feature() != len(FEATURE_NAMES):
            raise ValueError("Model feature count mismatch")
        return cls(config, model, metadata["model_version"])

    def candidate_features(self, spectra):
        """清洗谱图、生成候选并返回排序和特征所需的中间 bundle。

        相同指纹的谱图只保留一次证据；不同前体质量产生的候选取并集，随后按
        支持谱数、平均质量误差和公式字符串稳定排序并应用候选数量上限。搜索
        资源超限会继续向上抛出，由公共入口转换为 ``resource_limit``。
        """
        normalized = []
        seen = set()
        warnings = []
        if not spectra:
            raise ValueError("At least one spectrum is required")
        if not all(isinstance(row, Mapping) for row in spectra):
            raise ValueError("Each spectrum must be a mapping")
        # 一个 predict 调用只代表一个分子，避免跨分子合并质量证据。
        group_ids = {
            row["molecule_id"] for row in spectra if row.get("molecule_id") is not None
        }
        if len(group_ids) > 1:
            raise ValueError("All spectra must belong to one molecule")
        for index, row in enumerate(spectra):
            spec = normalize_spectrum(row, self.config, index)
            if spec.fingerprint not in seen:
                if spec.spectrum_id in {s.spectrum_id for s in normalized}:
                    raise ValueError("Conflicting duplicate spectrum_id")
                normalized.append(spec)
                seen.add(spec.fingerprint)
                if len(spec.mzs) < 5:
                    warnings.append(
                        f"{spec.spectrum_id}: insufficient fragment evidence"
                    )
        unique_vectors = set()
        searched = set()
        # 每种 (precursor_mz, adduct) 只搜索一次，再把合法向量合并成候选并集。
        for spec in normalized:
            key = (spec.precursor_mz, spec.adduct.name)
            if key in searched:
                continue
            searched.add(key)
            fraction = self.config.precursor_ppm * 1e-6
            low = spec.adduct.neutral_mass(spec.precursor_mz / (1 + fraction))
            high = spec.adduct.neutral_mass(spec.precursor_mz / (1 - fraction))
            mass = (low + high) / 2
            tol = (high - low) / 2
            hits = self.enumerator.search(mass, tol)
            if len(hits):
                valid = (spec.adduct.ion_counts(hits) >= 0).all(axis=1)
                unique_vectors.update(v.tobytes() for v in hits[valid])
        if not unique_vectors:
            return dict(
                vectors=np.empty((0, 10)),
                spectra=normalized,
                warnings=warnings,
                truncated=False,
                features=np.empty((0, len(FEATURE_NAMES))),
                errors=np.empty((0, len(normalized))),
                support=np.empty((0, len(normalized))),
                formulas=[],
                total_candidates=0,
            )
        vectors = np.frombuffer(
            b"".join(sorted(unique_vectors)), dtype=np.int64
        ).reshape(-1, 10)
        masses = vectors @ MASSES
        errors = np.column_stack(
            [
                (spec.precursor_mz - spec.adduct.mz(masses))
                / spec.adduct.mz(masses)
                * 1e6
                for spec in normalized
            ]
        )
        support = np.column_stack(
            [
                (np.abs(errors[:, i]) <= self.config.precursor_ppm + 1e-6)
                & (spec.adduct.ion_counts(vectors) >= 0).all(axis=1)
                for i, spec in enumerate(normalized)
            ]
        )
        if len(normalized) > 1 and not support.all(axis=1).any():
            warnings.append("inconsistent precursor masses: candidate union retained")
        formulas = [format_formula(v) for v in vectors]
        order = sorted(
            range(len(vectors)),
            key=lambda i: (
                -int(support[i].sum()),
                float(np.abs(errors[i]).mean()),
                formulas[i],
            ),
        )
        total = len(order)
        truncated = total > self.config.max_candidates
        order = np.array(order[: self.config.max_candidates], dtype=int)
        vectors = vectors[order]
        errors = errors[order]
        support = support[order]
        formulas = [formulas[i] for i in order]
        if truncated:
            warnings.append(f"candidate list truncated from {total} to {len(order)}")
        features = extract_features(vectors, normalized, self.config, errors, support)
        return dict(
            vectors=vectors,
            spectra=normalized,
            warnings=warnings,
            truncated=truncated,
            features=features,
            errors=errors,
            support=support,
            formulas=formulas,
            total_candidates=total,
        )

    @staticmethod
    def baseline_order(bundle):
        """按多谱支持、碎片强度解释率、质量误差和公式名稳定排序。"""
        x = bundle["features"]
        explained = FEATURE_NAMES.index("intensity_fraction_mean")
        return sorted(
            range(len(x)),
            key=lambda i: (
                -int(bundle["support"][i].sum()),
                -float(x[i, explained]),
                float(np.abs(bundle["errors"][i]).mean()),
                bundle["formulas"][i],
            ),
        )

    def predict(self, spectra, molecule_id=None, top_k=25):
        """预测一个分子并把所有可预期失败转换成 ``FormulaPrediction`` 状态。

        ``unsupported_input`` 表示未支持的加合物等域外输入，``invalid_input``
        表示格式/数值错误，``resource_limit`` 表示搜索未完成；这些状态不能
        被当作空候选的正常结果。
        """
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        spectra = list(spectra)
        molecule_id = str(
            molecule_id
            if molecule_id is not None
            else (
                spectra[0].get("molecule_id", "molecule")
                if spectra and isinstance(spectra[0], Mapping)
                else "molecule"
            )
        )
        result = FormulaPrediction(
            molecule_id,
            "ok",
            model_version=self.model_version,
            config_version=self.config.version,
        )
        start = time.perf_counter()
        # 公共 API 将内部异常映射为稳定状态，批量 JSONL 因此不会丢掉失败分子。
        try:
            bundle = self.candidate_features(spectra)
        except SearchLimitExceeded as error:
            result.status = "resource_limit"
            result.warnings = [str(error)]
            return result
        except UnsupportedInput as error:
            result.status = "unsupported_input"
            result.warnings = [str(error)]
            return result
        except (ValueError, KeyError, TypeError) as error:
            result.status = "invalid_input"
            result.warnings = [str(error)]
            return result
        result.warnings = bundle["warnings"]
        count = len(bundle["vectors"])
        result.status = (
            "truncated" if bundle["truncated"] else ("ok" if count else "no_candidates")
        )
        if count:
            order = self.baseline_order(bundle)
            if self.model is None:
                scores = np.empty(count)
                scores[order] = np.arange(count, 0, -1, dtype=float)
            else:
                scores = self.model.predict(bundle["features"], num_threads=1)
                order = sorted(
                    range(count), key=lambda i: (-scores[i], bundle["formulas"][i])
                )
            for rank, i in enumerate(order[:top_k], 1):
                result.candidates.append(
                    FormulaCandidate(
                        rank,
                        bundle["formulas"][i],
                        float(bundle["vectors"][i] @ MASSES),
                        float(scores[i]),
                        {
                            s.spectrum_id: float(bundle["errors"][i, j])
                            for j, s in enumerate(bundle["spectra"])
                        },
                        int(bundle["support"][i].sum()),
                    )
                )
        result.diagnostics = {
            "candidate_count": count,
            "total_candidates": bundle["total_candidates"],
            "spectra_used": len(bundle["spectra"]),
            "elapsed_seconds": time.perf_counter() - start,
        }
        return result

    def predict_parquet(self, path, top_k=25, workers=1):
        """批量读取 Parquet 并返回物化的分子级预测列表。"""
        return list(self.iter_predict_parquet(path, top_k, workers))

    def iter_predict_parquet(self, path, top_k=25, workers=1):
        """按分子分组逐条产生预测，支持串行或进程池模式。

        当前实现会把输入表和分组任务载入内存，适合项目现有规模；超大文件应由
        调用方先分组后直接调用 ``predict``。Windows 多进程由本模块在 worker 中
        重建配置和 LightGBM 模型，调用脚本仍需放在 ``__main__`` 保护下。
        """
        import polars as pl
        from concurrent.futures import ProcessPoolExecutor

        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be positive")
        data = pl.read_parquet(path)
        if "molecule_id" not in data.columns:
            raise ValueError("Prediction Parquet requires molecule_id")
        if data["molecule_id"].null_count():
            raise ValueError("Null molecule_id")
        # 任务粒度固定为 molecule_id，保证串行和多进程输出顺序/内容一致。
        tasks = [
            (group.to_dicts(), str(key[0]), top_k)
            for key, group in data.sort("molecule_id").group_by(
                "molecule_id", maintain_order=True
            )
        ]
        if workers == 1:
            for args in tasks:
                yield self.predict(*args)
        else:
            model_string = (
                self.model.model_to_string() if self.model is not None else None
            )
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(self.config.to_dict(), model_string, self.model_version),
            ) as pool:
                yield from pool.map(_predict_worker, tasks)


def _init_worker(config, model_string, version):
    global _worker
    model = None
    if model_string is not None:
        import lightgbm as lgb

        model = lgb.Booster(model_str=model_string)
    _worker = FormulaPredictor(FormulaConfig(**config), model, version)


def _predict_worker(args):
    return _worker.predict(*args)
