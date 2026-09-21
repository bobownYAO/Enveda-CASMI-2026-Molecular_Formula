"""候选组学习、磁盘特征缓存和端到端评估。

训练不重新实现候选或特征逻辑，而是调用 ``FormulaPredictor`` 生成与推理一致的
候选组，再把固定组大小的特征送入 LightGBM LambdaRank。缓存和模型元数据保存
数据指纹、配置版本与实际训练分子式来源，便于复现和解释 seen/unseen 分层。
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import threading

import numpy as np
import psutil

from .config import FormulaConfig
from .candidates import SearchLimitExceeded
from .data import iter_groups, write_json
from .features import FEATURE_NAMES
from .predictor import FormulaPredictor


class MemoryMonitor:
    """在训练或评估期间采样当前进程 RSS，并记录峰值内存。"""

    def __enter__(self):
        """启动后台采样线程并返回监控对象。"""
        self.peak = 0
        self.stop = threading.Event()

        def sample():
            process = psutil.Process()
            while not self.stop.is_set():
                self.peak = max(self.peak, process.memory_info().rss)
                self.stop.wait(0.05)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        """记录最终 RSS、停止采样并等待线程退出。"""
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.stop.set()
        self.thread.join()


def ranking_metrics(ranks):
    """从真实标签排名计算召回、Top-K 和 MRR@25。"""
    n = len(ranks)
    return {
        "molecules": n,
        "candidate_recall": sum(r is not None for r in ranks) / n if n else 0.0,
        **{
            f"top{k}": sum(r is not None and r <= k for r in ranks) / n if n else 0.0
            for k in (1, 5, 10, 25)
        },
        "mrr25": sum(1 / r for r in ranks if r is not None and r <= 25) / n
        if n
        else 0.0,
    }


def _rank(order, formulas, truth):
    """在候选顺序中查找真值分子式的 1-based 排名，缺失返回 ``None``。"""
    return next((r for r, i in enumerate(order, 1) if formulas[i] == truth), None)


def _cache(prepared, split, predictor, limit=None):
    """生成或复用带数据/配置/特征身份的磁盘特征缓存。

    训练只采样真值、基线高分负例和确定性随机负例；验证/测试保留完整候选。
    ``progress.json`` 使长任务可以从已完成分子继续，资源失败则记录为失败而不
    伪造完整候选。
    """
    prepared = Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    identity = {
        "cache_version": 4,
        "data": manifest["file_fingerprint"],
        "config": predictor.config.version,
        "split": split,
        "limit": limit,
        "feature_names": FEATURE_NAMES,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = prepared / "feature_cache" / key
    if (folder / "manifest.json").exists():
        return folder, json.loads(
            (folder / "manifest.json").read_text(encoding="utf-8")
        )
    folder.mkdir(parents=True, exist_ok=True)
    arrays = []
    labels = []
    sizes = []
    records = []
    files = []
    row_count = 0
    checkpoint = folder / "progress.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        records = saved["records"]
        files = saved["files"]
    processed = {record["molecule_id"] for record in records}

    def flush():
        nonlocal arrays, labels, sizes, row_count
        if arrays:
            name = f"features-{len(files):05d}.npz"
            np.savez(
                folder / name,
                x=np.concatenate(arrays).astype(np.float32),
                y=np.concatenate(labels),
                groups=np.array(sizes),
            )
            files.append(name)
            arrays = []
            labels = []
            sizes = []
            row_count = 0
        write_json(checkpoint, {"records": records, "files": files})

    for index, group in enumerate(iter_groups(prepared, split, limit)):
        if group["molecule_id"] in processed:
            continue
        print(
            f"features {split}: {index + 1} {group['molecule_id']} ({len(group['spectra'])} spectra)",
            flush=True,
        )
        try:
            bundle = predictor.candidate_features(group["spectra"])
        except SearchLimitExceeded:
            records.append(
                {
                    "molecule_id": group["molecule_id"],
                    "formula": group["formula"],
                    "sources": group["sources"],
                    "baseline_rank": None,
                    "candidate_count": 0,
                    "total_candidates": None,
                    "truncated": False,
                    "status": "resource_limit",
                }
            )
            continue
        truth = group["formula"]
        formulas = bundle["formulas"]
        order = predictor.baseline_order(bundle)
        true_index = formulas.index(truth) if truth in formulas else None
        record = {
            "molecule_id": group["molecule_id"],
            "formula": truth,
            "sources": group["sources"],
            "baseline_rank": _rank(order, formulas, truth),
            "candidate_count": len(formulas),
            "total_candidates": bundle["total_candidates"],
            "truncated": bundle["truncated"],
        }
        records.append(record)
        if split == "train":
            if true_index is None or len(formulas) < 2:
                continue
            hard = [i for i in order if i != true_index][:127]
            remaining = [i for i in order if i != true_index and i not in set(hard)]
            seed = int.from_bytes(
                hashlib.sha256(
                    f"{predictor.config.seed}:{group['molecule_id']}".encode()
                ).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(seed)
            random = rng.choice(
                remaining, size=min(128, len(remaining)), replace=False
            ).tolist()
            selected = [true_index] + hard + random
        else:
            # 稳定分数并列时必须与推理端的公式字符串 tiebreak 一致。
            selected = sorted(range(len(formulas)), key=lambda i: formulas[i])
        if selected:
            arrays.append(bundle["features"][selected])
            labels.append(
                np.array([int(i == true_index) for i in selected], dtype=np.int32)
            )
            sizes.append(len(selected))
            row_count += len(selected)
        if row_count >= 50000 or (index + 1) % 5 == 0:
            flush()
    flush()
    result = {
        **identity,
        "files": files,
        "records": records,
        "groups": sum(len(np.load(folder / f)["groups"]) for f in files),
    }
    write_json(folder / "manifest.json", result)
    return folder, result


def _matrix(folder, manifest):
    """把分块 NPZ 特征拼成 LightGBM 所需的 memmap、标签和组大小。"""
    rows = 0
    groups = []
    for file in manifest["files"]:
        with np.load(folder / file) as part:
            rows += len(part["y"])
            groups.extend(part["groups"].tolist())
    if not rows:
        raise ValueError(f"No usable candidate groups for {manifest['split']}")
    x = np.lib.format.open_memmap(
        folder / "matrix.npy",
        mode="w+",
        dtype=np.float32,
        shape=(rows, len(FEATURE_NAMES)),
    )
    y = np.empty(rows, dtype=np.int32)
    offset = 0
    for file in manifest["files"]:
        with np.load(folder / file) as part:
            count = len(part["y"])
            x[offset : offset + count] = part["x"]
            y[offset : offset + count] = part["y"]
            offset += count
    x.flush()
    return x, y, np.array(groups, dtype=np.int32)


def train(
    prepared,
    output,
    max_train_molecules=None,
    max_validation_molecules=None,
    n_estimators=1000,
):
    """训练候选组 LambdaRank 模型并保存模型、元数据和训练报告。

    验证 MRR@25 只用于报告是否推荐 learned backend，代码不会自动替用户切换
    后端；资源失败和候选召回会进入报告分母。
    """
    import lightgbm as lgb

    prepared = Path(prepared)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Model output must be empty")
    config = FormulaConfig.load(prepared / "config.json")
    predictor = FormulaPredictor.baseline(config)
    start = time.perf_counter()
    with MemoryMonitor() as memory:
        train_dir, train_meta = _cache(
            prepared, "train", predictor, max_train_molecules
        )
        valid_dir, valid_meta = _cache(
            prepared, "validation", predictor, max_validation_molecules
        )
        x, y, groups = _matrix(train_dir, train_meta)
        vx, vy, vgroups = _matrix(valid_dir, valid_meta)
        if any(g > 10000 for g in groups) or any(g > 10000 for g in vgroups):
            raise ValueError(
                "LightGBM ranking supports at most 10000 candidates per group"
            )
        training = lgb.Dataset(
            x, label=y, group=groups, feature_name=FEATURE_NAMES, free_raw_data=False
        )
        validation = lgb.Dataset(
            vx,
            label=vy,
            group=vgroups,
            feature_name=FEATURE_NAMES,
            reference=training,
            free_raw_data=False,
        )
        all_valid_count = len(valid_meta["records"])

        def mrr(predictions, dataset):
            labels = dataset.get_label()
            offset = 0
            total = 0.0
            for size in dataset.get_group():
                order = np.argsort(-predictions[offset : offset + size], kind="stable")
                matches = np.flatnonzero(labels[offset : offset + size][order][:25])
                if len(matches):
                    total += 1 / (matches[0] + 1)
                offset += size
            return "mrr25", total / max(all_valid_count, 1), True

        booster = lgb.train(
            {
                "objective": "lambdarank",
                "metric": "None",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "verbosity": -1,
                "seed": config.seed,
                "num_threads": min(8, psutil.cpu_count(logical=False) or 1),
                "deterministic": True,
                "force_col_wise": True,
            },
            training,
            num_boost_round=n_estimators,
            valid_sets=[validation],
            feval=mrr,
            callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
        )
        score = mrr(booster.predict(vx, num_threads=1), validation)[1]
        baseline = ranking_metrics([r["baseline_rank"] for r in valid_meta["records"]])
        output.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(output / "model.txt"))
        model_version = (
            "lambdarank-"
            + hashlib.sha256((output / "model.txt").read_bytes()).hexdigest()[:16]
        )
        metadata = {
            "artifact_version": 1,
            "model_version": model_version,
            "feature_names": FEATURE_NAMES,
            "config": config.to_dict(),
            "config_version": config.version,
            "best_iteration": booster.best_iteration,
            "training_data_fingerprint": train_meta["data"],
            "train_cache": train_dir.name,
            "validation_cache": valid_dir.name,
            "training_formulas": sorted(
                {
                    r["formula"]
                    for r in train_meta["records"]
                    if r["baseline_rank"] is not None and r["candidate_count"] >= 2
                }
            ),
            "limitations": [
                "Neutral formulas only; configured elements and bounds only; no MS1 isotope envelope",
                "Scores are ranks, not calibrated probabilities",
            ],
            "library_versions": {"lightgbm": lgb.__version__, "numpy": np.__version__},
        }
        write_json(output / "metadata.json", metadata)
    report = {
        "training_groups": len(groups),
        "training_molecules_examined": len(train_meta["records"]),
        "training_candidate_recall": ranking_metrics(
            [r["baseline_rank"] for r in train_meta["records"]]
        )["candidate_recall"],
        "training_resource_failures": sum(
            r.get("status") == "resource_limit" for r in train_meta["records"]
        ),
        "validation_resource_failures": sum(
            r.get("status") == "resource_limit" for r in valid_meta["records"]
        ),
        "validation_molecules": len(valid_meta["records"]),
        "validation_baseline": baseline,
        "validation_mrr25": score,
        "recommended_backend": "learned" if score > baseline["mrr25"] else "baseline",
        "max_train_molecules": max_train_molecules,
        "max_validation_molecules": max_validation_molecules,
        "best_iteration": booster.best_iteration,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_memory_bytes": memory.peak,
        "model_version": model_version,
    }
    write_json(output / "training_report.json", report)
    return report


def evaluate(prepared, model=None, split="test", max_molecules=None, ablations=True):
    """在指定 split 上评估基线或模型，并输出分层指标和消融对照。

    学习模型的 seen/unseen 来源以模型元数据中的实际拟合分子式为准；基线则以
    prepared 数据保存的训练分子式集合为准。候选资源失败、无候选和截断都会单独
    记录，不会从评估分母中隐藏。
    """
    prepared = Path(prepared)
    predictor = (
        FormulaPredictor.load(model)
        if model
        else FormulaPredictor.baseline(FormulaConfig.load(prepared / "config.json"))
    )
    if model:
        metadata = json.loads(
            (Path(model) / "metadata.json").read_text(encoding="utf-8")
        )
        if "training_formulas" not in metadata:
            raise ValueError("Model lacks training-formula provenance")
        seen = set(metadata["training_formulas"])
    else:
        seen = set(
            json.loads((prepared / "train_formulas.json").read_text(encoding="utf-8"))
        )
    rows = []
    strata = defaultdict(list)
    start = time.perf_counter()
    with MemoryMonitor() as memory:
        for index, group in enumerate(iter_groups(prepared, split, max_molecules)):
            resource_limited = False
            try:
                bundle = predictor.candidate_features(group["spectra"])
            except SearchLimitExceeded:
                resource_limited = True
                from .spectra import normalize_spectrum

                bundle = {
                    "formulas": [],
                    "features": np.empty((0, len(FEATURE_NAMES))),
                    "support": np.empty((0, len(group["spectra"]))),
                    "errors": np.empty((0, len(group["spectra"]))),
                    "truncated": False,
                    "spectra": [
                        normalize_spectrum(row, predictor.config, i)
                        for i, row in enumerate(group["spectra"])
                    ],
                }
            formulas = bundle["formulas"]
            truth = group["formula"]
            baseline = predictor.baseline_order(bundle)
            if predictor.model is not None and len(formulas):
                score = predictor.model.predict(bundle["features"], num_threads=1)
                order = sorted(
                    range(len(formulas)), key=lambda i: (-score[i], formulas[i])
                )
            else:
                order = baseline
            rank = _rank(order, formulas, truth)
            mass_order = sorted(
                range(len(formulas)),
                key=lambda i: (
                    -int(bundle["support"][i].sum()),
                    float(np.abs(bundle["errors"][i]).mean()),
                    formulas[i],
                ),
            )
            record = {
                "molecule_id": group["molecule_id"],
                "rank": rank,
                "baseline_rank": _rank(baseline, formulas, truth),
                "mass_only_rank": _rank(mass_order, formulas, truth),
                "truncated": bundle["truncated"],
                "no_candidates": not bool(formulas),
                "resource_limited": resource_limited,
            }
            if ablations:
                try:
                    single = predictor.candidate_features(group["spectra"][:1])
                    record["single_spectrum_rank"] = _rank(
                        predictor.baseline_order(single), single["formulas"], truth
                    )
                except SearchLimitExceeded:
                    record["single_spectrum_rank"] = None
            rows.append(record)
            modes = {
                "positive" if s.adduct.charge > 0 else "negative"
                for s in bundle["spectra"]
            }
            for mode in modes:
                strata[f"mode:{mode}"].append(rank)
            for source in group["sources"]:
                strata[f"source:{source}"].append(rank)
            mass = bundle["spectra"][0].adduct.neutral_mass(
                bundle["spectra"][0].precursor_mz
            )
            strata[
                "mass:"
                + ("<300" if mass < 300 else ("300-600" if mass < 600 else ">=600"))
            ].append(rank)
            strata["formula:" + ("seen" if truth in seen else "unseen")].append(rank)
            if (index + 1) % 25 == 0:
                print(f"evaluate {split}: {index + 1} molecules", flush=True)
    n = len(rows)
    return {
        **ranking_metrics([r["rank"] for r in rows]),
        "split": split,
        "max_molecules": max_molecules,
        "no_candidate_rate": sum(r["no_candidates"] for r in rows) / n if n else 0.0,
        "truncation_rate": sum(r["truncated"] for r in rows) / n if n else 0.0,
        "resource_limit_rate": sum(r["resource_limited"] for r in rows) / n
        if n
        else 0.0,
        "strata": {k: ranking_metrics(v) for k, v in strata.items()},
        "ablations": {
            name: ranking_metrics([r[name] for r in rows])
            for name in (
                ["mass_only_rank", "baseline_rank", "single_spectrum_rank"]
                if ablations
                else ["mass_only_rank", "baseline_rank"]
            )
        },
        "elapsed_seconds": time.perf_counter() - start,
        "peak_memory_bytes": memory.peak,
        "model_version": predictor.model_version,
        "config_version": predictor.config.version,
        "records": rows,
    }
