"""原始谱图审计、清洗、结构隔离和分区准备。

``prepare`` 以有限批次读取原始 Parquet，先拒绝冲突结构、域外分子式/加合物和
质量错误，再按固定哈希把结构分到 train/validation/test，并写出后续训练和推理
共用的 prepared 分片、配置、manifest 与训练分子式集合。
"""

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .chemistry import ADDUCTS, ELEMENTS, exact_mass, format_formula, parse_formula
from .config import FormulaConfig
from .spectra import normalize_spectrum

INFERENCE_COLUMNS = (
    "precursor_mz",
    "adduct",
    "ionization_mode",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "collision_energy_ev",
    "base_peak_intensity",
    "instrument_type",
)


def write_json(path, data):
    """以 UTF-8、可读格式写出机器可读报告或元数据。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def split_for(key, seed=42):
    """用稳定哈希把结构标识确定性地分到 train/validation/test。"""
    value = (
        int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")
        / 2**64
    )
    return "train" if value < 0.8 else ("validation" if value < 0.9 else "test")


def file_fingerprint(path):
    """分块计算原始文件 SHA-256，作为 prepared/cache 身份的一部分。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=100000)
def _label(formula):
    """解析并规范化标签分子式；不支持的标签返回 ``None``。"""
    try:
        vector = parse_formula(formula)
        return vector, format_formula(vector), exact_mass(vector)
    except ValueError:
        return None


def _conflicts(path):
    """找出同一结构标识对应多个规范分子式的冲突组。"""
    frame = (
        pl.scan_parquet(path)
        .select("inchikey14", "molecular_formula")
        .unique()
        .collect()
    )
    labels = defaultdict(set)
    for key, formula in frame.iter_rows():
        value = _label(formula)
        labels[key].add(value[1] if value else formula)
    return {key for key, values in labels.items() if len(values) > 1}


def audit(path):
    """用 Polars 惰性扫描原始表，生成不改数据的质量概览报告。"""
    path = Path(path)
    start = time.perf_counter()
    scan = pl.scan_parquet(path)
    schema = scan.collect_schema()
    result = {
        "file": str(path.resolve()),
        "bytes": path.stat().st_size,
        "rows": scan.select(pl.len()).collect().item(),
        "schema": {k: str(v) for k, v in schema.items()},
        "null_counts": scan.select(pl.all().null_count()).collect().to_dicts()[0],
        "adducts": scan.group_by("adduct").len().collect().to_dicts(),
    }
    peaks = (
        scan.select(
            (
                pl.col("ms2_mzs").list.len()
                != pl.col("ms2_normalized_intensities").list.len()
            )
            .sum()
            .alias("misaligned"),
            (pl.col("ms2_mzs").list.len() == 0).sum().alias("empty"),
            pl.col("ms2_mzs").list.len().median().alias("median"),
            pl.col("ms2_mzs").list.len().max().alias("max"),
        )
        .collect()
        .to_dicts()[0]
    )
    result["peaks"] = peaks
    if "molecular_formula" in schema:
        formulas = scan.group_by("molecular_formula").len().collect()
        supported = 0
        unsupported = []
        for formula, count in formulas.iter_rows():
            if _label(formula):
                supported += count
            else:
                unsupported.append({"formula": formula, "spectra": count})
        result.update(
            unique_formulas=len(formulas),
            supported_formula_spectra=supported,
            unsupported_formulas=unsupported,
            conflicting_structures=len(_conflicts(path)),
            unique_structures=scan.select(pl.col("inchikey14").n_unique())
            .collect()
            .item(),
        )
        result["sources"] = (
            scan.group_by("ingest_lib")
            .agg(
                pl.len().alias("spectra"),
                pl.col("precursor_error_ppm")
                .abs()
                .median()
                .alias("provided_error_median_ppm"),
                pl.col("precursor_error_ppm")
                .abs()
                .quantile(0.99)
                .alias("provided_error_p99_ppm"),
            )
            .collect()
            .to_dicts()
        )
    if "molecule_id" in schema:
        result["molecules"] = (
            scan.select(pl.col("molecule_id").n_unique()).collect().item()
        )
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


def prepare(path, output, config=None, buckets=32, batch_size=16384):
    """把原始谱图准备成按 split/bucket 划分的 Parquet 数据集。

    每行先经过标签、加合物、前体质量和谱图校验；只有训练划分的最大元素计数
    用来推导最终边界，避免验证/留出标签反向影响搜索空间。输出目录必须为空，
    以防不同输入运行的分片被静默混合。
    """
    path = Path(path)
    output = Path(output)
    config = config or FormulaConfig()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation output must be empty; use a new directory")
    if buckets < 1:
        raise ValueError("buckets must be positive")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    fingerprint = file_fingerprint(path)
    conflicts = _conflicts(path)
    parquet = pq.ParquetFile(path)
    rejected = Counter()
    by_source = defaultdict(Counter)
    accepted = Counter()
    writers = {}
    maximum = np.zeros(10, dtype=np.int64)
    formula_sets = {s: set() for s in ("train", "validation", "test")}
    row_index = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            partitions = defaultdict(list)
            for row in batch.to_pylist():
                index = row_index
                row_index += 1
                key = row.get("inchikey14")
                source = row.get("ingest_lib") or "unknown"
                label = _label(row.get("molecular_formula"))
                reason = None
                if not key:
                    reason = "missing_structure_id"
                elif key in conflicts:
                    reason = "conflicting_structure"
                elif label is None:
                    reason = "unsupported_formula"
                elif row.get("adduct") not in ADDUCTS:
                    reason = "unsupported_adduct"
                else:
                    adduct = ADDUCTS[row["adduct"]]
                    expected = adduct.mz(label[2])
                    measured = row.get("precursor_mz")
                    if expected <= 0 or (adduct.ion_counts(label[0]) < 0).any():
                        reason = "invalid_ion_composition"
                    elif (
                        measured is None
                        or not np.isfinite(measured)
                        or abs((measured - expected) / expected * 1e6) > 20
                    ):
                        reason = "precursor_error_gt20ppm"
                    else:
                        try:
                            normalize_spectrum(row, config)
                        except (ValueError, KeyError, TypeError):
                            reason = "invalid_spectrum"
                if reason:
                    rejected[reason] += 1
                    by_source[source][reason] += 1
                    continue
                split = split_for(key, config.seed)
                if split == "train":
                    maximum = np.maximum(maximum, label[0])
                formula_sets[split].add(label[1])
                accepted[split] += 1
                by_source[source]["accepted"] += 1
                bucket = (
                    int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
                    % buckets
                )
                record = {name: row.get(name) for name in INFERENCE_COLUMNS}
                record.update(
                    molecule_id=key,
                    spectrum_id=f"{fingerprint[:16]}:{index}",
                    molecular_formula=label[1],
                    ingest_lib=source,
                )
                partitions[(split, bucket)].append(record)
            for key, rows in partitions.items():
                # 显式 schema 防止首批全为 null 时把列类型固定成不可用的 null。
                table = pa.Table.from_pylist(rows, schema=_prepared_schema())
                if key not in writers:
                    dest = output / key[0] / f"part-{key[1]:03d}.parquet"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    writers[key] = pq.ParquetWriter(
                        dest, table.schema, compression="zstd"
                    )
                writers[key].write_table(table)
            if row_index // 100000 > (row_index - batch.num_rows) // 100000:
                print(
                    f"prepare: {row_index} spectra read, {sum(accepted.values())} accepted",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()
    limits = {e: int(math.ceil(int(n) * 1.2)) for e, n in zip(ELEMENTS, maximum)}
    derived = FormulaConfig(**{**config.to_dict(), "element_limits": limits})
    write_json(output / "config.json", derived.to_dict())
    write_json(output / "train_formulas.json", sorted(formula_sets["train"]))
    coverage = {}
    for split, formulas in formula_sets.items():
        coverage[split] = {
            "unique_formulas": len(formulas),
            "within_element_limits": sum(
                all(n <= limits[e] for e, n in zip(ELEMENTS, parse_formula(f)))
                for f in formulas
            ),
        }
    result = {
        "file_fingerprint": fingerprint,
        "input": str(path.resolve()),
        "input_rows": row_index,
        "accepted_spectra": sum(accepted.values()),
        "accepted_by_split": dict(accepted),
        "rejected": dict(rejected),
        "by_source": {k: dict(v) for k, v in by_source.items()},
        "conflicting_structures": len(conflicts),
        "element_limits": limits,
        "coverage": coverage,
        "seed": config.seed,
        "buckets": buckets,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(output / "manifest.json", result)
    return result


def _prepared_schema():
    """返回 prepared Parquet 的固定列类型，保证各分片可互换读取。"""
    return pa.schema(
        [
            ("precursor_mz", pa.float64()),
            ("adduct", pa.string()),
            ("ionization_mode", pa.string()),
            ("ms2_mzs", pa.list_(pa.float64())),
            ("ms2_normalized_intensities", pa.list_(pa.float64())),
            ("collision_energy_ev", pa.list_(pa.float64())),
            ("base_peak_intensity", pa.float64()),
            ("instrument_type", pa.string()),
            ("molecule_id", pa.string()),
            ("spectrum_id", pa.string()),
            ("molecular_formula", pa.string()),
            ("ingest_lib", pa.string()),
        ]
    )


def iter_groups(prepared, split, max_molecules=None):
    """按分子 ID 从一个 split 的分片中逐组读取谱图。

    读取时只在单个分片内排序和分组，适合当前分桶布局；训练/评估调用方可用
    ``max_molecules`` 做小规模试跑，而不会改变数据切分规则。
    """
    if split not in ("train", "validation", "test"):
        raise ValueError("Invalid split")
    count = 0
    for path in sorted((Path(prepared) / split).glob("part-*.parquet")):
        frame = pl.read_parquet(path).sort("molecule_id")
        for key, group in frame.group_by("molecule_id", maintain_order=True):
            if max_molecules is not None and count >= max_molecules:
                return
            rows = group.to_dicts()
            yield {
                "molecule_id": key[0],
                "formula": rows[0]["molecular_formula"],
                "sources": sorted({r["ingest_lib"] for r in rows}),
                "spectra": [
                    {
                        k: v
                        for k, v in r.items()
                        if k in INFERENCE_COLUMNS + ("molecule_id", "spectrum_id")
                    }
                    for r in rows
                ],
            }
            count += 1
