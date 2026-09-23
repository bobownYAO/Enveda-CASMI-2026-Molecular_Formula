"""Independent distribution-weighted, structure-held-out source experiment.

Run from the project root with .venv/Scripts/python scripts/train_multisource_ranker.py.
All models share the frozen candidate generator. No test-based parameter selection.
"""

import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import hashlib
import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from casmi.formula import FormulaConfig, FormulaPredictor
from casmi.formula.chemistry import ADDUCTS, exact_mass, parse_formula
from casmi.formula.data import write_json
from casmi.formula.features import FEATURE_NAMES
from casmi.formula.training import _matrix
import train_enveda_ranker as cache_tools
from train_enveda_ranker import digest, identity
from tune_enveda_ranker import BASE, validation_records, metrics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/prepared-multisource-v1"
OUT = ROOT / "artifacts/reports/multisource-v1"
MODEL = ROOT / "artifacts/formula/multisource-ranker-v1"
OLD = ROOT / "artifacts/formula/enveda-ranker-v1"
BUDGETS = {"train": (3000, 30), "validation": (550, 15), "test": (1500, 75)}


def allocate(populations, priors, budget, minimum):
    """Bounded proportional allocation with a small-source coverage floor."""
    quotas = {s: min(minimum, n) for s, n in populations.items()}
    assert sum(quotas.values()) <= budget <= sum(populations.values())
    while sum(quotas.values()) < budget:
        source = max(
            (s for s in quotas if quotas[s] < populations[s]),
            key=lambda s: (budget * priors[s] - quotas[s], s),
        )
        quotas[source] += 1
    return quotas


def prepare():
    if (DATA / "ready.json").exists():
        return json.loads((DATA / "selection.json").read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    original = ROOT / "artifacts/prepared"
    catalogs = {}
    for split in BUDGETS:
        frame = (
            pl.scan_parquet(original / split / "*.parquet")
            .select("molecule_id", "ingest_lib")
            .unique()
            .collect()
        )
        catalogs[split] = {
            k[0]: sorted(g["molecule_id"].to_list())
            for k, g in frame.group_by("ingest_lib")
        }
    populations = {s: len(ids) for s, ids in sorted(catalogs["train"].items())}
    priors = {s: n / sum(populations.values()) for s, n in populations.items()}
    selected = {}
    for split, (budget, floor) in BUDGETS.items():
        quotas = allocate(
            {s: len(ids) for s, ids in catalogs[split].items()}, priors, budget, floor
        )
        selected[split] = {}
        for s in sorted(priors):
            seed = int.from_bytes(
                hashlib.sha256(f"20260926:{split}:{s}".encode()).digest()[:8], "big"
            )
            selected[split][s] = sorted(
                random.Random(seed).sample(catalogs[split][s], quotas[s])
            )
    split_ids = {
        k: set().union(*map(set, sources.values())) for k, sources in selected.items()
    }
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        # Check entire split populations, not merely the sampled source/structure pairs.
        assert not set().union(*map(set, catalogs[a].values())) & set().union(
            *map(set, catalogs[b].values())
        )
    selection = dict(
        priors=priors,
        train_populations=populations,
        selected=selected,
        budgets=BUDGETS,
        protocol="Source/structure pairs; fixed upstream structure split; source priors from eligible training population only; within-source uniform sampling; minimum quotas corrected by query weights. Historical test reuse possible; no tuning on test.",
        cleaning="Inherited neutral supported-label/adduct and <=20ppm QC; remove peaks above precursor+2; NO raw base-intensity threshold; retain missing metadata.",
        original_manifest=json.loads((original / "manifest.json").read_text()),
        original_model_sha256=hashlib.sha256(
            (OLD / "model.txt").read_bytes()
        ).hexdigest(),
    )
    fingerprint = digest(selection)
    write_json(DATA / "selection.json", selection)
    write_json(DATA / "manifest.json", dict(file_fingerprint=fingerprint))
    shutil.copy2(original / "config.json", DATA / "config.json")
    diagnostics = {}
    for split in BUDGETS:
        rows = []
        for path in sorted((original / split).glob("*.parquet")):
            part = pl.read_parquet(path).filter(
                pl.col("molecule_id").is_in(sorted(split_ids[split]))
            )
            for row in part.to_dicts():
                s, key = row["ingest_lib"], row["molecule_id"]
                if key not in selected[split][s]:
                    continue
                row["molecule_id"] = s + "::" + key
                keep = [
                    i
                    for i, mz in enumerate(row["ms2_mzs"])
                    if mz <= row["precursor_mz"] + 2
                ]
                row["ms2_mzs"] = [row["ms2_mzs"][i] for i in keep]
                row["ms2_normalized_intensities"] = [
                    row["ms2_normalized_intensities"][i] for i in keep
                ]
                rows.append(row)
        schema = pl.read_parquet(
            next((original / split).glob("*.parquet")), n_rows=0
        ).schema
        frame = pl.DataFrame(rows, schema=schema)
        keys = sorted(frame["molecule_id"].unique().to_list())
        assert len(keys) == BUDGETS[split][0]
        (DATA / split).mkdir(exist_ok=True)
        frame.write_parquet(DATA / split / "part-000.parquet")
        for key, g in frame.group_by("molecule_id"):
            rs = g.to_dicts()
            mass = exact_mass(parse_formula(rs[0]["molecular_formula"]))
            errors = [
                abs(r["precursor_mz"] / ADDUCTS[r["adduct"]].mz(mass) - 1) * 1e6
                for r in rs
            ]
            diagnostics[key[0]] = dict(
                source=rs[0]["ingest_lib"],
                structure_id=key[0].split("::")[1],
                split=split,
                mass=mass,
                max_label_abs_ppm=max(errors),
                min_label_abs_ppm=min(errors),
                spectra=len(rs),
                missing_base_fraction=sum(r["base_peak_intensity"] is None for r in rs)
                / len(rs),
                instruments=sorted({str(r["instrument_type"]) for r in rs}),
            )
        # Small shards prevent a few multi-spectrum compounds blocking every worker.
        random.Random(42).shuffle(keys)
        for index in range(0, len(keys), 25):
            shard = DATA / "shards" / f"{split}-{index // 25:03d}"
            (shard / split).mkdir(parents=True, exist_ok=True)
            frame.filter(
                pl.col("molecule_id").is_in(keys[index : index + 25])
            ).write_parquet(shard / split / "part-000.parquet")
            shutil.copy2(DATA / "config.json", shard / "config.json")
            write_json(
                shard / "manifest.json",
                dict(
                    file_fingerprint=digest(
                        [fingerprint, split, index, keys[index : index + 25]]
                    )
                ),
            )
        print("PREPARED", split, len(keys), "groups", len(rows), "spectra", flush=True)
    write_json(DATA / "diagnostics.json", diagnostics)
    write_json(
        DATA / "ready.json",
        dict(
            structure_overlap=0,
            counts={s: {k: len(v) for k, v in d.items()} for s, d in selected.items()},
        ),
    )
    return selection


def load(split):
    folder = DATA / "feature_cache" / digest(identity(DATA, split))[:20]
    meta = json.loads((folder / "manifest.json").read_text())
    return (*_matrix(folder, meta), meta, folder)


def weighted(records, priors, field="rank"):
    by_source = {
        s: metrics([r[field] for r in records if r["sources"] == [s]]) for s in priors
    }
    assert all(v["molecules"] for v in by_source.values())
    overall = {
        k: sum(priors[s] * v[k] for s, v in by_source.items())
        for k in next(iter(by_source.values()))
        if k != "molecules"
    }
    return dict(
        weighted=overall,
        by_source=by_source,
        sample_micro=metrics([r[field] for r in records]),
    )


def fit(selection):
    if (MODEL / "metadata.json").exists():
        return FormulaPredictor.load(MODEL)
    start = time.perf_counter()
    x, y, groups, tm, tf = load("train")
    vx, vy, vg, vm, vf = load("validation")
    usable = [
        r
        for r in tm["records"]
        if r["baseline_rank"] is not None and r["candidate_count"] >= 2
    ]
    assert len(usable) == len(groups)
    requested = Counter(r["sources"][0] for r in tm["records"])
    # Do not renormalize away source-dependent candidate failures: each sampled
    # query has its design weight, including queries excluded from fitting.
    qw = np.array(
        [
            selection["priors"][r["sources"][0]] / requested[r["sources"][0]]
            for r in usable
        ]
    )
    qw /= qw.mean()
    weights = np.repeat(qw, groups).astype(np.float32)
    ds = lgb.Dataset(
        x,
        label=y,
        group=groups,
        weight=weights,
        feature_name=FEATURE_NAMES,
        free_raw_data=False,
    )
    vs = lgb.Dataset(
        vx,
        label=vy,
        group=vg,
        feature_name=FEATURE_NAMES,
        reference=ds,
        free_raw_data=False,
    )

    def score(pred, dataset):
        records = validation_records(pred, vy, vg, vm)
        return (
            "weighted_mrr25",
            weighted(records, selection["priors"])["weighted"]["mrr25"],
            True,
        )

    booster = lgb.train(
        BASE,
        ds,
        num_boost_round=1000,
        valid_sets=[vs],
        feval=score,
        callbacks=[
            lgb.early_stopping(50, first_metric_only=True, verbose=False),
            lgb.log_evaluation(50),
        ],
    )
    MODEL.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL / "model.txt"))
    config = FormulaConfig.load(DATA / "config.json")
    metadata = dict(
        artifact_version=1,
        model_version="lambdarank-"
        + hashlib.sha256((MODEL / "model.txt").read_bytes()).hexdigest()[:16],
        feature_names=FEATURE_NAMES,
        config=config.to_dict(),
        config_version=config.version,
        best_iteration=booster.best_iteration,
        training_data_fingerprint=tm["data"],
        train_cache=tf.name,
        validation_cache=vf.name,
        training_formulas=sorted({r["formula"] for r in usable}),
        hyperparameters=BASE,
        limitations=[
            "Neutral formulas and configured element bounds only; no MS1 isotope envelope",
            "Scores are not probabilities",
            "Sampled, QC-eligible source distribution; source/structure pairs, not deduplicated population of molecules",
        ],
        library_versions=dict(lightgbm=lgb.__version__, numpy=np.__version__),
        selection_manifest=str(DATA / "selection.json"),
    )
    write_json(MODEL / "metadata.json", metadata)
    write_json(
        MODEL / "training_report.json",
        dict(
            requested=len(tm["records"]),
            actual_groups=len(groups),
            actual_by_source=dict(Counter(r["sources"][0] for r in usable)),
            best_iteration=booster.best_iteration,
            candidate_coverage=weighted(
                tm["records"], selection["priors"], "baseline_rank"
            ),
            validation_weighted_mrr25=score(booster.predict(vx, num_threads=4), vs)[1],
            elapsed_seconds=time.perf_counter() - start,
        ),
    )
    return FormulaPredictor.load(MODEL)


def evaluate(selection, predictor):
    x, y, groups, meta, _ = load("test")
    old = FormulaPredictor.load(OLD)
    assert (
        old.config.version
        == predictor.config.version
        == FormulaConfig.load(DATA / "config.json").version
    )
    records = validation_records(
        predictor.model.predict(x, num_threads=4), y, groups, meta
    )
    old_records = validation_records(
        old.model.predict(x, num_threads=4), y, groups, meta
    )
    diagnostics = json.loads((DATA / "diagnostics.json").read_text())
    for r, o in zip(records, old_records):
        assert r["molecule_id"] == o["molecule_id"]
        r.update(diagnostics[r["molecule_id"]], old_rank=o["rank"])
        assert (
            (r["rank"] is None)
            == (r["old_rank"] is None)
            == (r["baseline_rank"] is None)
        )
    write_json(OUT / "test-records.json", records)
    summary = {
        name: weighted(records, selection["priors"], field)
        for name, field in [
            ("multisource", "rank"),
            ("enveda", "old_rank"),
            ("baseline", "baseline_rank"),
        ]
    }
    # Paired cluster bootstrap: a shared structure has the same Poisson weight
    # across sources; preserves source priors and cross-library dependence.
    ids = sorted({r["structure_id"] for r in records})
    indices = {
        s: np.array([ids.index(r["structure_id"]) for r in records if r["source"] == s])
        for s in selection["priors"]
    }
    rng = np.random.default_rng(20260926)
    boot = {s: [] for s in selection["priors"]}
    totals = []
    for _ in range(3000):
        w = rng.poisson(1, len(ids))
        total = 0.0
        for s, ix in indices.items():
            rs = [r for r in records if r["source"] == s]
            delta = np.array(
                [
                    (1 / r["rank"] if r["rank"] else 0)
                    - (1 / r["old_rank"] if r["old_rank"] else 0)
                    for r in rs
                ]
            )
            value = (
                float(np.average(delta, weights=w[ix]))
                if w[ix].sum()
                else float(delta.mean())
            )
            boot[s].append(value)
            total += selection["priors"][s] * value
        totals.append(total)
    summary["paired_delta_95ci"] = {
        s: np.quantile(v, [0.025, 0.975]).tolist()
        for s, v in {**boot, "weighted_total": totals}.items()
    }
    summary["diagnostic_strata"] = {}
    for source in selection["priors"]:
        rs = [r for r in records if r["source"] == source]
        subsets = dict(
            recalled=[r for r in rs if r["rank"]],
            matched_mass_accuracy=[
                r
                for r in rs
                if 237 <= r["mass"] <= 465 and r["max_label_abs_ppm"] <= 10
            ],
        )
        summary["diagnostic_strata"][source] = {
            name: {
                field: metrics([r[field] for r in subset])
                for field in ["rank", "old_rank"]
            }
            for name, subset in subsets.items()
        }
        summary["diagnostic_strata"][source]["resource_failures"] = sum(
            r.get("status") == "resource_limit" for r in rs
        )
        summary["diagnostic_strata"][source]["all_spectra_outside_10ppm"] = sum(
            r["min_label_abs_ppm"] > 10 for r in rs
        )
    summary["verification"] = dict(
        structure_overlap=0,
        identical_candidates=True,
        unique_test_structures=len(ids),
        test_pairs=len(records),
        original_model_unchanged=hashlib.sha256(
            (OLD / "model.txt").read_bytes()
        ).hexdigest()
        == selection["original_model_sha256"],
    )
    assert summary["verification"]["original_model_unchanged"]
    write_json(OUT / "summary.json", summary)
    print(
        json.dumps(
            {
                k: v["weighted"]
                for k, v in summary.items()
                if k in ("multisource", "enveda", "baseline")
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    selection = prepare()
    print("SELECTION_READY", flush=True)
    cache_tools.DATA = DATA
    cache_tools.cache_all(max_workers=8)
    print("TRAINING", flush=True)
    predictor = fit(selection)
    evaluate(selection, predictor)


if __name__ == "__main__":
    main()
