"""Reproducible structure-uniform repeated holdout baseline experiment."""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from casmi.formula import FormulaConfig, FormulaPredictor
from casmi.formula.candidates import SearchLimitExceeded
from casmi.formula.chemistry import exact_mass, parse_formula
from casmi.formula.data import INFERENCE_COLUMNS, write_json
from casmi.formula.training import ranking_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports/baseline-sampling"


def evaluate(group):
    predictor = FormulaPredictor.baseline(FormulaConfig.load(ROOT / "artifacts/prepared/config.json"))
    start = time.perf_counter()
    record = {k: v for k, v in group.items() if k != "spectra"}
    record["input_spectra"] = len(group["spectra"])
    try:
        bundle = predictor.candidate_features(group["spectra"])
        order = predictor.baseline_order(bundle)
        rank = next((k for k, i in enumerate(order, 1) if bundle["formulas"][i] == group["formula"]), None)
        record.update(rank=rank, status="truncated" if bundle["truncated"] else "ok" if order else "no_candidates",
                      candidate_count=len(order), total_candidates=bundle["total_candidates"], unique_spectra=len(bundle["spectra"]))
    except SearchLimitExceeded:
        record.update(rank=None, status="resource_limit", candidate_count=0)
    record["seconds"] = time.perf_counter() - start
    return record


def metrics(records):
    result = ranking_metrics([r["rank"] for r in records])
    result["mrr_full"] = sum(1 / r["rank"] for r in records if r["rank"] is not None) / len(records)
    result["statuses"] = dict(Counter(r["status"] for r in records))
    result["mean_seconds"] = float(np.mean([r["seconds"] for r in records]))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "artifacts/prepared/test").glob("*.parquet"))
    ids = sorted(pl.concat([pl.read_parquet(p, columns=["molecule_id"]) for p in paths])["molecule_id"].unique().to_list())
    rounds = {str(seed): random.Random(seed).sample(ids, 100) for seed in range(101, 106)}
    selected = set().union(*map(set, rounds.values()))
    config = FormulaConfig.load(ROOT / "artifacts/prepared/config.json")
    write_json(OUT / "sampling.json", dict(population=len(ids), rounds=rounds, config=config.to_dict(), config_version=config.version,
        protocol="Each round: simple random sample of 100 structures without replacement. Independent seeds 101..105; all prepared spectra. Failures score zero. Full MRR uses capped candidate list (10000)."))
    groups = []
    for path in paths:
        frame = pl.read_parquet(path).filter(pl.col("molecule_id").is_in(sorted(selected)))
        for key, group in frame.group_by("molecule_id"):
            rows = group.to_dicts()
            groups.append(dict(molecule_id=key[0], formula=rows[0]["molecular_formula"],
                mass=exact_mass(parse_formula(rows[0]["molecular_formula"])), sources=sorted({r["ingest_lib"] for r in rows}),
                modes=sorted({r["ionization_mode"] for r in rows}),
                spectra=[{k: v for k, v in r.items() if k in INFERENCE_COLUMNS + ("molecule_id", "spectrum_id")} for r in rows]))
    existing = OUT / "records.jsonl"
    records = {r["molecule_id"]: r for r in map(json.loads, existing.read_text().splitlines())} if existing.exists() else {}
    start = time.perf_counter()
    print(f"population={len(ids)} unique_selected={len(selected)} cached={len(records)}", flush=True)
    with existing.open("a", encoding="utf-8") as log, ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(evaluate, g): g["molecule_id"] for g in sorted(groups, key=lambda g: g["molecule_id"]) if g["molecule_id"] not in records}
        for future in as_completed(futures):
            record = future.result()
            records[record["molecule_id"]] = record
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"{len(records)}/{len(selected)} {record['molecule_id']} rank={record['rank']} {record['status']} {record['seconds']:.1f}s", flush=True)
    summary = dict(rounds={seed: metrics([records[k] for k in keys]) for seed, keys in rounds.items()},
        pooled_rounds=metrics([records[k] for keys in rounds.values() for k in keys]),
        unique_molecules=metrics(list(records.values())), elapsed_seconds=time.perf_counter()-start)
    summary["mass_strata"] = {label: metrics(rs) for label, lo, hi in [("<300", 0, 300), ("300-600", 300, 600), (">=600", 600, float("inf"))] if (rs := [r for r in records.values() if lo <= r["mass"] < hi])}
    rr = np.array([1/r["rank"] if r["rank"] else 0 for r in records.values()])
    rng = np.random.default_rng(2026)
    summary["unique_mrr_full_bootstrap_95ci"] = np.quantile(rng.choice(rr, (10000, len(rr)), replace=True).mean(axis=1), [.025, .975]).tolist()
    write_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
