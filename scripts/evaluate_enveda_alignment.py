"""Instrument/source-matched holdout experiment; raw data remain unchanged.

Run from the project root. Reuses the unchanged baseline and fixed holdout split.
Exports sampled source-only and test-cleaned Parquet files plus paired scores.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import copy
import hashlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from casmi.formula import FormulaConfig
from casmi.formula.data import INFERENCE_COLUMNS, write_json
from casmi.formula.chemistry import exact_mass, parse_formula, ADDUCTS
from sample_baseline import evaluate, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports/enveda-alignment"
PREPARED = ROOT / "artifacts/prepared"
SOURCE = (pl.col("ingest_lib") == "enveda-180") & (pl.col("instrument_type") == "timsTOF")


def clean_rows(rows):
    cleaned = []
    stats = dict(input_spectra=len(rows), removed_low_or_missing_base=0, input_peaks=0, removed_high_peaks=0)
    for row in rows:
        if row["base_peak_intensity"] is None or row["base_peak_intensity"] < 1000:
            stats["removed_low_or_missing_base"] += 1
            continue
        result = dict(row)
        mzs = row["ms2_mzs"]
        intensities = row["ms2_normalized_intensities"]
        assert len(mzs) == len(intensities)
        keep = [i for i, mz in enumerate(mzs) if mz <= row["precursor_mz"] + 2]
        stats["input_peaks"] += len(mzs)
        stats["removed_high_peaks"] += len(mzs) - len(keep)
        result["ms2_mzs"] = [mzs[i] for i in keep]
        result["ms2_normalized_intensities"] = [intensities[i] for i in keep]
        cleaned.append(result)
    stats["retained_spectra"] = len(cleaned)
    return cleaned, stats


def group_from(rows):
    return dict(molecule_id=rows[0]["molecule_id"], formula=rows[0]["molecular_formula"],
        mass=exact_mass(parse_formula(rows[0]["molecular_formula"])), sources=sorted({r["ingest_lib"] for r in rows}),
        modes=sorted({r["ionization_mode"] for r in rows}),
        spectra=[{k: v for k, v in r.items() if k in INFERENCE_COLUMNS + ("molecule_id", "spectrum_id")} for r in rows])


def run_task(arm, group):
    if not group["spectra"]:
        record = {k: v for k, v in group.items() if k != "spectra"}
        record.update(rank=None, status="no_spectra_after_cleaning", candidate_count=0, seconds=0, input_spectra=0)
    else:
        record = evaluate(group)
    record["arm"] = arm
    return record


def distribution(rows):
    f = pl.DataFrame(rows)
    counts = f.group_by("molecule_id").len()["len"].to_numpy()
    errors = []
    for row in rows:
        theoretical = ADDUCTS[row["adduct"]].mz(exact_mass(parse_formula(row["molecular_formula"])))
        errors.append(abs((row["precursor_mz"]-theoretical)/theoretical*1e6))
    return dict(spectra=len(rows), structures=len(counts), spectra_per_structure_quantiles=np.quantile(counts, [0,.5,.9,1]).tolist(),
        precursor_abs_ppm_quantiles=np.quantile(errors, [.5,.95,.99,1]).tolist(),
        adducts=f.group_by("adduct").len().sort("adduct").to_dicts())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pl.scan_parquet(ROOT / "enveda-CASMI26-molecule-id-mass-spectra/train.parquet").filter(pl.col("ingest_lib")=="enveda-180")
    test = pl.read_parquet(ROOT / "enveda-CASMI26-molecule-id-mass-spectra/test.parquet")
    audit = dict(raw_enveda=raw.select(pl.len().alias("spectra"), pl.col("inchikey14").n_unique().alias("structures"),
        pl.col("base_peak_intensity").is_null().sum().alias("null_base"), (pl.col("base_peak_intensity")<1000).sum().alias("low_base")).collect().to_dicts()[0],
        raw_instruments=raw.group_by("instrument_type").len().collect().to_dicts(),
        public_test_instruments=test.group_by("instrument_type").len().to_dicts(),
        public_test_spectra=test.height, public_test_structures=test["molecule_id"].n_unique(),
        prepared_source_counts={})
    for split in ("train", "validation", "test"):
        audit["prepared_source_counts"][split] = pl.scan_parquet(PREPARED / split / "*.parquet").filter(SOURCE).select(
            pl.len().alias("spectra"), pl.col("molecule_id").n_unique().alias("structures")).collect().to_dicts()[0]
    ids = sorted(pl.scan_parquet(PREPARED / "test/*.parquet").filter(SOURCE).select("molecule_id").unique().collect()["molecule_id"].to_list())
    rounds = {str(seed): random.Random(seed).sample(ids, 100) for seed in range(101, 106)}
    sampled = set().union(*map(set, rounds.values()))
    old = {r["molecule_id"]: r for r in map(json.loads, (ROOT / "artifacts/reports/baseline-sampling/records.jsonl").read_text().splitlines())}
    paired = set(ids) & set(old)
    selected = sampled | paired
    config = FormulaConfig.load(PREPARED / "config.json")
    manifest = dict(protocol_version=1, source="enveda-180", instrument="timsTOF", population=len(ids), rounds=rounds,
        sampled_unique=len(sampled), historical_paired_ids=sorted(paired), evaluated_unique=len(selected), config_version=config.version,
        config=config.to_dict(), cleaning=dict(minimum_base_peak_intensity=1000, maximum_peak_mz="precursor_mz + 2 Da", inherited_precursor_qc_ppm=20),
        note="Source/instrument matched, not chemistry or full spectral-count distribution matched. Same structure split; no new model training or algorithm changes.")
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if (OUT / "manifest.json").exists():
        assert json.loads((OUT / "manifest.json").read_text())["identity"] == identity, "Experiment changed: use a new output directory"
    manifest["identity"] = identity
    write_json(OUT / "manifest.json", manifest)
    source_rows = []
    for path in sorted((PREPARED / "test").glob("*.parquet")):
        source_rows.extend(pl.read_parquet(path).filter(SOURCE & pl.col("molecule_id").is_in(sorted(selected))).to_dicts())
    clean, stats = clean_rows(source_rows)
    schema = pl.read_parquet(next((PREPARED / "test").glob("*.parquet")), n_rows=0).schema
    pl.DataFrame(source_rows, schema=schema).write_parquet(OUT / "source_only.parquet")
    pl.DataFrame(clean, schema=schema).write_parquet(OUT / "test_cleaned.parquet")
    audit["cleaning"] = stats
    audit["sampled_source_distribution"] = distribution([r for r in source_rows if r["molecule_id"] in sampled])
    audit["sampled_clean_distribution"] = distribution([r for r in clean if r["molecule_id"] in sampled])
    for split in ("train", "validation"):
        others=set(pl.scan_parquet(PREPARED / split / "*.parquet").select("molecule_id").unique().collect()["molecule_id"].to_list())
        assert not selected & others
    audit["train_and_validation_structure_overlap"] = 0
    write_json(OUT / "audit.json", audit)
    by_id = {}
    for row in source_rows:
        by_id.setdefault(row["molecule_id"], []).append(row)
    existing = OUT / "records.jsonl"
    records = {(r["arm"], r["molecule_id"]):r for r in map(json.loads, existing.read_text().splitlines())} if existing.exists() else {}
    jobs=[]
    aliases=[]
    for key, rows in sorted(by_id.items()):
        group=group_from(rows)
        cleaned, counts=clean_rows(rows)
        group_clean=group_from(cleaned) if cleaned else {**group, "spectra":[]}
        jobs.append(("source_only",group))
        if counts["removed_high_peaks"] == 0 and counts["removed_low_or_missing_base"] == 0:
            aliases.append(key)
        else:
            jobs.append(("test_cleaned",group_clean))
    start=time.perf_counter()
    print(f"sampled={len(sampled)} historical_paired={len(paired)} union={len(selected)} jobs={len(jobs)} identical_cleaned={len(aliases)}",flush=True)
    with existing.open("a",encoding="utf-8") as log, ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run_task, arm, group):(arm,group["molecule_id"]) for arm,group in jobs if (arm,group["molecule_id"]) not in records}
        for future in as_completed(futures):
            record=future.result()
            records[(record["arm"],record["molecule_id"])]=record
            log.write(json.dumps(record)+"\n"); log.flush()
            print(f"{len(records)} {record['arm']} {record['molecule_id']} rank={record['rank']} {record['status']}",flush=True)
        for key in aliases:
            if ("test_cleaned",key) not in records:
                record={**records[("source_only",key)],"arm":"test_cleaned","reused_identical_input":True}
                records[("test_cleaned",key)]=record
                log.write(json.dumps(record)+"\n")
    assert len(records)==2*len(selected)
    summary=dict(elapsed_seconds=time.perf_counter()-start, rounds={}, unique={}, historical_paired={}, mass_strata={})
    for arm in ("source_only","test_cleaned"):
        summary["rounds"][arm]={seed:metrics([records[(arm,key)] for key in keys]) for seed,keys in rounds.items()}
        summary[arm]=metrics([records[(arm,key)] for keys in rounds.values() for key in keys])
        summary["unique"][arm]=metrics([records[(arm,key)] for key in sorted(sampled)])
        summary["historical_paired"][arm]=metrics([records[(arm,key)] for key in sorted(paired)])
        rs=[records[(arm,key)] for key in sorted(sampled)]
        summary["mass_strata"][arm]={label:metrics(sub) for label,lo,hi in [("<300",0,300),("300-600",300,600),(">=600",600,float("inf"))] if (sub:=[r for r in rs if lo<=r["mass"]<hi])}
    summary["historical_paired"]["original_mixed"]=metrics([old[key] for key in sorted(paired)])
    summary["paired_rank_changes_after_cleaning"]=sum(records[("source_only",key)]["rank"]!=records[("test_cleaned",key)]["rank"] for key in sampled)
    summary["historical_original_overall"]=json.loads((ROOT / "artifacts/reports/baseline-sampling/summary.json").read_text())["pooled_rounds"]
    rng=np.random.default_rng(2026)
    for label,left,right in [("clean_minus_source", "test_cleaned","source_only"),("source_minus_mixed", "source_only","original_mixed")]:
        keys=sorted(sampled if label=="clean_minus_source" else paired)
        def rr(record):
            return 1/record["rank"] if record["rank"] else 0
        delta=np.array([rr(records[(left,k)])-rr(old[k] if right=="original_mixed" else records[(right,k)]) for k in keys])
        summary[label]=dict(mean_mrr_difference=float(delta.mean()),paired_bootstrap_95ci=np.quantile(rng.choice(delta,(10000,len(delta))).mean(axis=1),[.025,.975]).tolist(),n=len(keys))
    write_json(OUT / "summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
