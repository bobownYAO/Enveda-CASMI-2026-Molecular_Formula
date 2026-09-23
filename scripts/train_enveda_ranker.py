"""Train a source-matched ranker with parallel, resumable standard feature caches.

Reuses casmi.formula.training's sampling, LambdaRank training and feature schema.
No changes to candidate generation or chemical filtering are made in this run.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import hashlib
import json
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from casmi.formula import FormulaConfig, FormulaPredictor
from casmi.formula.data import write_json
from casmi.formula.features import FEATURE_NAMES
from casmi.formula.training import _cache, _matrix, train, ranking_metrics
from evaluate_enveda_alignment import clean_rows, SOURCE

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"artifacts/prepared-enveda-ranker-v1"
MODEL=ROOT/"artifacts/formula/enveda-ranker-v1"
REPORT=ROOT/"artifacts/reports/enveda-ranker-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def identity(prepared,split):
    manifest=json.loads((prepared/"manifest.json").read_text())
    config=FormulaConfig.load(prepared/"config.json")
    return dict(cache_version=4,data=manifest["file_fingerprint"],config=config.version,split=split,limit=None,feature_names=FEATURE_NAMES)


def build_cache(prepared,split):
    folder,meta=_cache(Path(prepared),split,FormulaPredictor.baseline(FormulaConfig.load(Path(prepared)/"config.json")))
    return str(folder),meta


def prepare_samples():
    if (DATA/"ready.json").exists():
        return json.loads((DATA/"selection.json").read_text())
    DATA.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    original=ROOT/"artifacts/prepared"
    previous=json.loads((ROOT/"artifacts/reports/enveda-alignment/manifest.json").read_text())
    benchmark=sorted(set().union(*map(set,previous["rounds"].values())))
    excluded=set(previous["historical_paired_ids"])|set(benchmark)
    ids={split:sorted(pl.scan_parquet(original/split/"*.parquet").filter(SOURCE).select("molecule_id").unique().collect()["molecule_id"].to_list()) for split in ("train","validation","test")}
    selected=dict(train=random.Random(202601).sample(ids["train"],3000),validation=random.Random(202602).sample(ids["validation"],500),
        test=random.Random(202603).sample(sorted(set(ids["test"])-excluded),500))
    for left,right in (("train","validation"),("train","test"),("validation","test")):
        assert not set(selected[left])&set(selected[right])
    assert not set(selected["test"])&excluded
    assert not set(benchmark)&(set(selected["train"])|set(selected["validation"]))
    all_selected={**selected,"test":selected["test"]+benchmark}
    selection=dict(seeds=dict(train=202601,validation=202602,test=202603),ids=selected,benchmark_ids=benchmark,populations={k:len(v) for k,v in ids.items()},
        source="enveda-180",instrument="timsTOF",cleaning="base_peak>=1000 and peak_mz<=precursor_mz+2; inherited prepared QC",source_manifest=json.loads((original/"manifest.json").read_text()),
        config=FormulaConfig.load(original/"config.json").to_dict(),note="First 3000-structure training run, not full-corpus training. Fresh test excludes all previously evaluated Enveda alignment structures.")
    fingerprint=digest(selection)
    write_json(DATA/"selection.json",selection)
    write_json(DATA/"manifest.json",dict(file_fingerprint=fingerprint,selection="selection.json"))
    shutil.copy2(original/"config.json",DATA/"config.json")
    counts={}
    for split,keys in all_selected.items():
        rows=[]
        for path in sorted((original/split).glob("*.parquet")):
            rows.extend(pl.read_parquet(path).filter(SOURCE&pl.col("molecule_id").is_in(keys)).to_dicts())
        cleaned,stats=clean_rows(rows)
        frame=pl.DataFrame(cleaned,schema=pl.read_parquet(next((original/split).glob("*.parquet")),n_rows=0).schema)
        assert set(frame["molecule_id"])==set(keys)
        (DATA/split).mkdir(exist_ok=True)
        frame.write_parquet(DATA/split/"part-000.parquet")
        counts[split]={**stats,"structures":len(keys)}
        if split=="train":
            write_json(DATA/"train_formulas.json",sorted(frame["molecular_formula"].unique().to_list()))
        # Independent shard directories preserve standard cache identity and resume.
        for index in range(0,len(keys),100):
            shard=DATA/"shards"/f"{split}-{index//100:03d}"
            (shard/split).mkdir(parents=True,exist_ok=True)
            part=frame.filter(pl.col("molecule_id").is_in(keys[index:index+100]))
            part.write_parquet(shard/split/"part-000.parquet")
            shutil.copy2(DATA/"config.json",shard/"config.json")
            write_json(shard/"manifest.json",dict(file_fingerprint=digest([fingerprint,split,index,keys[index:index+100]])))
    write_json(DATA/"ready.json",dict(counts=counts,structure_overlap=0,source_fingerprint=fingerprint))
    return selection


def cache_all(max_workers=4):
    tasks=[]
    for split in ("train","validation","test"):
        ident=identity(DATA,split)
        target=DATA/"feature_cache"/digest(ident)[:20]
        if not (target/"manifest.json").exists():
            tasks.extend((p,split) for p in sorted((DATA/"shards").glob(f"{split}-*")))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures={pool.submit(build_cache,str(path),split):(path,split) for path,split in tasks}
        for future in as_completed(futures):
            path,split=futures[future]
            folder,meta=future.result()
            print(f"SHARD_COMPLETE {path.name} groups={meta['groups']} records={len(meta['records'])}",flush=True)
    for split in ("train","validation","test"):
        ident=identity(DATA,split)
        target=DATA/"feature_cache"/digest(ident)[:20]
        if (target/"manifest.json").exists():
            continue
        target.mkdir(parents=True,exist_ok=True)
        files=[];records=[];groups=0
        for shard in sorted((DATA/"shards").glob(f"{split}-*")):
            folder=shard/"feature_cache"/digest(identity(shard,split))[:20]
            meta=json.loads((folder/"manifest.json").read_text())
            assert meta["config"]==ident["config"] and meta["feature_names"]==ident["feature_names"]
            records.extend(meta["records"])
            groups+=meta["groups"]
            for file in meta["files"]:
                name=f"features-{len(files):05d}.npz"
                shutil.copy2(folder/file,target/name)
                files.append(name)
        assert len(records)==len({r['molecule_id'] for r in records})
        expected=set(pl.read_parquet(DATA/split/"part-000.parquet",columns=["molecule_id"])["molecule_id"])
        assert {r['molecule_id'] for r in records}==expected
        write_json(target/"manifest.json",{**ident,"files":files,"records":records,"groups":groups})
        print(f"CACHE_MERGED {split} records={len(records)} groups={groups}",flush=True)


def measure(records,field="rank"):
    ranks=[r[field] for r in records]
    return {**ranking_metrics(ranks),"mrr_full":sum(1/r for r in ranks if r is not None)/len(ranks),
        "resource_failures":sum(r.get("status")=="resource_limit" for r in records),"truncated":sum(r.get("truncated",False) for r in records)}


def evaluate_cache(split,predictor):
    folder=DATA/"feature_cache"/digest(identity(DATA,split))[:20]
    meta=json.loads((folder/"manifest.json").read_text())
    x,y,groups=_matrix(folder,meta)
    assert sum(groups)==len(y)==len(x)
    predictions=predictor.model.predict(x,num_threads=4)
    records=[];offset=0;group_index=0
    for original in meta["records"]:
        record=dict(original)
        n=record["candidate_count"]
        if n:
            assert n==groups[group_index]
            order=np.argsort(-predictions[offset:offset+n],kind="stable")
            positive=np.flatnonzero(y[offset:offset+n][order])
            record["rank"]=int(positive[0])+1 if len(positive) else None
            offset+=n;group_index+=1
        else:
            record["rank"]=None
        records.append(record)
    assert offset==len(y) and group_index==len(groups)
    return records


def main():
    start=time.perf_counter()
    selection=prepare_samples()
    print("DATA_READY train=3000 validation=500 fresh_test=500 benchmark=491",flush=True)
    cache_all()
    if not (MODEL/"training_report.json").exists():
        print("TRAINING_LIGHTGBM",flush=True)
        train(DATA,MODEL,n_estimators=1000)
    predictor=FormulaPredictor.load(MODEL)
    metadata=json.loads((MODEL/"metadata.json").read_text())
    metadata["data_scope"]={"source":"enveda-180","instrument":"timsTOF","training_requested":3000,"validation_requested":500,"selection_manifest":str(DATA/"selection.json"),"cleaning":selection["cleaning"]}
    scope_limit="Trained on 3000 enveda-180 timsTOF structures; natural-product and other-instrument performance is not established by this evaluation"
    if scope_limit not in metadata["limitations"]:
        metadata["limitations"].append(scope_limit)
    write_json(MODEL/"metadata.json",metadata)
    validation=evaluate_cache("validation",predictor)
    test=evaluate_cache("test",predictor)
    fresh=set(selection["ids"]["test"])
    benchmark=set(selection["benchmark_ids"])
    partitions=dict(validation=validation,fresh_test=[r for r in test if r["molecule_id"] in fresh],previous_benchmark=[r for r in test if r["molecule_id"] in benchmark])
    assert len(partitions["fresh_test"])==500 and len(partitions["previous_benchmark"])==491
    summary=dict(model_version=predictor.model_version,config_version=predictor.config.version,training=json.loads((MODEL/"training_report.json").read_text()),metrics={},elapsed_seconds=time.perf_counter()-start)
    for name,records in partitions.items():
        summary["metrics"][name]={"learned":measure(records),"baseline":measure(records,"baseline_rank")}
        write_json(REPORT/f"{name}-records.json",records)
    known=set(metadata["training_formulas"])
    summary["fresh_formula_strata"]={name:measure(rs) for name,flag in [("seen",True),("unseen",False)] if (rs:=[r for r in partitions["fresh_test"] if (r["formula"] in known)==flag])}
    old={r["molecule_id"]:r for r in map(json.loads,(ROOT/"artifacts/reports/enveda-alignment/records.jsonl").read_text().splitlines()) if r["arm"]=="test_cleaned"}
    assert all(r["baseline_rank"]==old[r["molecule_id"]]["rank"] for r in partitions["previous_benchmark"])
    # Verify saved-model public API ranks agree with full cached inference.
    checks=[]
    frame=pl.read_parquet(DATA/"test/part-000.parquet")
    for record in sorted(partitions["fresh_test"],key=lambda r:r["molecule_id"])[:3]:
        rows=frame.filter(pl.col("molecule_id")==record["molecule_id"]).to_dicts()
        result=predictor.predict(rows,molecule_id=record["molecule_id"],top_k=10000)
        rank=next((c.rank for c in result.candidates if c.formula==record["formula"]),None)
        assert rank==record["rank"]
        checks.append(dict(molecule_id=record["molecule_id"],rank=rank,status=result.status))
    summary["verification"]=dict(public_api_cached_rank_agreement=checks,previous_491_baseline_ranks_identical=True,structure_overlap=0)
    # Importance describes this fitted model, not causal evidence about the spectra.
    summary["feature_importance_gain"]=sorted(zip(FEATURE_NAMES,predictor.model.feature_importance(importance_type="gain").tolist()),key=lambda item:-item[1])[:15]
    rng=np.random.default_rng(202604)
    rr=np.array([1/r["rank"] if r["rank"] else 0 for r in partitions["fresh_test"]])
    summary["fresh_mrr_full_bootstrap_95ci"]=np.quantile(rng.choice(rr,(10000,len(rr))).mean(axis=1),[.025,.975]).tolist()
    write_json(REPORT/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
