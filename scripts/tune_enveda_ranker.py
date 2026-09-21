"""Fixed, validation-selected hyperparameter comparison with a new holdout."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import gc
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import lightgbm as lgb
import numpy as np
import polars as pl
from casmi.formula import FormulaPredictor
from casmi.formula.features import FEATURE_NAMES
from casmi.formula.data import write_json
from casmi.formula.training import _matrix, ranking_metrics
from train_enveda_ranker import DATA, identity, digest, build_cache
from evaluate_enveda_alignment import SOURCE, clean_rows

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/reports/enveda-tuning-v1"
MODELS=ROOT/"artifacts/formula/enveda-tuning-v1"
ORIGINAL=ROOT/"artifacts/formula/enveda-ranker-v1"
SELECTED=ROOT/"artifacts/formula/enveda-ranker-tuned-v1"
NEW=ROOT/"artifacts/prepared-enveda-tuning-v1"
TRIALS={
    "leaves15":dict(num_leaves=15),
    "leaves63":dict(num_leaves=63),
    "slow31":dict(learning_rate=.025),
    "regularized31":dict(min_data_in_leaf=50,lambda_l2=1.0),
    "regularized63":dict(num_leaves=63,min_data_in_leaf=50,lambda_l2=1.0),
    "subsample31":dict(feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=1.0),
    "top10rank31":dict(lambdarank_truncation_level=10),
}
BASE=dict(objective="lambdarank",metric="None",num_leaves=31,learning_rate=.05,verbosity=-1,seed=42,
          num_threads=8,deterministic=True,force_col_wise=True)


def load_matrix(split):
    folder=DATA/"feature_cache"/digest(identity(DATA,split))[:20]
    meta=json.loads((folder/"manifest.json").read_text())
    return (*_matrix(folder,meta),meta)


def rank_targets(predictions,labels,groups):
    """O(n) true-label ranks, identical to stable descending argsort."""
    ranks=[];offset=0
    for count in groups:
        scores=predictions[offset:offset+count]
        positive=np.flatnonzero(labels[offset:offset+count])
        assert len(positive)<=1 and np.isfinite(scores).all()
        if len(positive):
            i=int(positive[0]);value=scores[i]
            ranks.append(int(1+np.count_nonzero(scores>value)+np.count_nonzero(scores[:i]==value)))
        else:
            ranks.append(None)
        offset+=count
    assert offset==len(predictions)
    return ranks


def metrics(ranks):
    return {**ranking_metrics(ranks),"mrr_full":sum(1/r for r in ranks if r)/len(ranks)}


def validation_records(predictions,y,groups,meta):
    ranks=iter(rank_targets(predictions,y,groups))
    return [{**r,"rank":next(ranks) if r["candidate_count"] else None} for r in meta["records"]]


def save_model(booster,path,params):
    path.mkdir(parents=True,exist_ok=True)
    booster.save_model(str(path/"model.txt"))
    metadata=json.loads((ORIGINAL/"metadata.json").read_text())
    metadata.update(model_version="lambdarank-"+hashlib.sha256((path/"model.txt").read_bytes()).hexdigest()[:16],
                    best_iteration=booster.best_iteration,hyperparameters=params,tuning_protocol=str(OUT/"protocol.json"))
    write_json(path/"metadata.json",metadata)
    return metadata["model_version"]


def tune():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(trials=TRIALS,base=BASE,early_stopping_rounds=50,max_rounds=1000,slow31_max_rounds=1600,
        selection="validation MRR@25, then Top-1, then fewer trees, then name; no test-based selection",
        training_selection=str(DATA/"selection.json"),new_test_seed=20260921,new_test_size=500)
    if (OUT/"protocol.json").exists():
        assert json.loads((OUT/"protocol.json").read_text())==protocol
    write_json(OUT/"protocol.json",protocol)
    x,y,g,tmeta=load_matrix("train")
    vx,vy,vg,vmeta=load_matrix("validation")
    # Boundary/tie test, independent of any trained model.
    assert rank_targets(np.array([2.,2.,1.,0.,3.]),np.array([0,1,0,0,1]),[3,2])==[2,1]
    original=FormulaPredictor.load(ORIGINAL)
    pred=original.model.predict(vx,num_threads=4)
    ranks=rank_targets(pred,vy,vg)
    # Explicit equivalence check against the pre-existing stable-sort metric.
    offset=0;expected=[]
    for count in vg:
        order=np.argsort(-pred[offset:offset+count],kind="stable")
        found=np.flatnonzero(vy[offset:offset+count][order])
        expected.append(int(found[0])+1 if len(found) else None)
        offset+=count
    assert ranks==expected
    original_score=metrics([r["rank"] for r in validation_records(pred,vy,vg,vmeta)])
    assert abs(original_score["mrr25"]-.8928333333333331)<1e-12
    trials=[dict(name="original31",parameters=BASE,best_iteration=227,validation=original_score,path=str(ORIGINAL),elapsed_seconds=0)]
    def mrr(predictions,dataset):
        value=sum(1/r for r in rank_targets(predictions,vy,vg) if r and r<=25)/len(vmeta["records"])
        return "mrr25",value,True
    for name,changes in TRIALS.items():
        path=MODELS/name
        if (path/"trial.json").exists():
            trial=json.loads((path/"trial.json").read_text())
        else:
            params={**BASE,**changes}
            print(f"TRIAL_START {name} {changes}",flush=True)
            start=time.perf_counter()
            # New Dataset per trial avoids min_data_in_leaf pre-filter carry-over.
            ds=lgb.Dataset(x,label=y,group=g,feature_name=FEATURE_NAMES,params={"min_data_in_leaf":params.get("min_data_in_leaf",20)},free_raw_data=False)
            vs=lgb.Dataset(vx,label=vy,group=vg,feature_name=FEATURE_NAMES,reference=ds,free_raw_data=False)
            booster=lgb.train(params,ds,num_boost_round=1600 if name=="slow31" else 1000,valid_sets=[vs],feval=mrr,
                callbacks=[lgb.early_stopping(50,first_metric_only=True,verbose=False)])
            score=metrics([r["rank"] for r in validation_records(booster.predict(vx,num_threads=4),vy,vg,vmeta)])
            version=save_model(booster,path,params)
            trial=dict(name=name,parameters=params,best_iteration=booster.best_iteration,validation=score,path=str(path),model_version=version,elapsed_seconds=time.perf_counter()-start)
            write_json(path/"trial.json",trial)
            del booster,ds,vs
            gc.collect()
        trials.append(trial)
        write_json(OUT/"trials.json",trials)
        print(f"TRIAL_DONE {name} rounds={trial['best_iteration']} MRR25={trial['validation']['mrr25']:.6f} Top1={trial['validation']['top1']:.3f}",flush=True)
    best=sorted(trials,key=lambda r:(-r["validation"]["mrr25"],-r["validation"]["top1"],r["best_iteration"],r["name"]))[0]
    write_json(OUT/"selected.json",best)
    SELECTED.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(best["path"])/"model.txt",SELECTED/"model.txt")
    metadata=json.loads((Path(best["path"])/"metadata.json").read_text())
    metadata["selection_basis"]={"trial":best["name"],"validation":best["validation"],"protocol":protocol["selection"]}
    write_json(SELECTED/"metadata.json",metadata)
    print(f"SELECTED {best['name']} before any new test scoring",flush=True)
    return best,trials


def new_holdout():
    NEW.mkdir(parents=True,exist_ok=True)
    if not (NEW/"selection.json").exists():
        old=json.loads((DATA/"selection.json").read_text())
        alignment=json.loads((ROOT/"artifacts/reports/enveda-alignment/manifest.json").read_text())
        previous=set(old["ids"]["test"])|set(old["benchmark_ids"])|set(alignment["historical_paired_ids"])
        ids=set(pl.scan_parquet(ROOT/"artifacts/prepared/test/*.parquet").filter(SOURCE).select("molecule_id").unique().collect()["molecule_id"].to_list())
        selected=random.Random(20260921).sample(sorted(ids-previous),500)
        assert not set(selected)&(previous|set(old["ids"]["train"])|set(old["ids"]["validation"]))
        rows=[]
        for path in sorted((ROOT/"artifacts/prepared/test").glob("*.parquet")):
            rows.extend(pl.read_parquet(path).filter(SOURCE&pl.col("molecule_id").is_in(selected)).to_dicts())
        cleaned,stats=clean_rows(rows)
        frame=pl.DataFrame(cleaned,schema=pl.read_parquet(DATA/"test/part-000.parquet",n_rows=0).schema)
        assert set(frame["molecule_id"])==set(selected)
        frame.write_parquet(NEW/"spectra.parquet")
        for index in range(5):
            shard=NEW/f"shard-{index}"
            (shard/"test").mkdir(parents=True,exist_ok=True)
            frame.filter(pl.col("molecule_id").is_in(selected[index*100:(index+1)*100])).write_parquet(shard/"test/part-000.parquet")
            shutil.copy2(DATA/"config.json",shard/"config.json")
            write_json(shard/"manifest.json",dict(file_fingerprint=digest(["tuning-v1",index,selected,stats])))
        write_json(NEW/"selection.json",dict(seed=20260921,ids=selected,excluded_previous_count=len(previous),cleaning=stats,overlap=0))
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(build_cache,str(NEW/f"shard-{i}"),"test") for i in range(5)]
        for future in as_completed(futures):
            folder,meta=future.result()
            print(f"NEW_TEST_FEATURES {len(meta['records'])} records ready",flush=True)


def evaluate_new():
    models=dict(original=FormulaPredictor.load(ORIGINAL),tuned=FormulaPredictor.load(SELECTED))
    records={name:[] for name in models}
    for i in range(5):
        shard=NEW/f"shard-{i}"
        folder=shard/"feature_cache"/digest(identity(shard,"test"))[:20]
        meta=json.loads((folder/"manifest.json").read_text())
        x,y,g=_matrix(folder,meta)
        for name,p in models.items():
            records[name].extend(validation_records(p.model.predict(x,num_threads=4),y,g,meta))
    for name,rs in records.items():
        assert len(rs)==len({r['molecule_id'] for r in rs})==500
        write_json(OUT/f"new-test-{name}.json",rs)
    assert [r['molecule_id'] for r in records['original']]==[r['molecule_id'] for r in records['tuned']]
    summary={name:metrics([r['rank'] for r in rs]) for name,rs in records.items()}
    rng=np.random.default_rng(20260922)
    for metric in ("mrr25","top1"):
        def value(r):
            return (1/r if r and r<=25 else 0) if metric=="mrr25" else int(r==1)
        delta=np.array([value(t['rank'])-value(o['rank']) for o,t in zip(records['original'],records['tuned'])])
        summary[metric+"_paired_difference"]=dict(mean=float(delta.mean()),bootstrap_95ci=np.quantile(rng.choice(delta,(10000,500)).mean(axis=1),[.025,.975]).tolist(),improved=int((delta>0).sum()),worsened=int((delta<0).sum()))
    # Public API parity for the new artifact, with no labels passed as features.
    from casmi.formula.data import INFERENCE_COLUMNS
    frame=pl.read_parquet(NEW/"spectra.parquet")
    checks=[]
    for record in sorted(records['tuned'],key=lambda r:r['molecule_id'])[:3]:
        rows=frame.filter(pl.col('molecule_id')==record['molecule_id']).to_dicts()
        spectra=[{k:v for k,v in row.items() if k in INFERENCE_COLUMNS+("molecule_id","spectrum_id")} for row in rows]
        result=models['tuned'].predict(spectra,molecule_id=record['molecule_id'],top_k=10000)
        rank=next((c.rank for c in result.candidates if c.formula==record['formula']),None)
        assert rank==record['rank']
        checks.append(dict(molecule_id=record['molecule_id'],rank=rank))
    summary['api_checks']=checks
    write_json(OUT/"new-test-summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    tune()
    new_holdout()
    evaluate_new()
