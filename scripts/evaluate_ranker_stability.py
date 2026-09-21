"""Fixed-model stability across 10 disjoint, previously unused holdout batches."""
import os
os.environ.setdefault("POLARS_MAX_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import hashlib
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import polars as pl
from casmi.formula import FormulaPredictor
from casmi.formula.candidates import SearchLimitExceeded
from casmi.formula.data import write_json
from casmi.formula.training import ranking_metrics
from evaluate_enveda_alignment import SOURCE,clean_rows,group_from

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/reports/enveda-stability-v1"
MODEL=ROOT/"artifacts/formula/enveda-ranker-v1"


def init_worker():
    global predictor,seen_formulas
    predictor=FormulaPredictor.load(MODEL)
    seen_formulas=set(json.loads((MODEL/"metadata.json").read_text())["training_formulas"])


def predict_group(group):
    start=time.perf_counter()
    record={k:v for k,v in group.items() if k!="spectra"}
    record.update(input_spectra=len(group["spectra"]),seen_formula=group["formula"] in seen_formulas)
    try:
        b=predictor.candidate_features(group["spectra"])
        order=sorted(range(len(b['formulas'])),key=lambda i:b['formulas'][i])
        if order:
            scores=predictor.model.predict(b['features'],num_threads=1)
            order=sorted(order,key=lambda i:-scores[i])
        record.update(rank=next((rank for rank,i in enumerate(order,1) if b['formulas'][i]==group['formula']),None),
            candidate_count=len(order),total_candidates=b['total_candidates'],status='truncated' if b['truncated'] else 'ok' if order else 'no_candidates',
            unique_spectra=len(b['spectra']))
    except SearchLimitExceeded:
        record.update(rank=None,candidate_count=0,status='resource_limit')
    record['seconds']=time.perf_counter()-start
    return record


def metrics(records):
    ranks=[r['rank'] for r in records]
    return {**ranking_metrics(ranks),'mrr_full':sum(1/r for r in ranks if r)/len(ranks),
        'statuses':dict(Counter(r['status'] for r in records))}


def previous_ids():
    previous=set()
    for file in ['baseline-sampling/records.jsonl','enveda-alignment/records.jsonl']:
        previous.update(json.loads(line)['molecule_id'] for line in (ROOT/'artifacts/reports'/file).read_text().splitlines())
    for file in ['enveda-ranker-v1/fresh_test-records.json','enveda-ranker-v1/previous_benchmark-records.json','enveda-tuning-v1/new-test-original.json']:
        previous.update(r['molecule_id'] for r in json.loads((ROOT/'artifacts/reports'/file).read_text()))
    previous.update(r['molecule_id'] for r in json.loads((ROOT/'artifacts/reports/pilot-holdout.json').read_text())['records'])
    return previous


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    old=previous_ids()
    available=set(pl.scan_parquet(ROOT/'artifacts/prepared/test/*.parquet').filter(SOURCE).select('molecule_id').unique().collect()['molecule_id'].to_list())
    selected=random.Random(20260923).sample(sorted(available-old),3000)
    rounds={str(i+1):selected[i*300:(i+1)*300] for i in range(10)}
    metadata=json.loads((MODEL/'metadata.json').read_text())
    manifest=dict(seed=20260923,model_version=metadata['model_version'],config_version=metadata['config_version'],
        model_sha256=hashlib.sha256((MODEL/'model.txt').read_bytes()).hexdigest(),source_population=len(available),
        previously_evaluated_source_structures=len(available&old),eligible_population=len(available-old),rounds=rounds,
        protocol='Frozen 31-leaf model, 10 disjoint random batches of 300 structures; no training or parameter changes; failures score zero.',
        source='enveda-180',instrument='timsTOF',cleaning='inherit prepared QC; base_peak>=1000; peaks<=precursor_mz+2')
    if (OUT/'manifest.json').exists():
        assert json.loads((OUT/'manifest.json').read_text())==manifest
    write_json(OUT/'manifest.json',manifest)
    assert len(selected)==len(set(selected))==3000 and not set(selected)&old
    for split in ('train','validation'):
        other=set(pl.scan_parquet(ROOT/f'artifacts/prepared/{split}/*.parquet').select('molecule_id').unique().collect()['molecule_id'])
        assert not set(selected)&other
    rows=[]
    for path in sorted((ROOT/'artifacts/prepared/test').glob('*.parquet')):
        rows.extend(pl.read_parquet(path).filter(SOURCE&pl.col('molecule_id').is_in(selected)).to_dicts())
    cleaned,stats=clean_rows(rows)
    schema=pl.read_parquet(ROOT/'artifacts/prepared-enveda-ranker-v1/test/part-000.parquet',n_rows=0).schema
    frame=pl.DataFrame(cleaned,schema=schema)
    assert set(frame['molecule_id'])==set(selected)
    frame.write_parquet(OUT/'spectra.parquet')
    write_json(OUT/'cleaning.json',stats)
    round_of={key:int(index) for index,keys in rounds.items() for key in keys}
    groups=[{**group_from(g.to_dicts()),'round':round_of[key[0]]} for key,g in frame.group_by('molecule_id')]
    existing=OUT/'records.jsonl'
    records={r['molecule_id']:r for r in map(json.loads,existing.read_text().splitlines())} if existing.exists() else {}
    with existing.open('a',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=4,initializer=init_worker) as pool:
        futures=[pool.submit(predict_group,g) for g in sorted(groups,key=lambda g:g['molecule_id']) if g['molecule_id'] not in records]
        for future in as_completed(futures):
            record=future.result()
            records[record['molecule_id']]=record
            log.write(json.dumps(record)+'\n');log.flush()
            if len(records)%50==0:
                print(f"PROGRESS {len(records)}/3000",flush=True)
    assert set(records)==set(selected)
    rs=[records[key] for key in selected]
    summary=dict(model_version=metadata['model_version'],overall=metrics(rs),rounds={i:metrics([records[key] for key in keys]) for i,keys in rounds.items()},
        strata={},elapsed_seconds=time.perf_counter()-start)
    for label,subset in [('seen',[r for r in rs if r['seen_formula']]),('unseen',[r for r in rs if not r['seen_formula']]),
        ('mass<300',[r for r in rs if r['mass']<300]),('mass300-600',[r for r in rs if 300<=r['mass']<600]),('mass>=600',[r for r in rs if r['mass']>=600]),
        ('single_spectrum',[r for r in rs if r.get('unique_spectra',r['input_spectra'])==1]),('multi_spectrum',[r for r in rs if r.get('unique_spectra',r['input_spectra'])>1])]:
        if subset: summary['strata'][label]=metrics(subset)
    rng=np.random.default_rng(20260924)
    summary['bootstrap_95ci']={}
    for name,values in [('mrr_full',[1/r['rank'] if r['rank'] else 0 for r in rs]),
                        ('mrr25',[1/r['rank'] if r['rank'] and r['rank']<=25 else 0 for r in rs]),('top1',[float(r['rank']==1) for r in rs])]:
        samples=np.concatenate([rng.choice(values,(1000,len(rs))).mean(axis=1) for _ in range(10)])
        summary['bootstrap_95ci'][name]=np.quantile(samples,[.025,.975]).tolist()
    summary['batch_variation']={name:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),minimum=float(min(v)),maximum=float(max(v))) for name in ['mrr_full','mrr25','top1'] if (v:=[x[name] for x in summary['rounds'].values()])}
    # Two prior 500-structure fresh holdouts are disjoint, so cumulative metrics are interpretable.
    earlier1=json.loads((ROOT/'artifacts/reports/enveda-ranker-v1/fresh_test-records.json').read_text())
    earlier2=json.loads((ROOT/'artifacts/reports/enveda-tuning-v1/new-test-original.json').read_text())
    combined=rs+[{**r,'status':r.get('status','truncated' if r.get('truncated') else 'ok')} for r in earlier1+earlier2]
    assert len({r['molecule_id'] for r in combined})==4000
    summary['cumulative_4000']=metrics(combined)
    # Save-load/public interface check, deliberately without truth columns.
    from casmi.formula.data import INFERENCE_COLUMNS
    predictor=FormulaPredictor.load(MODEL)
    checks=[]
    for key in sorted(selected)[:3]:
        spectra=[{k:v for k,v in row.items() if k in INFERENCE_COLUMNS+('molecule_id','spectrum_id')} for row in frame.filter(pl.col('molecule_id')==key).to_dicts()]
        result=predictor.predict(spectra,molecule_id=key,top_k=10000)
        rank=next((c.rank for c in result.candidates if c.formula==records[key]['formula']),None)
        assert rank==records[key]['rank']
        checks.append(dict(molecule_id=key,rank=rank))
    assert hashlib.sha256((MODEL/'model.txt').read_bytes()).hexdigest()==manifest['model_sha256']
    summary['verification']=dict(model_unchanged=True,train_validation_previous_overlap=0,public_api_checks=checks)
    write_json(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
