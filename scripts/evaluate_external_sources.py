"""Frozen Enveda ranker on source-isolated, structure-held-out library samples."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
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
from casmi.formula.chemistry import ADDUCTS,exact_mass,parse_formula
from casmi.formula.data import write_json
from casmi.formula.training import ranking_metrics
from evaluate_enveda_alignment import group_from

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reports/external-sources-v1'
MODEL=ROOT/'artifacts/formula/enveda-ranker-v1'


def clean(rows):
    result=[]
    stats=dict(input_spectra=len(rows),missing_base=0,dropped_known_low_base=0,input_peaks=0,removed_high_peaks=0)
    for row in rows:
        if row['base_peak_intensity'] is None:
            stats['missing_base']+=1
        elif row['base_peak_intensity']<1000:
            stats['dropped_known_low_base']+=1
            continue
        mzs=row['ms2_mzs'];intensities=row['ms2_normalized_intensities']
        assert len(mzs)==len(intensities)
        keep=[i for i,mz in enumerate(mzs) if mz<=row['precursor_mz']+2]
        stats['input_peaks']+=len(mzs)
        stats['removed_high_peaks']+=len(mzs)-len(keep)
        result.append({**row,'ms2_mzs':[mzs[i] for i in keep],'ms2_normalized_intensities':[intensities[i] for i in keep]})
    stats['retained_spectra']=len(result)
    return result,stats


def init_worker():
    global predictor,seen
    predictor=FormulaPredictor.load(MODEL)
    seen=set(json.loads((MODEL/'metadata.json').read_text())['training_formulas'])


def evaluate(group):
    start=time.perf_counter()
    record={k:v for k,v in group.items() if k!='spectra'}
    record.update(input_spectra=len(group['spectra']),seen_formula=group['formula'] in seen)
    if not group['spectra']:
        record.update(rank=None,baseline_rank=None,candidate_count=0,status='no_spectra_after_cleaning',seconds=0.)
        return record
    try:
        b=predictor.candidate_features(group['spectra'])
        scores=predictor.model.predict(b['features'],num_threads=1) if b['formulas'] else []
        order=sorted(range(len(scores)),key=lambda i:(-scores[i],b['formulas'][i]))
        baseline=predictor.baseline_order(b)
        def rank(order):
            return next((r for r,i in enumerate(order,1) if b['formulas'][i]==group['formula']),None)
        record.update(rank=rank(order),baseline_rank=rank(baseline),candidate_count=len(order),total_candidates=b['total_candidates'],
            status='truncated' if b['truncated'] else 'ok' if order else 'no_candidates',unique_spectra=len(b['spectra']),warnings=b['warnings'])
    except SearchLimitExceeded:
        record.update(rank=None,baseline_rank=None,candidate_count=0,status='resource_limit')
    record['seconds']=time.perf_counter()-start
    return record


def metrics(rs,field='rank'):
    ranks=[r[field] for r in rs]
    return {**ranking_metrics(ranks),'mrr_full':sum(1/r for r in ranks if r)/len(rs),'statuses':dict(Counter(r['status'] for r in rs))}


def main():
    start=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    base=ROOT/'artifacts/prepared'
    catalog=pl.scan_parquet(base/'test/*.parquet').filter(pl.col('ingest_lib')!='enveda-180').select('molecule_id','ingest_lib').unique().collect()
    selection={}
    for key,g in catalog.group_by('ingest_lib'):
        ids=sorted(g['molecule_id'].to_list())
        seed=int.from_bytes(hashlib.sha256(f'20260925:{key[0]}'.encode()).digest()[:8],'big')
        selection[key[0]]=dict(population=len(ids),seed=seed,ids=random.Random(seed).sample(ids,min(200,len(ids))))
    selection=dict(sorted(selection.items()))
    metadata=json.loads((MODEL/'metadata.json').read_text())
    manifest=dict(model_version=metadata['model_version'],config_version=metadata['config_version'],model_sha256=hashlib.sha256((MODEL/'model.txt').read_bytes()).hexdigest(),
        selection=selection,protocol='Frozen Enveda model. Up to 200 structures per non-Enveda-180 source from fixed test split; all within-source spectra; sources may share held-out structures. Not restricted to never previously evaluated structures.',
        cleaning='Inherit prepared QC including supported neutral labels/adducts and <=20ppm precursor error; retain missing raw base intensity, drop known base<1000; remove peaks>precursor+2.',
        source_coverage=json.loads((base/'manifest.json').read_text())['by_source'])
    if (OUT/'manifest.json').exists():
        assert json.loads((OUT/'manifest.json').read_text())==manifest
    write_json(OUT/'manifest.json',manifest)
    selected=set().union(*(set(s['ids']) for s in selection.values()))
    for split in ('train','validation'):
        others=set(pl.scan_parquet(base/split/'*.parquet').select('molecule_id').unique().collect()['molecule_id'])
        assert not selected&others
    rows=[]
    for path in sorted((base/'test').glob('*.parquet')):
        rows.extend(pl.read_parquet(path).filter((pl.col('ingest_lib')!='enveda-180')&pl.col('molecule_id').is_in(sorted(selected))).to_dicts())
    frame=pl.DataFrame(rows,schema=pl.read_parquet(next((base/'test').glob('*.parquet')),n_rows=0).schema)
    jobs=[];cleaned_rows=[];audit={}
    for source,config in selection.items():
        data=frame.filter((pl.col('ingest_lib')==source)&pl.col('molecule_id').is_in(config['ids']))
        cleaned,stats=clean(data.to_dicts())
        cleaned_rows.extend(cleaned)
        audit[source]={**stats,'structures':len(config['ids']),'instrument_counts':data.group_by('instrument_type').len().to_dicts(),
            'collision_energy_missing_spectra':data['collision_energy_ev'].null_count()}
        for key,g in data.group_by('molecule_id'):
            original=g.to_dicts();kept,_=clean(original)
            group=group_from(kept or original)
            if not kept: group['spectra']=[]
            mass=group['mass'];errors=[]
            for row in kept:
                theoretical=ADDUCTS[row['adduct']].mz(mass)
                errors.append(abs((row['precursor_mz']-theoretical)/theoretical*1e6))
            group.update(source=source,original_spectra=len(original),mass_in_enveda_test_range=237<=mass<=465,
                max_label_abs_ppm=max(errors) if errors else None,minimum_label_abs_ppm=min(errors) if errors else None,
                instruments=sorted({r['instrument_type'] or 'unknown' for r in kept}))
            jobs.append(group)
    pl.DataFrame(cleaned_rows,schema=frame.schema).write_parquet(OUT/'spectra.parquet')
    write_json(OUT/'audit.json',dict(by_source=audit,source_structure_pairs=len(jobs),unique_structures=len(selected),train_validation_overlap=0))
    existing=OUT/'records.jsonl'
    records={(r['source'],r['molecule_id']):r for r in map(json.loads,existing.read_text().splitlines())} if existing.exists() else {}
    print(f"START sources={len(selection)} pairs={len(jobs)} structures={len(selected)}",flush=True)
    with existing.open('a',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=4,initializer=init_worker) as pool:
        futures=[pool.submit(evaluate,g) for g in sorted(jobs,key=lambda g:(g['source'],g['molecule_id'])) if (g['source'],g['molecule_id']) not in records]
        for future in as_completed(futures):
            r=future.result();records[(r['source'],r['molecule_id'])]=r
            log.write(json.dumps(r)+'\n');log.flush()
            if len(records)%25==0:
                print(f"PROGRESS {len(records)}/{len(jobs)} {r['source']} {r['status']}",flush=True)
    assert len(records)==len(jobs)
    rs=list(records.values())
    summary=dict(model_version=metadata['model_version'],sources={},unique_structures=len(selected),source_structure_pairs=len(jobs),elapsed_seconds=time.perf_counter()-start)
    rng=np.random.default_rng(20260926)
    for source in selection:
        subset=sorted([r for r in rs if r['source']==source],key=lambda r:r['molecule_id'])
        values=np.array([1/r['rank'] if r['rank'] else 0 for r in subset])
        stats=dict(learned=metrics(subset),baseline=metrics(subset,'baseline_rank'),mass_range=[min(r['mass'] for r in subset),max(r['mass'] for r in subset)],
            mrr_bootstrap_95ci=np.quantile(rng.choice(values,(10000,len(values))).mean(axis=1),[.025,.975]).tolist(),strata={})
        for label,part in [('seen',[r for r in subset if r['seen_formula']]),('unseen',[r for r in subset if not r['seen_formula']]),
            ('mass237-465',[r for r in subset if r['mass_in_enveda_test_range']]),('mass<300',[r for r in subset if r['mass']<300]),
            ('mass300-600',[r for r in subset if 300<=r['mass']<600]),('mass>=600',[r for r in subset if r['mass']>=600]),
            ('all_spectra_within10ppm',[r for r in subset if r['max_label_abs_ppm'] is not None and r['max_label_abs_ppm']<=10]),
            ('truth_recalled',[r for r in subset if r['rank'] is not None])]:
            if part: stats['strata'][label]=metrics(part)
        summary['sources'][source]=stats
    # Source rows are not independent if structures overlap; do not attach an IID CI to pooled rows.
    summary['pooled_source_pairs_descriptive_only']=metrics(rs)
    summary['verification']=dict(model_unchanged=hashlib.sha256((MODEL/'model.txt').read_bytes()).hexdigest()==manifest['model_sha256'],train_validation_overlap=0)
    assert summary['verification']['model_unchanged']
    write_json(OUT/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
