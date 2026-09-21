"""数据准备、训练来源、CLI 配置和模型持久化的集成测试。"""

import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from casmi.formula import FormulaConfig, FormulaPredictor
from casmi.formula.data import audit, prepare, iter_groups, split_for
from casmi.formula.training import train, evaluate, ranking_metrics


def row(key,formula='C6H12O6',mz=181.07066456,adduct='[M+H]+'):
    return dict(inchikey14=key,molecular_formula=formula,normalized_smiles='OCC(O)C(O)C(O)C(O)CO',
                ingest_lib='fixture',precursor_mz=mz,adduct=adduct,ionization_mode='positive',
                ms2_mzs=[163.06009988,145.0495352],ms2_normalized_intensities=[1.,.4],
                precursor_error_ppm=0.,collision_energy_ev=None,base_peak_intensity=None,instrument_type=None)


def test_prepare_quarantines_groups_filters_errors_and_derives_limits_only_from_train(tmp_path):
    train_key=next(f'key{i}' for i in range(100) if split_for(f'key{i}')=='train')
    path=tmp_path/'raw.parquet'
    pl.DataFrame([row(train_key),row('conflict'),row('conflict','C5H10O5',151.0601),
                  row('charged','C6H13O6+'),row('error',mz=190),row('unsupported',adduct='[2M+H]+')]).write_parquet(path)
    report=audit(path)
    assert report['rows']==6 and report['conflicting_structures']==1
    out=tmp_path/'prepared';report=prepare(path,out,buckets=2)
    assert report['accepted_spectra']==1
    groups=list(iter_groups(out,'train'))
    assert len(groups)==1 and groups[0]['molecule_id']==train_key
    assert groups[0]['spectra'][0]['spectrum_id'].startswith(report['file_fingerprint'][:16])
    cfg=FormulaConfig.load(out/'config.json')
    assert cfg.element_limits['C']==8 and cfg.element_limits['H']==15
    assert cfg.element_limits['N']==0
    ids=[{g['molecule_id'] for g in iter_groups(out,s)} for s in ['train','validation','test']]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]


def test_rank_metrics_count_missed_candidates_in_denominator():
    m=ranking_metrics([1,2,None,30])
    assert m['top1']==.25 and m['top5']==.5
    assert m['mrr25']==pytest.approx(.375)
    assert m['candidate_recall']==.75


def test_cli_honors_partial_config_and_protects_source_file(tmp_path):
    path=tmp_path/'input.parquet'
    pl.DataFrame([dict(row('m'),molecule_id='m')]).write_parquet(path)
    original=path.read_bytes()
    config=tmp_path/'partial.json';config.write_text(json.dumps({'max_search_results':1}))
    out=tmp_path/'out.jsonl'
    command=[sys.executable,'-m','casmi','formula','predict','--input',str(path),'--config',str(config)]
    run=subprocess.run(command+['--output',str(out)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads(out.read_text())['status']=='resource_limit'
    run=subprocess.run(command+['--output',str(path)],capture_output=True,text=True)
    assert run.returncode==2
    assert path.read_bytes()==original
    config.write_text('{"precusor_ppm": 5}')
    run=subprocess.run(command+['--output',str(out)],capture_output=True,text=True)
    assert run.returncode==2


def test_training_save_load_evaluation_and_cli(tmp_path):
    rows=[]
    for i in range(100):
        key=f'mol{i}'
        # 多个分子式与同质量干扰项确保这里确实测试候选排序，而不是单一命中。
        if i%2: rows.append(row(key))
        else: rows.append(row(key,'C5H10O5',151.06009988))
    raw=tmp_path/'train.parquet';pl.DataFrame(rows).write_parquet(raw)
    prepared=tmp_path/'prepared';prepare(raw,prepared,FormulaConfig(precursor_ppm=1000),buckets=2)
    model=tmp_path/'model'
    report=train(prepared,model,max_train_molecules=30,max_validation_molecules=10,n_estimators=10)
    assert report['training_groups']>0
    predictor=FormulaPredictor.load(model)
    r=predictor.predict([row('example')])
    assert r.candidates
    second=FormulaPredictor.load(model).predict([row('example')])
    assert r.candidates==second.candidates
    # seen/unseen 必须表示模型实际拟合过的分子式，不能把整个训练池误当成模型来源。
    (prepared/'train_formulas.json').write_text('[]')
    metrics=evaluate(prepared,model=model,split='test',max_molecules=5)
    assert metrics['molecules']>0 and 'mrr25' in metrics
    assert metrics['strata']['formula:seen']['molecules']==metrics['molecules']
    testfile=tmp_path/'test.parquet';pl.DataFrame([dict(row('example'),molecule_id='example')]).write_parquet(testfile)
    outfile=tmp_path/'predictions.jsonl'
    proc=subprocess.run([sys.executable,'-m','casmi','formula','predict','--input',str(testfile),
                         '--output',str(outfile),'--model',str(model)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    output=json.loads(outfile.read_text())
    assert output['candidates']==r.to_dict()['candidates']
    override=tmp_path/'override.json';override.write_text('{"precursor_ppm": 5}')
    run=subprocess.run([sys.executable,'-m','casmi','formula','predict','--input',str(testfile),
                        '--output',str(outfile),'--model',str(model),'--config',str(override)],capture_output=True,text=True)
    assert run.returncode==2 and 'config' in run.stderr.lower()
    meta=json.loads((model/'metadata.json').read_text());meta['artifact_version']=999
    (model/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError):FormulaPredictor.load(model)
