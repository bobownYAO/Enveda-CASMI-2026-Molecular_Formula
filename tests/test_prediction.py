"""公共预测器的输入边界、特征一致性和资源状态测试。"""

import json

import numpy as np
import pytest

from casmi.formula import FormulaConfig, FormulaPredictor
from casmi.formula.spectra import normalize_spectrum
from casmi.formula.features import fragment_features
from casmi.formula.chemistry import parse_formula
from casmi.formula.candidates import SearchLimitExceeded


def glucose(**updates):
    return dict(precursor_mz=181.07066456, adduct='[M+H]+',
                ms2_mzs=[163.06009988,145.04953520],
                ms2_normalized_intensities=[1.0,0.4], **updates)


def config(**kw):
    return FormulaConfig(element_limits={'C':10,'H':24,'N':2,'O':8}, **kw)


def test_predict_unseen_formula_and_duplicate_spectra_do_not_change_evidence():
    predictor=FormulaPredictor.baseline(config())
    one=predictor.predict([glucose()], molecule_id='g')
    two=predictor.predict([glucose(),glucose(spectrum_id='another')],molecule_id='g')
    assert one.status=='ok'
    assert one.candidates[0].formula=='C6H12O6'
    assert one.candidates==two.candidates
    assert one.candidates[0].supporting_spectra==1
    assert json.loads(one.to_json())['molecule_id']=='g'


@pytest.mark.parametrize('field,value',[
    ('precursor_mz',float('nan')),('ms2_mzs',[float('inf')]),
    ('ms2_normalized_intensities',[-1,2]),('ms2_mzs',[1]),
    ('ms2_mzs',[0,1]),('ionization_mode','negative'),('instrument_type',123)])
def test_invalid_spectrum_is_an_explicit_result(field,value):
    row=glucose(); row[field]=value
    r=FormulaPredictor.baseline(config()).predict([row])
    assert r.status=='invalid_input'
    assert not r.candidates


def test_unsupported_and_missing_fragment_evidence():
    p=FormulaPredictor.baseline(config())
    row=glucose();row['adduct']='[2M+H]+'
    assert p.predict([row]).status=='unsupported_input'
    row=glucose();row['ms2_mzs']=[];row['ms2_normalized_intensities']=[]
    r=p.predict([row]);assert r.candidates
    assert any('fragment' in w for w in r.warnings)


def test_cleaning_merges_exact_duplicates_before_normalization():
    row=glucose(); row['ms2_mzs']=[100,50,100,80]; row['ms2_normalized_intensities']=[2,1,2,.001]
    s=normalize_spectrum(row,config())
    assert s.mzs.tolist()==[50,100]
    assert s.intensities.tolist()==[.25,1]
    assert len(s.raw_mzs)==4


def test_conflicting_precursors_use_union_and_report_warning():
    p=FormulaPredictor.baseline(config())
    row=glucose(); row['precursor_mz']=151.0601
    r=p.predict([glucose(),row],top_k=100)
    assert any(c.formula=='C6H12O6' for c in r.candidates)
    assert any('inconsistent' in w for w in r.warnings)


def test_truncation_and_empty_candidates_are_visible():
    p=FormulaPredictor.baseline(config(max_candidates=1,precursor_ppm=1000))
    r=p.predict([glucose()])
    assert r.status=='truncated'
    assert len(r.candidates)==1
    row=glucose();row['precursor_mz']=1.1
    assert p.predict([row]).status=='no_candidates'


def test_metadata_leakage_is_ignored_and_model_missing_is_error(tmp_path):
    p=FormulaPredictor.baseline(config())
    assert p.predict([glucose()]).candidates==p.predict([glucose(molecular_formula='WRONG',precursor_error_ppm=0)]).candidates
    with pytest.raises(FileNotFoundError):
        FormulaPredictor.load(tmp_path/'absent')


@pytest.mark.parametrize('adduct,mz,peak',[
    ('[M+Na]+',203.05260881,185.04204413),
    ('[M-H]-',179.05611163,161.04554695),
    ('[M+Cl]-',215.03278936,34.96940126),
])
def test_fragment_atom_balance_handles_metals_and_negative_ions(adduct,mz,peak):
    row={'precursor_mz':mz,'adduct':adduct,'ms2_mzs':[peak],'ms2_normalized_intensities':[1.]}
    spectrum=normalize_spectrum(row,config())
    evidence=fragment_features(np.array([parse_formula('C6H12O6')]),spectrum,config())
    assert evidence[0,0]==1 and evidence[0,1]==1


def test_precursor_ppm_uses_theoretical_denominator_at_both_boundaries():
    p=FormulaPredictor.baseline(FormulaConfig(element_limits={'C':6,'H':12,'O':6}))
    theoretical=181.07066455650093
    for sign in [-1,1]:
        row=glucose();row['precursor_mz']=theoretical*(1+sign*10e-6)
        r=p.predict([row])
        assert any(c.formula=='C6H12O6' and c.supporting_spectra==1 for c in r.candidates)


def test_parallel_file_predictions_match_serial(tmp_path):
    import polars as pl
    file=tmp_path/'spectra.parquet'
    pl.DataFrame([dict(glucose(),molecule_id='a'),dict(glucose(),molecule_id='b')]).write_parquet(file)
    p=FormulaPredictor.baseline(config())
    assert [r.candidates for r in p.predict_parquet(file)]==[r.candidates for r in p.predict_parquet(file,workers=2)]


def test_search_budget_failure_is_not_misrepresented_as_complete_candidates():
    p=FormulaPredictor.baseline(config(precursor_ppm=1000,max_search_results=1))
    r=p.predict([glucose()])
    assert r.status=='resource_limit' and not r.candidates


def test_fragment_search_limit_is_reported():
    cfg=config(max_search_results=1,fragment_da=1.)
    row=glucose();row['ms2_mzs']=[100.];row['ms2_normalized_intensities']=[1.]
    with pytest.raises(SearchLimitExceeded):
        fragment_features(np.array([parse_formula('C10H24N2O8')]),normalize_spectrum(row,cfg),cfg)


def test_fragment_matching_agrees_with_exhaustive_subsets():
    import itertools
    vectors=np.array([parse_formula('C6H12O6'),parse_formula('C9H8O4')])
    row=glucose();row['ms2_mzs']=[31.01839,145.04953520,163.06009988,181.07066456]
    row['ms2_normalized_intensities']=[.2,.3,.7,1.]
    spec=normalize_spectrum(row,config());actual=fragment_features(vectors,spec,config())
    for i,v in enumerate(vectors):
        # 用独立的离子原子子集穷举作对照，避免测试复用生产公式辅助函数。
        counts=np.array(list(itertools.product(range(int(v[0])+1),range(int(v[1])+2),range(int(v[3])+1))))
        masses=counts@np.array([12.,1.00782503223,15.99491461957])-.000548579909065
        errors=np.array([np.min(np.abs(masses-peak)) for peak in spec.mzs])
        tolerance=np.maximum(.002,spec.mzs*1e-5)
        hits=errors<=tolerance
        assert actual[i,0]==pytest.approx(hits.mean())
        assert actual[i,1]==pytest.approx(spec.intensities[hits].sum()/spec.intensities.sum())
        assert actual[i,2]==pytest.approx((errors[hits]/tolerance[hits]).mean(),abs=1e-8)


def test_impossible_heavier_fragments_do_not_consume_search_budget():
    cfg=config(max_search_results=1,fragment_da=1.)
    row=glucose();row['ms2_mzs']=[200.];row['ms2_normalized_intensities']=[1.]
    vectors=np.array([parse_formula(f) for f in ['C6H12O6','C13H24','N12O']])
    evidence=fragment_features(vectors,normalize_spectrum(row,cfg),cfg)
    assert (evidence[:,:2]==0).all()
