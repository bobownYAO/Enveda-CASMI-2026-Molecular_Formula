"""化学质量、加合物换算和候选枚举的基础不变量测试。"""

import itertools

import numpy as np
import pytest

from casmi.formula.chemistry import ADDUCTS, exact_mass, parse_formula, format_formula
from casmi.formula.candidates import FormulaEnumerator


@pytest.mark.parametrize('name,mz', [
    ('[M+H]+',181.07066456), ('[M+NH4]+',198.09721366),
    ('[M-H2O+H]+',163.06009988), ('[M-2H2O+H]+',145.04953520),
    ('[M+Na]+',203.05260881), ('[M+K]+',219.02654601),
    ('[M-H]-',179.05611163), ('[M-H2O-H]-',161.04554695),
    ('[M+CH2O2-H]-',225.06159093), ('[M+Cl]-',215.03278936),
])
def test_adduct_masses_against_glucose_reference(name, mz):
    mass = exact_mass(parse_formula('C6H12O6'))
    assert mass == pytest.approx(180.063388104, abs=1e-8)
    assert ADDUCTS[name].mz(mass) == pytest.approx(mz, abs=2e-7)
    assert ADDUCTS[name].neutral_mass(mz) == pytest.approx(mass, abs=2e-7)


@pytest.mark.parametrize('formula',['C6H12O6+', 'C0H2', 'C6H12O6junk', 'NaCl', '', 'C1.5H3'])
def test_invalid_or_out_of_scope_formula_not_silently_rewritten(formula):
    with pytest.raises(ValueError):
        parse_formula(formula)


def test_formula_canonicalization():
    assert format_formula(parse_formula('H12O6C6')) == 'C6H12O6'


def test_mass_search_agrees_with_independent_small_brute_force():
    engine = FormulaEnumerator({'C':4,'H':10,'N':2,'O':3})
    target, tolerance = 60.05, 0.06
    actual = {format_formula(v) for v in engine.search(target,tolerance)}
    expected = set()
    for c,h,n,o in itertools.product(range(5),range(11),range(3),range(4)):
        mass = c*12+h*1.00782503223+n*14.00307400443+o*15.99491461957
        if target-tolerance <= mass <= target+tolerance and c+h+n+o:
            expected.add(''.join(s+(str(k) if k>1 else '') for s,k in [('C',c),('H',h),('N',n),('O',o)] if k))
    assert actual == expected


def test_search_can_generate_formula_without_a_formula_database():
    engine = FormulaEnumerator({'C':8,'H':20,'O':8})
    assert 'C6H12O6' in {format_formula(v) for v in engine.search(180.063388104, .001)}
    assert len(engine.search(-10,.001)) == 0


def test_random_mass_windows_with_halogens_match_brute_force():
    limits={'C':3,'H':6,'N':1,'O':2,'P':1,'S':1,'F':2,'Cl':1,'Br':1,'I':1}
    vectors=np.array(list(itertools.product(*[range(limits[e]+1) for e in
                         ('C','H','N','O','P','S','F','Cl','Br','I')])))
    reference_masses=vectors@np.array([12.,1.00782503223,14.00307400443,15.99491461957,
                         30.97376199842,31.9720711744,18.99840316273,34.968852682,78.9183376,126.9044719])
    rng=np.random.default_rng(371)
    engine=FormulaEnumerator(limits)
    for target in rng.choice(reference_masses[reference_masses>1],size=12,replace=False):
        expected={tuple(v) for v in vectors[np.abs(reference_masses-target)<=.003]}
        assert {tuple(v) for v in engine.search(float(target),.003)}==expected
