"""验收真实数据产物的一致性；从项目根目录在批量预测后运行。

该脚本检查分子覆盖、排名连续性、候选唯一性、split 隔离和配置版本，属于接口
与产物核验，不使用公开 test 标签衡量模型效果。
"""
import json
from pathlib import Path
from collections import Counter

import numpy as np
import polars as pl

from casmi.formula.data import write_json
from casmi.formula import FormulaConfig


def main():
    """读取现有产物并写出机器可读的 integration 报告。"""
    root=Path('artifacts')
    prediction_file=root/'predictions/baseline-final.jsonl'
    rows=[json.loads(line) for line in prediction_file.read_text(encoding='utf-8').splitlines()]
    expected=set(pl.read_parquet('enveda-CASMI26-molecule-id-mass-spectra/test.parquet',columns=['molecule_id'])['molecule_id'])
    assert len(rows)==400 and {r['molecule_id'] for r in rows}==expected
    config=FormulaConfig.load(root/'prepared/config.json')
    for row in rows:
        candidates=row['candidates']
        assert row['config_version']==config.version
        assert len(candidates)<=25
        assert [c['rank'] for c in candidates]==list(range(1,len(candidates)+1))
        assert len({c['formula'] for c in candidates})==len(candidates)
        assert all(c['supporting_spectra']>=1 for c in candidates)
    ids={split:set(pl.scan_parquet(str(root/'prepared'/split/'*.parquet')).select('molecule_id').unique().collect()['molecule_id'])
         for split in ('train','validation','test')}
    assert not ids['train']&ids['validation'] and not ids['train']&ids['test'] and not ids['validation']&ids['test']
    report={'prediction_file':str(prediction_file),'molecules':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
            'config_version':config.version,'median_candidates':float(np.median([r['diagnostics']['candidate_count'] for r in rows])),
            'mean_candidates':float(np.mean([r['diagnostics']['candidate_count'] for r in rows])),
            'molecule_elapsed_seconds_sum':sum(r['diagnostics']['elapsed_seconds'] for r in rows),
            'molecule_elapsed_seconds_median':float(np.median([r['diagnostics']['elapsed_seconds'] for r in rows])),
            'structures_by_split':{k:len(v) for k,v in ids.items()},'split_intersections':0,
            'rank_and_formula_uniqueness':'passed','input_id_coverage':'passed',
            'interpretation':'Public test validates integration only; no accuracy measured against public test labels.'}
    write_json(root/'reports/integration.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
