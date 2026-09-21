"""Same GNPS cohort, removing the platform-specific raw-count gate only."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import polars as pl
from casmi.formula.data import write_json
from evaluate_enveda_alignment import group_from
from evaluate_external_sources import init_worker,evaluate,metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/reports/external-sources-v1'


def main():
    m=json.loads((OUT/'manifest.json').read_text())
    ids=m['selection']['gnps']['ids']
    rows=[]
    for path in sorted((ROOT/'artifacts/prepared/test').glob('*.parquet')):
        rows.extend(pl.read_parquet(path).filter((pl.col('ingest_lib')=='gnps')&pl.col('molecule_id').is_in(ids)).to_dicts())
    for row in rows:
        keep=[i for i,mz in enumerate(row['ms2_mzs']) if mz<=row['precursor_mz']+2]
        row['ms2_mzs']=[row['ms2_mzs'][i] for i in keep]
        row['ms2_normalized_intensities']=[row['ms2_normalized_intensities'][i] for i in keep]
    frame=pl.DataFrame(rows)
    jobs=[{**group_from(g.to_dicts()),'source':'gnps','raw_intensity_gate':False} for key,g in frame.group_by('molecule_id')]
    path=OUT/'gnps-no-raw-threshold.jsonl'
    records={r['molecule_id']:r for r in map(json.loads,path.read_text().splitlines())} if path.exists() else {}
    with path.open('a',encoding='utf-8') as log,ProcessPoolExecutor(max_workers=2,initializer=init_worker) as pool:
        futures=[pool.submit(evaluate,g) for g in jobs if g['molecule_id'] not in records]
        for future in as_completed(futures):
            r=future.result();records[r['molecule_id']]=r
            log.write(json.dumps(r)+'\n');log.flush()
            print(f"GNPS_NO_GATE {len(records)}/200",flush=True)
    assert set(records)==set(ids)
    write_json(OUT/'gnps-no-raw-threshold-summary.json',dict(learned=metrics(list(records.values())),baseline=metrics(list(records.values()),'baseline_rank'),
        protocol='Same 200 GNPS structures and model; preserve original base intensity including low values; only remove peaks above precursor+2; no raw intensity threshold.'))


if __name__=='__main__':
    main()
