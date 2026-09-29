"""Independently replay the original 612 pairs with fresh final-protocol searches."""
from _common import arguments,ROOT
import os
import pandas as pd
if __name__=='__main__':
    a=arguments(__doc__)
    if a.results_dir.resolve()==(ROOT/'results').resolve():raise SystemExit('Use an isolated --results-dir for the replay')
    subset=pd.read_csv(ROOT/'data/replay_subset.csv',dtype={'image_id':str})
    manifest=pd.read_csv(a.manifest,dtype={'image_id':str,'group':str},float_precision='round_trip')
    ids=subset[['domain','image_id']].drop_duplicates()
    selected=manifest.merge(ids,on=['domain','image_id'],validate='one_to_one')
    path=a.results_dir/'manifest.csv';selected.to_csv(path,index=False)
    os.environ['EQUIVALENCE_MANIFEST']=str(path.resolve())
    if a.dataset_root:
        from src.crops import rebuild
        rebuild(path,a.dataset_root,a.crops_dir)
    from src.search import run
    run()
    from src.verification import compare_replay
    report=compare_replay(ROOT/'results/pairs.csv',a.results_dir/'pairs.csv',a.results_dir)
    print(report)
    if report['pairs']!=len(subset) or report['scientific_differences']:raise SystemExit('Replay discrepancies: see replay_check.json')
