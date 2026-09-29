"""Shared CLI setup; configure paths before importing stage modules."""
import argparse
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def arguments(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--results-dir',type=Path,default=ROOT/'results')
    p.add_argument('--paper-dir',type=Path,default=ROOT/'paper')
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manifest.csv')
    p.add_argument('--crops-dir',type=Path,default=ROOT/'data/crops')
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--split-file',type=Path)
    p.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    a=p.parse_args()
    if a.split_file:os.environ['EQUIVALENCE_SPLITS']=str(a.split_file.resolve())
    for key,value in [('RESULTS',a.results_dir),('PAPER',a.paper_dir),('MANIFEST',a.manifest),('CROPS',a.crops_dir),('WORKERS',a.workers)]:
        os.environ['EQUIVALENCE_'+key]=str(value.resolve() if isinstance(value,Path) else value)
    a.results_dir.mkdir(parents=True,exist_ok=True);a.paper_dir.mkdir(parents=True,exist_ok=True)
    return a
