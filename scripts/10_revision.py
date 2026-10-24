"""Revision sensitivity experiments: encoder configuration, image size and prediction robustness."""
import argparse
import os
from pathlib import Path
from _common import ROOT
import pandas as pd

STAGES = ['audit', 'configuration', 'size', 'prediction', 'reporting', 'validation']

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=[*STAGES, 'all'])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dataset-root', type=Path, help='rebuild and verify the revision images from the original downloads first')
    p.add_argument('--results-dir', type=Path, default=ROOT/'results/revision')
    p.add_argument('--paper-dir', type=Path, default=ROOT/'paper/revision')
    p.add_argument('--subset', type=Path, default=ROOT/'data/revision/subset.csv')
    p.add_argument('--size-manifest', type=Path, default=ROOT/'data/revision/size_manifest.csv')
    p.add_argument('--split-file', type=Path, help='explicit group-disjoint prediction splits (seed, domain, group, split)')
    p.add_argument('--crops-dir', type=Path, default=ROOT/'data/crops')
    p.add_argument('--revision-crops-dir', type=Path, default=ROOT/'data/revision_crops')
    p.add_argument('--reselect-size', type=Path, metavar='LANDMARKS', help='re-derive size eligibility (needs Shenzhen lung landmarks) and compare with the frozen manifest')
    a = p.parse_args()
    for key, value in [('REVISION_RESULTS', a.results_dir), ('REVISION_PAPER', a.paper_dir), ('REVISION_SUBSET', a.subset),
                       ('REVISION_SIZE_MANIFEST', a.size_manifest), ('REVISION_SPLITS', a.split_file),
                       ('CROPS', a.crops_dir), ('REVISION_CROPS', a.revision_crops_dir)]:
        if value is not None:
            os.environ['EQUIVALENCE_'+key] = str(value.resolve())
    a.results_dir.mkdir(parents=True, exist_ok=True)
    from src import revision
    if a.dataset_root:
        from src.crops import rebuild
        size = pd.read_csv(a.size_manifest, dtype={'image_id': str, 'group': str})
        ids = pd.concat([pd.read_csv(a.subset, dtype={'image_id': str})[['domain', 'image_id']],
                         size[size.path.str.startswith('data/crops/')][['domain', 'image_id']]]).drop_duplicates()
        manifest = pd.read_csv(ROOT/'data/manifest.csv', dtype={'image_id': str, 'group': str}, float_precision='round_trip')
        a.revision_crops_dir.mkdir(parents=True, exist_ok=True)
        path = a.revision_crops_dir/'baseline_crops_manifest.csv'
        manifest.merge(ids, on=['domain', 'image_id'], validate='one_to_one').to_csv(path, index=False)
        rebuild(path, a.dataset_root, a.crops_dir)
        revision.rebuild_size(a.dataset_root)
        if a.reselect_size:
            revision.select_size(a.dataset_root, a.reselect_size)
    for stage in STAGES if a.stage == 'all' else [a.stage]:
        revision.run_stage(stage, a.workers)
