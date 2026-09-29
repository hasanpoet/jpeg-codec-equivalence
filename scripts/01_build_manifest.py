"""Rebuild and verify the published manifest's crops; do not select new images."""
from _common import arguments
if __name__=='__main__':
    a=arguments(__doc__)
    if a.dataset_root is None:raise SystemExit('--dataset-root is required')
    from src.crops import rebuild
    rebuild(a.manifest,a.dataset_root,a.crops_dir)
