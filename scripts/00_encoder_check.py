"""Rebuild the frozen sanity sample and compare JPEG 2000 settings."""
from _common import arguments,ROOT
import os
if __name__=='__main__':
    a=arguments(__doc__)
    if a.manifest==ROOT/'data/manifest.csv':
        a.manifest=ROOT/'data/sanity_manifest.csv';os.environ['EQUIVALENCE_MANIFEST']=str(a.manifest)
    if a.crops_dir==ROOT/'data/crops':
        a.crops_dir=ROOT/'data/sanity_crops';os.environ['EQUIVALENCE_CROPS']=str(a.crops_dir)
    from src.crops import rebuild
    if a.dataset_root:rebuild(a.manifest,a.dataset_root,a.crops_dir)
    from src.encoder_check import run
    run()
