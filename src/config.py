"""Repository-relative locations; stage scripts may select an isolated output directory."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('EQUIVALENCE_RESULTS', ROOT / 'results')).resolve()
GEN = Path(os.environ.get('EQUIVALENCE_PAPER', ROOT / 'paper')).resolve()
MANIFEST = Path(os.environ.get('EQUIVALENCE_MANIFEST', ROOT / 'data/manifest.csv')).resolve()
CROPS = Path(os.environ.get('EQUIVALENCE_CROPS', ROOT / 'data/crops')).resolve()
WORKERS = int(os.environ.get('EQUIVALENCE_WORKERS', min(4, os.cpu_count() or 1)))
def crop_path(record):
    return CROPS / record['domain'] / (record['image_id'] + '.png')
