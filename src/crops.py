"""Rebuild frozen crops directly from source archives and manifest coordinates."""
import gzip
import hashlib
import io
import tarfile
import zipfile
from functools import lru_cache
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from .codecs import canonical_pixels, ProtocolFailure

@lru_cache(maxsize=8)
def archive(path):
    return zipfile.ZipFile(path) if str(path).endswith('.zip') else tarfile.open(path)

def source_bytes(source, dataset_root):
    source = source.removeprefix('data/')
    if '::' in source:
        filename, member = source.split('::', 1)
        if not (Path(dataset_root)/filename).exists():
            extracted=Path(dataset_root)/member
            if extracted.exists():return extracted.read_bytes()
        a = archive(str(Path(dataset_root)/filename))
        return a.read(member) if isinstance(a,zipfile.ZipFile) else a.extractfile(member).read()
    p = Path(dataset_root)/source
    if p.exists():
        return p.read_bytes()
    if source.startswith('ixi_nifti/'):
        a = archive(str(Path(dataset_root)/'IXI-T1.tar'))
        member = next(m for m in a.getmembers() if Path(m.name).name == Path(source).name)
        return a.extractfile(member).read()
    if source.startswith('ucmerced/Images/'):
        a = archive(str(Path(dataset_root)/'ucmerced/UCMerced_LandUse.zip'))
        suffix = source.removeprefix('ucmerced/')
        member = next(n for n in a.namelist() if n.endswith(suffix))
        return a.read(member)
    raise FileNotFoundError(p)

@lru_cache(maxsize=1)
def volume(source, dataset_root):
    raw = source_bytes(source,dataset_root)
    if source.endswith('.gz'):raw=gzip.decompress(raw)
    image=nib.as_closest_canonical(nib.Nifti1Image.from_bytes(raw))
    return image.get_fdata(dtype=np.float32)

def rebuild(manifest, dataset_root, destination):
    frame=pd.read_csv(manifest,dtype={'image_id':str,'group':str},float_precision='round_trip')
    if 'image_id' not in frame:frame['image_id']=frame.group
    if frame.sha256.duplicated().any():raise ProtocolFailure('Duplicate crop hashes')
    for i,row in enumerate(frame.to_dict('records')):
        x,y=int(row['crop_x']),int(row['crop_y'])
        if row['domain']=='mri':
            v=volume(row['source'],str(dataset_root))
            a=np.rint(np.clip((v[int(row['slice_index_canonical'])]-row['window_low'])/(row['window_high']-row['window_low']),0,1)*255).astype(np.uint8)
            a=a[y:y+256,x:x+256]
        else:
            im=Image.open(io.BytesIO(source_bytes(row['source'],dataset_root))).convert('L' if row['domain']=='xray' else 'RGB')
            factor=int(row['integer_factor'])
            if factor>1:im=im.resize((im.width//factor,im.height//factor),Image.Resampling.BOX)
            a=np.asarray(im.crop((x,y,x+256,y+256)))
        a=canonical_pixels(a)
        if a.shape[:2]!=(256,256) or hashlib.sha256(a.tobytes()).hexdigest()!=row['sha256']:
            raise ProtocolFailure('Rebuilt crop hash mismatch: '+row['image_id'])
        path=Path(destination)/row['domain']/(row['image_id']+'.png')
        path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(a).save(path)
        if (i+1)%25==0:print(f'Rebuilt and verified {i+1}/{len(frame)} crops',flush=True)
    print(f'All {len(frame)} crops match manifest SHA-256.',flush=True)
