"""Four-setting JPEG 2000 encoder check on the frozen 20-image sample."""
import io
import numpy as np
import pandas as pd
import imagecodecs
from PIL import Image
from .config import OUT, MANIFEST, crop_path
from .codecs import ProtocolFailure
CONFIGS={'j2k_defaults':(0,False),'j2k_mct_only':(1,False),'j2k_irreversible_only':(0,True),'j2k_both_fixed':(1,True)}
def pixels(a):
    if a.ndim == 3 and np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]):
        a = a[..., 0]
    if a.dtype != np.uint8 or a.ndim not in (2, 3):
        raise ValueError('Expected uint8 grayscale or RGB')
    return np.ascontiguousarray(a)


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float))**2)
    return float('inf') if mse == 0 else float(10 * np.log10(255**2 / mse))


def encode(a, codec, parameter):
    a = pixels(a)
    if codec == 'jpegxl':
        data = imagecodecs.jpegxl_encode(a, distance=float(parameter), effort=7)
        decoded = imagecodecs.jpegxl_decode(data)
    else:
        buf = io.BytesIO()
        im = Image.fromarray(a)
        if codec == 'jpeg':
            im.save(buf, format='JPEG', quality=int(parameter), progressive=False,
                    subsampling=2, optimize=False)
        else:
            mct, irreversible = CONFIGS[codec]
            im.save(buf, format='JPEG2000', quality_mode='rates',
                    quality_layers=[float(parameter)], mct=mct if a.ndim == 3 else 0,
                    irreversible=irreversible, tile_size=im.size)
        data = buf.getvalue()
        decoded = np.array(Image.open(io.BytesIO(data)))
    if decoded.shape != a.shape or decoded.dtype != a.dtype:raise ProtocolFailure('Encoder-check round-trip mismatch')
    return dict(psnr=psnr(a, decoded), bpp=8 * len(data) / (a.shape[0] * a.shape[1]),
                encoded_bytes=len(data))


def search(a, codec, target, tolerance=.01, max_calls=42):
    low, high = ((.05, 25.) if codec == 'jpegxl' else (1.5, 1000.))
    trials = []
    def evaluate(p):
        result = encode(a, codec, p)
        trials.append(dict(parameter=p, **result, residual_db=abs(result['psnr'] - target)))
        return result['psnr']
    pl, ph = evaluate(low), evaluate(high)
    outside = target > max(pl, ph) or target < min(pl, ph)
    if not outside:
        while len(trials) < max_calls and min(t['residual_db'] for t in trials) > tolerance:
            middle = float(np.sqrt(low * high))
            if abs(np.log(high / low)) < 1e-10:
                break
            value = evaluate(middle)
            if value > target:
                low = middle
            else:
                high = middle
    best = min(trials, key=lambda r: r['residual_db'])
    return dict(**best, encoder_calls=len(trials), out_of_range=outside,
                tolerance_met=best['residual_db'] <= tolerance)


def run():
    from .environment import verify
    verify()
    manifest=pd.read_csv(MANIFEST,dtype={'group':str,'image_id':str})
    rows=[]
    for record in manifest.to_dict('records'):
        if 'image_id' not in record:record['image_id']=record['group']
        a=np.array(Image.open(crop_path(record)))
        import hashlib
        if hashlib.sha256(pixels(a).tobytes()).hexdigest()!=record['sha256']:raise ProtocolFailure('Encoder-check pixel hash mismatch')
        for q in [30,50,70]:
            target=encode(a,'jpeg',q)
            row=dict(domain=record['domain'],group=record['group'],jpeg_quality=q,jpeg_psnr=target['psnr'],jpeg_bpp=target['bpp'])
            for codec in [*CONFIGS,'jpegxl']:
                row.update({codec+'_'+k:v for k,v in search(a,codec,target['psnr']).items()})
            rows.append(row)
            print(f"Encoder check: {record['domain']} {record['group']} q={q}",flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'sanity_mct_per_image.csv',index=False)
    summarize(frame)

def summarize(frame):
    means=frame.groupby(['domain','jpeg_quality'])[[c for c in frame if c.endswith('_bpp')]].mean()
    means['j2k_fixed_savings_vs_jpeg_pct']=100*(1-means.j2k_both_fixed_bpp/means.jpeg_bpp)
    means['j2k_fixed_at_or_below_jxl']=means.j2k_both_fixed_bpp<=means.jpegxl_bpp
    means.to_csv(OUT/'sanity_mct.csv')
    return means
