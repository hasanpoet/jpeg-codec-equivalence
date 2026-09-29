"""Fresh PSNR/SSIM searches with checkpointed observations and explicit gates."""


import hashlib


import time


from concurrent.futures import ProcessPoolExecutor


import numpy as np


import pandas as pd


from PIL import Image


from .codecs import BOUNDS, QUALITIES, ProtocolFailure, encode


from .config import OUT, MANIFEST, WORKERS, crop_path


STATUSES = ['matched', 'matched_scan', 'unreachable_low', 'unreachable_high', 'unmatched']


def bisect(evaluate, codec, target, metric):
    tolerance = 1e-4 if metric == 'ssim' else .02 if codec == 'jpeg2000' else .01
    cap = 40 if codec == 'jpeg2000' else 60
    lo, hi = BOUNDS[codec]
    trials = []
    def call(p):
        r = dict(evaluate(p))
        if not np.isfinite(r[metric]):
            # Lossless endpoints can have infinite PSNR; NaN is never valid.
            if metric != 'psnr' or r[metric] != float('inf'):
                raise ProtocolFailure('Invalid decoded fidelity')
        r['residual'] = abs(r[metric] - target)
        trials.append(r)
        return r[metric]
    vl, vh = call(lo), call(hi)
    # Low/high refer to target fidelity relative to the achievable range.
    status = 'unreachable_high' if target > vl else 'unreachable_low' if target < vh else 'matched'
    if min(r['residual'] for r in trials) <= tolerance:
        status = 'matched'
    if status == 'matched':
        while len(trials) < cap and min(r['residual'] for r in trials) > tolerance:
            if np.log(hi / lo) < 1e-10:
                break
            mid = float(np.sqrt(lo * hi))
            value = call(mid)
            if value > target:
                lo = mid
            else:
                hi = mid
    ordered = sorted(trials, key=lambda r: r['parameter'])
    # Only fidelity reversals greater than the matching tolerance are material.
    inversion = max([0.] + [b[metric] - a[metric] for a, b in zip(ordered, ordered[1:])])
    if inversion > tolerance:
        status = 'nonmonotone'
    best = min(trials, key=lambda r: r['residual'])
    return dict(**{k: best[k] for k in ['parameter', 'psnr', 'ssim', 'encoded_bytes', 'bpp', 'residual']},
                calls=len(trials), status=status, tolerance_met=best['residual'] <= tolerance,
                max_inversion=inversion), trials


def scan(evaluate, codec, target, metric, previous, trials):
    tolerance = 1e-3 if metric == 'ssim' else .1
    scanned = []
    for p in np.geomspace(*BOUNDS[codec], 64):
        r = dict(evaluate(float(p)))
        r['residual'] = abs(r[metric] - target)
        scanned.append(r)
    # Retain the best observation, including the already measured bisection point.
    best = min([previous, *scanned], key=lambda r: r['residual'])
    passed = best['residual'] <= tolerance
    result = dict(previous)
    result.update({k: best[k] for k in ['parameter', 'psnr', 'ssim', 'encoded_bytes', 'bpp', 'residual']})
    result.update(status='matched_scan' if passed else 'unmatched', path='scan' if passed else 'unmatched',
                  initial_status='nonmonotone', initially_nonmonotone=True,
                  bisection_calls=previous['calls'], scan_calls=64, calls=previous['calls'] + 64,
                  tolerance_met=best['residual'] <= (1e-4 if metric == 'ssim' else .02 if codec == 'jpeg2000' else .01),
                  scan_tolerance_met=passed)
    return result, dict(bisection=trials, scan=scanned)


def match(evaluate, codec, target, metric):
    result, trials = bisect(evaluate, codec, target, metric)
    if result['status'] == 'nonmonotone':
        return scan(evaluate, codec, target, metric, result, trials)
    result.update(initial_status=result['status'], initially_nonmonotone=False,
                  path='bisection', bisection_calls=result['calls'], scan_calls=0, scan_tolerance_met=False)
    return result, dict(bisection=trials)


def counts(pairs):
    parts = []
    for codec in BOUNDS:
        for metric in ['psnr', 'ssim']:
            name = f'{codec}_{metric}_status'
            part = pairs.groupby(['domain', 'q', name]).size().unstack(fill_value=0).reindex(columns=STATUSES, fill_value=0).reset_index()
            part['codec'], part['metric'] = codec, metric
            initial = pairs.groupby(['domain', 'q'])[f'{codec}_{metric}_initially_nonmonotone'].sum()
            part['nonmonotone'] = [int(initial.loc[d, q]) for d, q in zip(part.domain, part.q)]
            part['total'] = part[STATUSES].sum(axis=1)
            part['excluded_pct'] = 100 * (part.unmatched + part.unreachable_low) / part.total
            part['flag_over_20pct'] = part.excluded_pct > 20
            parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result.to_csv(OUT / 'status_counts.csv', index=False)
    return result


def image_pairs(record):
    """Fresh final-protocol searches; reuse only identical encodes within this image."""
    from .config import crop_path
    a=np.array(Image.open(crop_path(record)))
    if hashlib.sha256(a.tobytes()).hexdigest()!=record['sha256']:
        raise ProtocolFailure('Pixel hash mismatch')
    rows=[];cache={};actual=0
    def evaluate(codec,p):
        nonlocal actual
        key=codec,float(p)
        if key not in cache:
            cache[key]=encode(a,codec,p);actual+=1
        return cache[key]
    for q in QUALITIES:
        jpeg=evaluate('jpeg',q)
        row={k:record[k] for k in ['domain','image_id','group']}
        row.update(q=q,jpeg_psnr=jpeg['psnr'],jpeg_ssim=jpeg['ssim'],jpeg_bytes=jpeg['encoded_bytes'],jpeg_bpp=jpeg['bpp'])
        for codec in BOUNDS:
            for metric in ['psnr','ssim']:
                before=actual
                result,_=match(lambda p:evaluate(codec,p),codec,row['jpeg_'+metric],metric)
                result['new_encode_calls']=actual-before
                row.update({codec+'_'+metric+'_'+k:v for k,v in result.items()})
        rows.append(row)
    return rows

def run():
    from .environment import verify
    verify()
    from .config import MANIFEST,WORKERS
    from .analysis import write_json
    manifest=pd.read_csv(MANIFEST,dtype={'image_id':str,'group':str})
    if manifest.sha256.duplicated().any() or manifest.duplicated(['domain','image_id']).any():
        raise ProtocolFailure('Duplicate hashes or image identifiers')
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i,part in enumerate(pool.map(image_pairs,manifest.to_dict('records'))):
            rows.extend(part)
            pd.DataFrame(rows).to_csv(OUT/'pairs.partial.csv',index=False)
            print(f'Searched {i+1}/{len(manifest)} images; {len(rows)} pairs; {time.monotonic()-start:.1f}s',flush=True)
    pairs=pd.DataFrame(rows)
    if len(pairs)!=9*len(manifest):raise ProtocolFailure('Incomplete search')
    pairs.to_csv(OUT/'pairs.csv',index=False);(OUT/'pairs.partial.csv').unlink()
    counts(pairs)
    write_json('search_check.json',dict(status='passed',images=len(manifest),pairs=len(pairs),seconds=time.monotonic()-start,protocol='final',workers=WORKERS))
