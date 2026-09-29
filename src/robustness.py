"""PSNR error tails and SSIM-matched parameter/model robustness."""

from .config import WORKERS, crop_path


import gzip


import hashlib


import json


from concurrent.futures import ProcessPoolExecutor


import numpy as np


import pandas as pd


from PIL import Image


from .analysis import OUT, DOMAINS, LABELS, CODECS, accepted, read, bootstrap_median, ci


from .models import difference_bootstrap


from .formatting import format_number


from .codecs import BOUNDS, encode, ProtocolFailure


VALUE_NAMES = '''PNinetyToneJtk PNinetyConeJtk PNinetyToneJxl PNinetyConeJxl
RatioRSsim RatioRSsimLo RatioRSsimHi RatioDSsim RatioDSsimLo RatioDSsimHi
IqrRSsimMin IqrRSsimMax IqrDSsimMin IqrDSsimMax
RsqRSsimMin RsqRSsimMax RsqDSsimMin RsqDSsimMax
ErrToneJtkSsim ErrConeJtkSsim ErrToneJxlSsim ErrConeJxlSsim'''.split()


def eligible(data, codec):
    f = accepted(data, codec, 'ssim')
    if codec == 'jpegxl':
        f = f[~((f.domain == 'xray') & (f.q == 10))]
    return f


def fit_ssim(data):
    rows = []
    for codec in CODECS:
        train = eligible(data[data.split == 'train'], codec)
        for (domain, q), cell in train.groupby(['domain', 'q']):
            for predictor in ['T1', 'C1']:
                f = cell if predictor == 'T1' else cell[cell.C_DCT > 0]
                y = np.log(f[f'{codec}_ssim_parameter'].to_numpy())
                if predictor == 'T1':
                    intercept, slope = float(np.median(y)), 0.
                else:
                    x = np.log(f.C_DCT.to_numpy())
                    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
                rows.append(dict(domain=domain, q=int(q), codec=codec, predictor=predictor,
                                 intercept=float(intercept), slope=float(slope), train_n=len(f)))
    return pd.DataFrame(rows)


def evaluate_ssim_image(task):
    record, pairs, models, fingerprint = task
    dest = OUT / 'checkpoints/models_ssim' / record['domain'] / (record['image_id'] + '.json.gz')
    if dest.exists():
        with gzip.open(dest, 'rt') as f:
            checkpoint = json.load(f)
        if checkpoint['fingerprint'] != fingerprint:
            raise ProtocolFailure('SSIM checkpoint fingerprint mismatch')
        return str(dest)
    a = np.array(Image.open(crop_path(record)))
    if hashlib.sha256(a.tobytes()).hexdigest() != record['sha256']:
        raise ProtocolFailure('SSIM test image pixel hash mismatch')
    cache, rows = {}, []
    for pair in pairs:
        for codec in CODECS:
            prefix = codec + '_ssim_'
            if pair[prefix+'status'] not in ['matched', 'matched_scan'] or record['C_DCT'] <= 0:
                continue
            if codec == 'jpegxl' and record['domain'] == 'xray' and pair['q'] == 10:
                continue
            for predictor in ['T1', 'C1']:
                model = models[record['domain'], pair['q'], codec, predictor]
                logp = model['intercept'] + model['slope'] * np.log(record['C_DCT'])
                parameter = float(np.exp(np.clip(logp, *np.log(BOUNDS[codec]))))
                if (codec, parameter) not in cache:
                    cache[codec, parameter] = encode(a, codec, parameter)
                measured = cache[codec, parameter]
                rows.append(dict(domain=record['domain'], image_id=record['image_id'], group=record['group'],
                    q=pair['q'], codec=codec, predictor=predictor, exact_parameter=pair[prefix+'parameter'],
                    predicted_log_parameter=float(logp), parameter=parameter,
                    clipped=bool(logp < np.log(BOUNDS[codec][0]) or logp > np.log(BOUNDS[codec][1])),
                    target_ssim=pair['jpeg_ssim'], achieved_ssim=measured['ssim'],
                    encoded_bytes=measured['encoded_bytes'], error_ssim=abs(measured['ssim'] - pair['jpeg_ssim'])))
    if not np.isfinite([r['error_ssim'] for r in rows]).all():
        raise ProtocolFailure('Corrupted SSIM output')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.tmp')
    with gzip.open(temp, 'wt') as f:
        json.dump(dict(fingerprint=fingerprint, rows=rows), f)
    temp.replace(dest)
    return str(dest)


def table3(summary):
    rows = []
    for domain in DOMAINS:
        if rows:
            rows.append(r'\midrule')
        for codec, label in zip(CODECS, ['J2K', 'JXL']):
            f = summary[(summary.domain == domain) & (summary.codec == codec)].set_index('predictor')
            best = min(['T0', 'T1', 'C0', 'C1'], key=lambda p: f.loc[p, 'mean'])
            cells = []
            for predictor in ['T0', 'T1', 'C0', 'C1', 'floor']:
                mean = format_number(f.loc[predictor, 'mean'], '.2f')
                cells.extend([r'\textbf{' + mean + '}' if predictor == best else mean,
                              format_number(f.loc[predictor, 'p90'], '.2f')])
            rows.append(LABELS[domain] + ' & ' + label + ' & ' + ' & '.join(cells) + r' \\')
    return '\n'.join(rows) + '\n'


def dispersion(pairs):
    values, rows, ratios = {}, [], []
    rng = np.random.default_rng(0)
    for codec, symbol in zip(CODECS, ['R', 'D']):
        medians, reps, spreads = [], [], []
        for domain in DOMAINS:
            f = accepted(pairs[(pairs.q == 50) & (pairs.domain == domain)], codec, 'ssim')
            column = codec+'_ssim_parameter'
            q1, median, q3 = f[column].quantile([.25, .5, .75])
            medians.append(float(median))
            spreads.append(float(q3 / q1))
            reps.append(bootstrap_median(f, column, rng))
            rows.append(dict(domain=domain, codec=codec, q=50, n=len(f), q1=q1, median=median, q3=q3, quartile_ratio=q3/q1))
        reps = np.stack(reps)
        ratio = max(medians) / min(medians)
        lo, hi = ci(reps.max(axis=0) / reps.min(axis=0))
        values.update({f'Ratio{symbol}Ssim': ratio, f'Ratio{symbol}SsimLo': lo, f'Ratio{symbol}SsimHi': hi,
                       f'Iqr{symbol}SsimMin': min(spreads), f'Iqr{symbol}SsimMax': max(spreads)})
        ratios.append(dict(codec=codec, ratio=ratio, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(OUT/'ssim_equivalence_q50.csv', index=False)
    pd.DataFrame(ratios).to_csv(OUT/'ssim_between_domain_ratios.csv', index=False)
    return values


def summarize_ssim(errors):
    from .verification import validate_model_measurements
    validate_model_measurements(errors,'ssim')
    from .models import difference_bootstrap
    summary=errors.groupby(['domain','codec','predictor']).error_ssim.agg(n='size',mean='mean',p90=lambda x:x.quantile(.9)).reset_index()
    summary.to_csv(OUT/'ssim_model_error_summary.csv',index=False)
    r2rows=[]
    for (domain,codec),f in errors[(errors.q==50)&(errors.predictor=='C1')].groupby(['domain','codec']):
        y=np.log(f.exact_parameter);total=float(((y-y.mean())**2).sum())
        r2=float(1-((y-f.predicted_log_parameter)**2).sum()/total) if total else float('nan')
        r2rows.append(dict(domain=domain,codec=codec,q=50,n=len(f),r2=r2))
    r2=pd.DataFrame(r2rows);r2.to_csv(OUT/'ssim_complexity_r2.csv',index=False)
    rng=np.random.default_rng(0)
    differences=[difference_bootstrap(errors.rename(columns={'error_ssim':'error_db'}),c,'C1',rng) for c in CODECS]
    for row in differences:
        for k in ['difference','lo','hi']:row[k]*=1000
        row['unit']='10^-3 SSIM'
    pd.DataFrame(differences).to_csv(OUT/'ssim_model_differences.csv',index=False)
    psnr=read('model_error_summary.csv');values={}
    for codec,suffix,symbol in zip(CODECS,['Jtk','Jxl'],['R','D']):
        for predictor,tag in [('T1','Tone'),('C1','Cone')]:
            values[f'PNinety{tag}{suffix}']=float(psnr[(psnr.codec==codec)&(psnr.predictor==predictor)].p90.mean())
            values[f'Err{tag}{suffix}Ssim']=1000*float(summary[(summary.codec==codec)&(summary.predictor==predictor)]['mean'].mean())
        values.update({f'Rsq{symbol}SsimMin':float(r2[r2.codec==codec].r2.min()),f'Rsq{symbol}SsimMax':float(r2[r2.codec==codec].r2.max())})
    return values

def run_ssim():
    from .environment import verify
    verify()
    from .config import WORKERS
    pairs,features,manifest=read('pairs.csv'),read('complexity.csv'),read('manifest.csv')
    joined=manifest.merge(features,on=['domain','image_id','group'],validate='one_to_one')
    if joined.groupby(['domain','group']).split.nunique().max()!=1:raise ProtocolFailure('Group leakage')
    models=fit_ssim(pairs.merge(features,on=['domain','image_id','group'],validate='many_to_one'))
    models.to_csv(OUT/'ssim_model_parameters.csv',index=False)
    fingerprint=hashlib.sha256((OUT/'ssim_model_parameters.csv').read_bytes()+(OUT/'pairs.csv').read_bytes()+(OUT/'complexity.csv').read_bytes()).hexdigest()
    model_map={(x.domain,x.q,x.codec,x.predictor):x._asdict() for x in models.itertuples(index=False)}
    grouped={(d,i):f.to_dict('records') for (d,i),f in pairs.groupby(['domain','image_id'])}
    tasks=[(x,grouped[x['domain'],x['image_id']],model_map,fingerprint) for x in joined[joined.split=='test'].to_dict('records')]
    rows=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i,path in enumerate(pool.map(evaluate_ssim_image,tasks)):
            with gzip.open(path,'rt') as stream:rows.extend(json.load(stream)['rows'])
            print(f'SSIM models: {i+1}/{len(tasks)} test images',flush=True)
    errors=pd.DataFrame(rows)
    if errors.empty or not np.isfinite(errors.error_ssim).all() or errors.duplicated(['domain','image_id','q','codec','predictor']).any():raise ProtocolFailure('Invalid SSIM evaluation')
    errors.to_csv(OUT/'ssim_model_evaluation.csv',index=False)
    summarize_ssim(errors)
