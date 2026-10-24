"""Train-only letter predictors and explicit held-out re-encoding."""

from .features import Q50, complexity
from .config import WORKERS, crop_path


import gzip


import hashlib


import json


import os


from concurrent.futures import ProcessPoolExecutor


import numpy as np


import pandas as pd


from PIL import Image


from .analysis import OUT, DOMAINS, CODECS, REPLICATES, accepted, read, write_json, ci


from .codecs import BOUNDS, encode, ProtocolFailure


PREDICTORS = ['T0', 'T1', 'C0', 'C1', 'C1_BPP']


def split_and_features(manifest, pairs):
    rng = np.random.default_rng(0)
    splits = []
    for domain in DOMAINS:
        groups = sorted(manifest.loc[manifest.domain == domain, 'group'].unique())
        order = rng.permutation(len(groups))
        train = set(order[:int(.8 * len(groups))])
        splits.extend(dict(domain=domain, group=g, split='train' if i in train else 'test') for i, g in enumerate(groups))
    splits = pd.DataFrame(splits)
    if os.environ.get('EQUIVALENCE_SPLITS'):
        splits = pd.read_csv(os.environ['EQUIVALENCE_SPLITS'], dtype={'group': str})
        wanted = manifest[['domain', 'group']].drop_duplicates()
        splits = wanted.merge(splits, on=['domain', 'group'], validate='one_to_one')
        if len(splits) != len(wanted) or not splits.split.isin(['train','test']).all():
            raise ProtocolFailure('Incomplete or invalid explicit split')
    joined = manifest.merge(splits, on=['domain', 'group'], validate='many_to_one')
    if joined.groupby(['domain', 'group']).split.nunique().max() != 1:
        raise ProtocolFailure('Group leakage')
    splits.to_csv(OUT / 'model_splits.csv', index=False)
    bpp = pairs[pairs.q == 50].set_index(['domain', 'image_id']).jpeg_bpp
    rows = []
    for r in joined.itertuples():
        a = np.array(Image.open(crop_path(r._asdict())))
        if hashlib.sha256(a.tobytes()).hexdigest() != r.sha256:
            raise ProtocolFailure('Pixel hash mismatch in complexity pass')
        rows.append(dict(domain=r.domain, image_id=r.image_id, group=r.group, split=r.split,
                         C_DCT=complexity(a), C_BPP=float(bpp.loc[r.domain, r.image_id])))
    features = pd.DataFrame(rows)
    features.to_csv(OUT / 'complexity.csv', index=False)
    return joined, features


def fit_models(pairs, features):
    data = pairs.merge(features.drop(columns='group'), on=['domain', 'image_id'], validate='many_to_one')
    rows = []
    for codec in CODECS:
        train = accepted(data[data.split == 'train'], codec)
        if codec == 'jpegxl':
            train = train[~((train.domain == 'xray') & (train.q == 10))]
        for predictor in PREDICTORS:
            feature = 'C_BPP' if predictor == 'C1_BPP' else 'C_DCT'
            eligible = train if predictor.startswith('T') else train[train[feature] > 0]
            grouped = eligible.groupby('q') if predictor in ['T0', 'C0'] else eligible.groupby(['domain', 'q'])
            for key, f in grouped:
                domain, q = ('all', key) if predictor in ['T0', 'C0'] else key
                y = np.log(f[f'{codec}_psnr_parameter'].to_numpy())
                feature = 'C_BPP' if predictor == 'C1_BPP' else 'C_DCT'
                if predictor.startswith('T'):
                    intercept, slope = float(np.median(y)), 0.
                else:
                    x = np.log(f[feature].to_numpy())
                    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
                rows.append(dict(domain=domain, q=int(q), codec=codec, predictor=predictor,
                                 feature=feature, intercept=float(intercept), slope=float(slope), train_n=len(f)))
    models = pd.DataFrame(rows)
    models.to_csv(OUT / 'model_parameters.csv', index=False)
    return data, models


def evaluate_image(task):
    record, pairs, features, models, fingerprint = task
    dest = OUT / 'checkpoints/models' / record['domain'] / (record['image_id'] + '.json.gz')
    if dest.exists():
        with gzip.open(dest, 'rt') as f:
            data = json.load(f)
        if data['fingerprint'] != fingerprint:
            raise ProtocolFailure('Model checkpoint fingerprint mismatch')
        return str(dest)
    a = np.array(Image.open(crop_path(record)))
    if hashlib.sha256(a.tobytes()).hexdigest() != record['sha256']:
        raise ProtocolFailure('Test image pixel hash mismatch')
    cache, rows = {}, []
    for pair in pairs:
        for codec in CODECS:
            if codec == 'jpegxl' and record['domain'] == 'xray' and pair['q'] == 10:
                continue
            prefix = codec + '_psnr_'
            if pair[prefix + 'status'] not in ['matched', 'matched_scan']:
                continue
            # A logarithmic model is undefined at zero complexity. Report these
            # images explicitly; do not fabricate an epsilon or change the model.
            if features['C_DCT'] <= 0 or features['C_BPP'] <= 0:
                continue
            base = {k: record[k] for k in ['domain', 'image_id', 'group']}
            base.update(q=pair['q'], codec=codec, target_psnr=pair['jpeg_psnr'],
                        exact_parameter=pair[prefix + 'parameter'])
            for predictor in PREDICTORS:
                domain = 'all' if predictor in ['T0', 'C0'] else record['domain']
                model = models[(codec, predictor, domain, pair['q'])]
                logp = model['intercept'] + model['slope'] * np.log(features[model['feature']])
                parameter = float(np.exp(np.clip(logp, *np.log(BOUNDS[codec]))))
                key = codec, parameter
                if key not in cache:
                    cache[key] = encode(a, codec, parameter)
                measured = cache[key]
                rows.append(dict(base, predictor=predictor, predicted_log_parameter=float(logp), parameter=parameter,
                                 clipped=bool(logp < np.log(BOUNDS[codec][0]) or logp > np.log(BOUNDS[codec][1])),
                                 achieved_psnr=measured['psnr'], encoded_bytes=measured['encoded_bytes'],
                                 error_db=abs(measured['psnr'] - pair['jpeg_psnr'])))
            rows.append(dict(base, predictor='floor', predicted_log_parameter=float(np.log(pair[prefix + 'parameter'])),
                             parameter=pair[prefix + 'parameter'], clipped=False,
                             achieved_psnr=pair[prefix + 'psnr'], encoded_bytes=pair[prefix + 'encoded_bytes'],
                             error_db=pair[prefix + 'residual']))
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.tmp')
    with gzip.open(temp, 'wt') as f:
        json.dump(dict(fingerprint=fingerprint, rows=rows), f)
    temp.replace(dest)
    return str(dest)


def difference_bootstrap(errors, codec, right, rng):
    f = errors[errors.codec == codec]
    wide = f.pivot(index=['domain', 'group', 'image_id', 'q'], columns='predictor', values='error_db').reset_index()
    wide['difference'] = wide.T1 - wide[right]
    if wide[['T1', right]].isna().any().any():
        raise ProtocolFailure('Unpaired model comparison')
    means = f.groupby(['domain', 'predictor']).error_db.mean().unstack()
    observed = float(means.T1.mean() - means[right].mean())
    totals = np.zeros(REPLICATES)
    for domain in DOMAINS:
        groups = wide[wide.domain == domain].groupby('group').difference.agg(['sum', 'size'])
        sums, sizes = groups['sum'].to_numpy(), groups['size'].to_numpy()
        for begin in range(0, REPLICATES, 250):
            ix = rng.integers(len(groups), size=(min(250, REPLICATES - begin), len(groups)))
            totals[begin:begin+len(ix)] += sums[ix].sum(axis=1) / sizes[ix].sum(axis=1)
    lo, hi = ci(totals / len(DOMAINS))
    return dict(codec=codec, comparison='T1-' + right, difference=observed, lo=lo, hi=hi)


def summarize_psnr(errors):
    from .verification import validate_model_measurements
    validate_model_measurements(errors,'psnr')
    summary = errors.groupby(['domain', 'codec', 'predictor']).error_db.agg(
        n='size', mean='mean', p90=lambda x: x.quantile(.9), within_half_db_pct=lambda x: 100 * (x <= .5).mean()).reset_index()
    summary.to_csv(OUT / 'model_error_summary.csv', index=False)
    r2rows = []
    for (domain, codec), f in errors[(errors.q == 50) & (errors.predictor == 'C1')].groupby(['domain', 'codec']):
        y = np.log(f.exact_parameter)
        total = float(((y - y.mean())**2).sum())
        r2 = float(1 - ((y - f.predicted_log_parameter)**2).sum() / total) if total else None
        r2rows.append(dict(domain=domain, codec=codec, q=50, n=len(f), r2=r2))
    r2 = pd.DataFrame(r2rows)
    r2.to_csv(OUT / 'complexity_r2.csv', index=False)
    rng = np.random.default_rng(0)
    differences = [difference_bootstrap(errors, codec, predictor, rng) for codec in CODECS for predictor in ['C1', 'C0']]
    pd.DataFrame(differences).to_csv(OUT / 'model_differences.csv', index=False)
    values = {}
    for codec, suffix, symbol in zip(CODECS, ['Jtk', 'Jxl'], ['R', 'D']):
        for predictor, tag in zip(['T0', 'T1', 'C0', 'C1', 'floor', 'C1_BPP'], ['Tzero', 'Tone', 'Czero', 'Cone', 'Floor', 'ConeBpp']):
            values['Err' + tag + suffix] = float(summary[(summary.codec == codec) & (summary.predictor == predictor)]['mean'].mean())
        quality = r2[r2.codec == codec].r2
        values.update({f'Rsq{symbol}min': float(quality.min()), f'Rsq{symbol}max': float(quality.max())})
        for d in [r for r in differences if r['codec'] == codec]:
            stem = 'DiffTone' + ('Cone' if d['comparison'] == 'T1-C1' else 'Czero') + suffix
            values.update({stem: d['difference'], stem + 'Lo': d['lo'], stem + 'Hi': d['hi']})
    write_json('model_values.json', values)
    return values


def run_psnr():
    from .environment import verify
    verify()
    from .config import MANIFEST, WORKERS
    manifest,pairs=read('manifest.csv'),read('pairs.csv')
    joined,features=split_and_features(manifest,pairs)
    _,parameters=fit_models(pairs,features)
    models={(x.codec,x.predictor,x.domain,x.q):x._asdict() for x in parameters.itertuples(index=False)}
    grouped={(d,i):f.to_dict('records') for (d,i),f in pairs.groupby(['domain','image_id'])}
    feature_map={(x.domain,x.image_id):x._asdict() for x in features.itertuples(index=False)}
    fingerprint=hashlib.sha256((OUT/'model_parameters.csv').read_bytes()+(OUT/'complexity.csv').read_bytes()+(OUT/'pairs.csv').read_bytes()).hexdigest()
    tasks=[(x,grouped[x['domain'],x['image_id']],feature_map[x['domain'],x['image_id']],models,fingerprint) for x in joined[joined.split=='test'].to_dict('records')]
    frames=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i,path in enumerate(pool.map(evaluate_image,tasks)):
            with gzip.open(path,'rt') as stream:frames.extend(json.load(stream)['rows'])
            print(f'PSNR models: {i+1}/{len(tasks)} test images',flush=True)
    errors=pd.DataFrame(frames)
    if errors.empty or errors.duplicated(['domain','image_id','q','codec','predictor']).any() or not np.isfinite(errors.error_db).all():raise ProtocolFailure('Invalid PSNR evaluation')
    errors.to_csv(OUT/'model_evaluation.csv',index=False)
    summarize_psnr(errors)
