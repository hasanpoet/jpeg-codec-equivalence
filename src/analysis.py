"""Letter tables and group-bootstrap estimates from the completed current run."""


import json


import numpy as np


import pandas as pd


from .formatting import format_number


from .config import ROOT, OUT, GEN, MANIFEST


DOMAINS = ['natural', 'xray', 'mri', 'aerial']


LABELS = dict(natural='Natural', xray='X-ray', mri='MRI', aerial='Aerial')


CODECS = ['jpeg2000', 'jpegxl']


ACCEPTED = ['matched', 'matched_scan']


REPLICATES = 10000


def read(name):
    return pd.read_csv(MANIFEST if name == 'manifest.csv' else OUT / name, dtype={'image_id': str, 'group': str}, float_precision='round_trip')


def write_json(name, value):
    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, np.generic):
            item = item.item()
        return None if isinstance(item, float) and not np.isfinite(item) else item
    (OUT / name).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def accepted(frame, codec, metric='psnr'):
    return frame[frame[f'{codec}_{metric}_status'].isin(ACCEPTED)].copy()


def bootstrap_median(frame, column, rng):
    groups = [f[column].to_numpy() for _, f in frame.groupby('group', sort=True)]
    if not groups:
        return np.full(REPLICATES, np.nan)
    # Cluster multiplicity is preserved: a selected group contributes all its images.
    out = np.empty(REPLICATES)
    for i in range(REPLICATES):
        sampled = rng.integers(len(groups), size=len(groups))
        out[i] = np.median(np.concatenate([groups[j] for j in sampled]))
    return out


def ci(values):
    if np.isnan(values).all():
        return [float('nan'), float('nan')]
    assert np.isfinite(values).all()
    return [float(x) for x in np.quantile(values, [.025, .975])]


def table2(eq):
    table = []
    for codec, heading in [('jpeg2000', r'JPEG~2000 ratio $r^*$'), ('jpegxl', r'JPEG~XL distance $d^*$')]:
        if table:
            table.append(r'\midrule')
        table.append(r'\multicolumn{4}{l}{\emph{' + heading + r'}}\\')
        for domain in DOMAINS:
            cells = []
            for q in [20, 50, 80]:
                r = eq[(eq.codec == codec) & (eq.domain == domain) & (eq.q == q)].iloc[0]
                text = (format_number(r['median'], '.3g') + ' [' + format_number(r.q1, '.3g') + ', '
                        + format_number(r.q3, '.3g') + ']') if r.n else 'NA'
                if r.unreachable_pct > 5:
                    text += r'$^\dagger$'
                cells.append(text)
            table.append(LABELS[domain] + ' & ' + ' & '.join(cells) + r' \\')
    return '\n'.join(table) + '\n'


def analyze_pairs():
    pairs = read('pairs.csv')
    from .verification import validate_pairs
    validate_pairs(read('manifest.csv'),pairs)
    rng = np.random.default_rng(0)
    equivalence = []
    for codec in CODECS:
        for (domain, q), cell in pairs.groupby(['domain', 'q']):
            f = accepted(cell, codec)
            vals = f[f'{codec}_psnr_parameter']
            quantiles = vals.quantile([.25, .5, .75])
            equivalence.append(dict(domain=domain, q=int(q), codec=codec, n=len(f), total=len(cell),
                q1=float(quantiles.loc[.25]), median=float(quantiles.loc[.5]), q3=float(quantiles.loc[.75]),
                unreachable_pct=100 * cell[f'{codec}_psnr_status'].isin(['unreachable_low', 'unreachable_high']).mean(),
                excluded_pct=100 * (~cell[f'{codec}_psnr_status'].isin(ACCEPTED)).mean()))
    eq = pd.DataFrame(equivalence)
    eq.to_csv(OUT / 'equivalence_full.csv', index=False)
    (GEN / 'table2_body.tex').write_text(table2(eq))
    q50 = pairs[pairs.q == 50]
    values, rates, ratios, spreads, wins = {}, [], [], [], []
    for codec, symbol in zip(CODECS, ['R', 'D']):
        medians, replicates = [], []
        for domain in DOMAINS:
            f = accepted(q50[q50.domain == domain], codec)
            column = f'{codec}_psnr_parameter'
            medians.append(float(f[column].median()))
            replicates.append(bootstrap_median(f, column, rng))
            q1, q3 = f[column].quantile([.25, .75])
            spreads.append(dict(domain=domain, codec=codec, q=50, quartile_ratio=float(q3 / q1)))
        reps = np.stack(replicates)
        ratio = float(np.max(medians) / np.min(medians))
        lo, hi = ci(reps.max(axis=0) / reps.min(axis=0))
        ratios.append(dict(codec=codec, q=50, ratio=ratio, lo=lo, hi=hi))
        values.update({f'Ratio{symbol}': ratio, f'Ratio{symbol}lo': lo, f'Ratio{symbol}hi': hi})
        spread = [r['quartile_ratio'] for r in spreads if r['codec'] == codec]
        values.update({f'Iqr{symbol}min': float(np.min(spread)), f'Iqr{symbol}max': float(np.max(spread))})
        for metric in ['psnr', 'ssim']:
            for domain in DOMAINS:
                f = accepted(q50[q50.domain == domain], codec, metric)
                f['saving_pct'] = 100 * (1 - f[f'{codec}_{metric}_encoded_bytes'] / f.jpeg_bytes)
                lo, hi = ci(bootstrap_median(f, 'saving_pct', rng))
                rates.append(dict(domain=domain, codec=codec, metric=metric, q=50, n=len(f),
                                  median=float(f.saving_pct.median()), lo=lo, hi=hi))
            vals = [r['median'] for r in rates if r['codec'] == codec and r['metric'] == metric]
            stem = ('Jtk' if codec == 'jpeg2000' else 'Jxl') + metric.title()
            values.update({f'Save{stem}Min': float(np.min(vals)), f'Save{stem}Max': float(np.max(vals))})
    for metric in ['psnr', 'ssim']:
        for domain in [*DOMAINS, 'overall']:
            f = q50 if domain == 'overall' else q50[q50.domain == domain]
            f = accepted(accepted(f, 'jpeg2000', metric), 'jpegxl', metric)
            won = int((f[f'jpegxl_{metric}_encoded_bytes'] < f[f'jpeg2000_{metric}_encoded_bytes']).sum())
            wins.append(dict(domain=domain, metric=metric, q=50, n=len(f), wins=won, win_pct=100 * won / len(f) if len(f) else float('nan')))
        values['JxlWins' + metric.title()] = wins[-1]['win_pct']
    ssim_domain_wins = [r['win_pct'] for r in wins if r['metric'] == 'ssim' and r['domain'] != 'overall']
    values['JxlWinsSsimDomainsMajority'] = sum(x > 50 for x in ssim_domain_wins) if np.isfinite(ssim_domain_wins).all() else None
    for codec in CODECS:
        for metric in ['psnr', 'ssim']:
            key = 'Unreach' + ('Jtk' if codec == 'jpeg2000' else 'Jxl') + metric.title()
            values[key] = 100 * pairs[f'{codec}_{metric}_status'].eq('unreachable_low').mean()
    status = read('status_counts.csv')
    worst = status[(status.codec == 'jpegxl') & (status.metric == 'psnr')].copy()
    worst['pct'] = 100 * worst.unreachable_low / worst.total
    w = worst.sort_values(['pct', 'domain', 'q'], ascending=[False, True, True]).iloc[0]
    values.update(UnreachWorstDomain=LABELS[w.domain], UnreachWorstQ=int(w.q), UnreachWorstPct=float(w.pct),
                  NonmonoPct=100 * float(status.nonmonotone.sum()) / float(status.total.sum()),
                  UnmatchedPct=100 * float(status.unmatched.sum()) / float(status.total.sum()))
    for name, data in [('rate_savings.csv', rates), ('between_domain_ratios.csv', ratios),
                       ('within_domain_spread.csv', spreads), ('ranking.csv', wins)]:
        pd.DataFrame(data).to_csv(OUT / name, index=False)
    write_json('analysis_values.json', values)
    write_json('analysis_check.json', dict(status='passed', bootstrap_replicates=REPLICATES, seed=0,
        bootstrap='Percentile intervals; independent group resampling within each domain and eligible codec/metric subset',
        equivalence_rows=len(eq), rate_rows=len(rates), accepted_statuses=ACCEPTED))
    print('Pair analysis complete', flush=True)
