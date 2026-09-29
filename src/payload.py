"""Payload-only sensitivity check using the measured, fixed header definition."""


import json


import numpy as np


import pandas as pd


from .analysis import OUT, DOMAINS, CODECS, accepted, read, write_json


VALUE_NAMES = '''SaveJtkPsnrPayMin SaveJtkPsnrPayMax SaveJxlPsnrPayMin SaveJxlPsnrPayMax
JxlWinsPsnrPay JxlWinsSsimPay HdrJpegBytes HdrJtkBytes HdrJxlBytes'''.split()


KEY = ['domain', 'image_id', 'q', 'codec', 'metric']


def payload_measurements(headers, pairs):
    expected = [pairs[['domain', 'image_id', 'q', 'jpeg_bytes']].rename(columns={'jpeg_bytes': 'encoded_bytes'})
                .assign(codec='jpeg', metric='target')]
    for codec in CODECS:
        for metric in ['psnr', 'ssim']:
            column = f'{codec}_{metric}_encoded_bytes'
            expected.append(accepted(pairs, codec, metric)[['domain', 'image_id', 'q', column]]
                            .rename(columns={column: 'encoded_bytes'}).assign(codec=codec, metric=metric))
    expected = pd.concat(expected, ignore_index=True)
    check = expected.merge(headers, on=KEY, how='outer', validate='one_to_one', indicator=True,
                           suffixes=('_expected', ''))
    if not check._merge.eq('both').all() or not check.encoded_bytes.eq(check.encoded_bytes_expected).all():
        raise RuntimeError('Header coverage or file-size mismatch')
    data = headers.copy()
    if not np.isfinite(data[['header_bytes', 'encoded_bytes']].to_numpy()).all():
        raise RuntimeError('Corrupted byte counts')
    if not (data[['header_bytes', 'encoded_bytes']] % 1 == 0).all().all():
        raise RuntimeError('Noninteger byte counts')
    data['payload_bytes'] = data.encoded_bytes - data.header_bytes
    if not ((data.header_bytes > 0) & (data.payload_bytes > 0)).all():
        raise RuntimeError('Invalid header or payload length')
    data['payload_bpp'] = 8 * data.payload_bytes / (256 * 256)
    return data


def majority(jxl, jtk, n):
    return 'JPEG-XL' if 2*jxl > n else 'JPEG2000' if 2*jtk > n else 'no majority'


def payload_statistics(data):
    q50 = data[data.q == 50]
    ids = ['domain', 'image_id', 'q']
    target = q50[q50.codec == 'jpeg'][ids + ['payload_bpp']].rename(columns={'payload_bpp': 'jpeg_payload_bpp'})
    rates, rankings, values = [], [], {}
    for codec, tag in zip(CODECS, ['Jtk', 'Jxl']):
        f = q50[(q50.codec == codec) & (q50.metric == 'psnr')].merge(target, on=ids, validate='one_to_one')
        f['saving_pct'] = 100 * (1 - f.payload_bpp / f.jpeg_payload_bpp)
        for domain in DOMAINS:
            d = f[f.domain == domain]
            rates.append(dict(domain=domain, codec=codec, metric='psnr', q=50, n=len(d), median=float(d.saving_pct.median())))
        medians = [r['median'] for r in rates if r['codec'] == codec]
        values[f'Save{tag}PsnrPayMin'] = float(np.min(medians))
        values[f'Save{tag}PsnrPayMax'] = float(np.max(medians))
    for metric in ['psnr', 'ssim']:
        f = q50[(q50.codec == 'jpeg2000') & (q50.metric == metric)].merge(
            q50[(q50.codec == 'jpegxl') & (q50.metric == metric)], on=ids, suffixes=('_jtk', '_jxl'), validate='one_to_one')
        for domain in [*DOMAINS, 'overall']:
            d = f if domain == 'overall' else f[f.domain == domain]
            for basis, column in [('file', 'encoded_bytes'), ('payload', 'payload_bytes')]:
                xl, tk = d[column+'_jxl'], d[column+'_jtk']
                won, lost, tied = int((xl < tk).sum()), int((xl > tk).sum()), int((xl == tk).sum())
                pct = 100 * won / len(d) if len(d) else float('nan')
                rankings.append(dict(domain=domain, metric=metric, q=50, basis=basis, n=len(d),
                                     wins=won, jtk_wins=lost, ties=tied, win_pct=pct,
                                     majority=majority(won, lost, len(d))))
                if domain == 'overall' and basis == 'payload':
                    values['JxlWins'+metric.title()+'Pay'] = pct
    header_summary = []
    for codec, tag in [('jpeg', 'Jpeg'), ('jpeg2000', 'Jtk'), ('jpegxl', 'Jxl')]:
        f = q50[q50.codec == codec]
        median = float(f.header_bytes.median())
        header_summary.append(dict(codec=codec, q=50, n=len(f), median_header_bytes=median))
        values['Hdr'+tag+'Bytes'] = int(round(median))
    return values, pd.DataFrame(rates), pd.DataFrame(rankings), pd.DataFrame(header_summary)


def run_payload():
    header_check = json.loads((OUT / 'header_check.json').read_text())
    if header_check['status'] != 'passed':
        raise RuntimeError('Header measurements not validated')
    data = payload_measurements(read('header_measurements.csv'), read('pairs.csv'))
    if len(data) != header_check['files']:
        raise RuntimeError('Header record count mismatch')
    values, rates, ranking, header_summary = payload_statistics(data)
    # Reproducing the original full-file rankings checks the eligibility join.
    previous = read('ranking.csv').merge(ranking[ranking.basis == 'file'],
        on=['domain', 'metric', 'q'], suffixes=('_old', '_new'), validate='one_to_one')
    if len(previous) != 10 or not previous.n_old.eq(previous.n_new).all() or not previous.wins_old.eq(previous.wins_new).all():
        raise RuntimeError('Full-file ranking changed during payload check')
    for name, frame in [('payload_measurements.csv', data), ('payload_rate_savings.csv', rates),
                        ('payload_ranking.csv', ranking), ('payload_header_summary.csv', header_summary)]:
        frame.to_csv(OUT / name, index=False)
    write_json('payload_values.json', values)
    return values
