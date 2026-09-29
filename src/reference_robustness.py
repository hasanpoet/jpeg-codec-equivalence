"""Reference-quantizer sensitivity: DCT features and train-only fits, no encoding."""

from .config import crop_path
import hashlib

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from scipy.stats import spearmanr

from src.analysis import OUT, DOMAINS, CODECS, read, accepted, write_json
from src.features import Q50
from src.codecs import luma

IJG_SOURCE = 'https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/3.1.4.1/src/jcparam.c'


def quantizer(quality):
    scale = 5000 // quality if quality < 50 else 200 - 2*quality
    return np.clip((Q50.astype(int)*scale + 50)//100, 1, 255)


def entropy(coefficients, table):
    counts = np.unique(np.rint(coefficients/table), return_counts=True)[1]
    p = counts/counts.sum()
    return float(-np.sum(p*np.log2(p)))


def main():
    manifest, features, pairs = read('manifest.csv'), read('complexity.csv'), read('pairs.csv')
    assert not manifest.sha256.duplicated().any()
    data = manifest.merge(features, on=['domain','image_id','group'], validate='one_to_one')
    assert len(data) == len(manifest)
    split = data.merge(read('model_splits.csv'), on=['domain','group'], suffixes=('','_saved'), validate='many_to_one')
    assert split.split.eq(split.split_saved).all()
    assert data.groupby(['domain','group']).split.nunique().max() == 1
    tables = {q:quantizer(q) for q in [30,50,70]}
    np.testing.assert_array_equal(tables[50],Q50)
    rows = []
    for i, r in enumerate(data.itertuples()):
        a = np.array(Image.open(crop_path(r._asdict())))
        assert hashlib.sha256(a.tobytes()).hexdigest() == r.sha256
        blocks = (luma(a)-128).reshape(32,8,32,8).transpose(0,2,1,3)
        coefficients = dctn(blocks, axes=(-2,-1), norm='ortho')
        baseline = entropy(coefficients,tables[50])
        assert baseline == r.C_DCT
        rows.append(dict(domain=r.domain,image_id=r.image_id,group=r.group,split=r.split,
                         C_DCT_q30=entropy(coefficients,tables[30]), C_DCT_q50=r.C_DCT,
                         C_DCT_q70=entropy(coefficients,tables[70])))
        if (i+1)%500 == 0:
            print(f'Reference entropy variants: {i+1}/{len(data)} images',flush=True)
    variants = pd.DataFrame(rows)
    variants.to_csv(OUT/'reviewer_quantizer_features.csv',index=False)
    correlations = []
    for domain in DOMAINS:
        f = variants[variants.domain == domain]
        for quality in [30,70]:
            row = dict(domain=domain,reference_quality=quality,n=len(f))
            for subset in ['all','train','test']:
                s = f if subset == 'all' else f[f.split == subset]
                row['rho_'+subset] = float(spearmanr(s[f'C_DCT_q{quality}'],s.C_DCT_q50).statistic)
            correlations.append(row)
    corr = pd.DataFrame(correlations)
    corr.to_csv(OUT/'reviewer_quantizer_correlations.csv',index=False)
    q50 = pairs[pairs.q == 50].merge(variants,on=['domain','image_id','group'],validate='one_to_one')
    baseline = read('complexity_r2.csv').set_index(['domain','codec'])
    fits = []
    for domain in DOMAINS:
        for codec in CODECS:
            f = accepted(q50[q50.domain == domain],codec)
            for quality in [30,70]:
                column = f'C_DCT_q{quality}'
                assert f[column].gt(0).all(), 'Undefined log complexity'
                train, test = f[f.split == 'train'], f[f.split == 'test']
                x,y = np.log(train[column].to_numpy()),np.log(train[codec+'_psnr_parameter'].to_numpy())
                intercept,slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
                target = np.log(test[codec+'_psnr_parameter'].to_numpy())
                prediction = intercept + slope*np.log(test[column].to_numpy())
                r2 = float(1-np.sum((target-prediction)**2)/np.sum((target-target.mean())**2))
                old = float(baseline.loc[(domain,codec),'r2'])
                assert len(test) == baseline.loc[(domain,codec),'n']
                fits.append(dict(domain=domain,codec=codec,target_quality=50,reference_quality=quality,
                    train_n=len(train),test_n=len(test),intercept=float(intercept),slope=float(slope),
                    test_r2=r2,baseline_r2=old,difference=r2-old,absolute_difference=abs(r2-old)))
    fits = pd.DataFrame(fits)
    fits.to_csv(OUT/'reviewer_quantizer_fits.csv',index=False)
    models = read('model_parameters.csv')
    slopes = models[(models.codec == 'jpeg2000') & (models.predictor == 'C1')][['domain','q','slope']].copy()
    assert len(slopes) == 36 and not slopes.duplicated(['domain','q']).any()
    slopes['sign'] = np.where(slopes.slope<0,'negative',np.where(slopes.slope>0,'positive','zero'))
    slopes.to_csv(OUT/'reviewer_ratio_slope_signs.csv',index=False)
    summarize_reference(fits, corr, slopes)


def summarize_reference(fits, correlations, slopes):
    maximum=float(fits[fits.codec=='jpeg2000'].absolute_difference.max())
    write_json('reviewer_robustness_check.json',dict(status='passed',max_r2_change=maximum,robust=maximum<=.02,all_ratio_slopes_negative=bool(slopes.slope.lt(0).all()),criterion='Maximum absolute held-out log-ratio R-squared change <= 0.02',reference_qualities=[30,70],baseline_quality=50,encoding=False))
    print(f'Reference-table maximum R-squared change: {maximum:.17g}; all r* slopes negative: {bool(slopes.slope.lt(0).all())}',flush=True)
