"""Exact comparisons for the published replay subset."""
import json
from pathlib import Path
import pandas as pd

def compare_replay(published,replayed,output):
    keys=['domain','image_id','q']
    def read(p):return pd.read_csv(p,dtype={'image_id':str,'group':str},float_precision='round_trip').set_index(keys)
    a,b=read(published),read(replayed)
    if b.index.duplicated().any():raise RuntimeError('Duplicate replay keys')
    a=a.loc[b.index];counts=[];changes=[]
    if set(a.columns)!=set(b.columns):raise RuntimeError('Replay column schema mismatch')
    for c in a.columns:
        same=a[c].eq(b[c])|(a[c].isna()&b[c].isna())
        if not same.all():
            category='execution_counter' if c.endswith('new_encode_calls') else 'scientific_or_search'
            counts.append(dict(column=c,different_rows=int((~same).sum()),category=category))
            for key in a.index[~same]:changes.append(dict(domain=key[0],image_id=key[1],q=key[2],column=c,published=a.loc[key,c],replayed=b.loc[key,c],category=category))
    report=dict(pairs=len(b),images=len(b.reset_index()[['domain','image_id']].drop_duplicates()),common_columns=len(a.columns),scientific_differences=sum(x['different_rows'] for x in counts if x['category']=='scientific_or_search'),execution_counter_differences=sum(x['different_rows'] for x in counts if x['category']=='execution_counter'),columns=counts,comparison='Exact equality after round-trip CSV parsing; execution counters reported separately')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'replay_check.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame(changes,columns=['domain','image_id','q','column','published','replayed','category']).to_csv(output/'replay_differences.csv',index=False)
    return report


def validate_pairs(manifest,pairs):
    import numpy as np
    from .codecs import QUALITIES,ProtocolFailure
    keys=['domain','image_id']
    if manifest.sha256.duplicated().any() or manifest.duplicated(keys).any() or pairs.duplicated(keys+['q']).any():raise ProtocolFailure('Duplicate hashes or measurement identities')
    if len(pairs)!=len(manifest)*len(QUALITIES):raise ProtocolFailure('Incomplete image-quality coverage')
    joined=pairs.merge(manifest[keys+['group']],on=keys,how='outer',validate='many_to_one',indicator=True,suffixes=('','_manifest'))
    if not joined._merge.eq('both').all() or not joined.group.eq(joined.group_manifest).all():raise ProtocolFailure('Measurement/manifest group mismatch')
    if not pairs.groupby(keys).q.apply(lambda q:set(q)==set(QUALITIES)).all():raise ProtocolFailure('Missing quality levels')
    for codec in ['jpeg2000','jpegxl']:
        for metric in ['psnr','ssim']:
            prefix=codec+'_'+metric+'_';status=pairs[prefix+'status']
            if not status.isin(['matched','matched_scan','unmatched','unreachable_low','unreachable_high']).all():raise ProtocolFailure('Unknown search status')
            if not np.isfinite(pairs[prefix+'residual']).all():raise ProtocolFailure('Invalid search residual')
            if not np.allclose(pairs[prefix+'residual'],abs(pairs[prefix+metric]-pairs['jpeg_'+metric]),rtol=0,atol=1e-12):raise ProtocolFailure('Residual/fidelity mismatch')
            scan=pairs[prefix+'initially_nonmonotone']
            if not pairs.loc[scan,prefix+'scan_calls'].eq(64).all() or not pairs.loc[~scan,prefix+'scan_calls'].eq(0).all():raise ProtocolFailure('Scan coverage mismatch')
            tolerance=.1 if metric=='psnr' else .001
            if not pairs.loc[status.eq('matched_scan'),prefix+'residual'].le(tolerance).all() or not pairs.loc[status.eq('unmatched'),prefix+'residual'].gt(tolerance).all():raise ProtocolFailure('Scan acceptance mismatch')
            if not pairs[prefix+'encoded_bytes'].gt(0).all() or not np.allclose(pairs[prefix+'bpp'],pairs[prefix+'encoded_bytes']*8/256**2,rtol=0,atol=1e-14):raise ProtocolFailure('Invalid byte counts or bpp')
    return True

def validate_model_measurements(errors,metric):
    import numpy as np
    from .codecs import ProtocolFailure
    key=['domain','image_id','q','codec','predictor']
    column='error_db' if metric=='psnr' else 'error_ssim'
    if errors.duplicated(key).any() or not np.isfinite(errors[column]).all():raise ProtocolFailure('Duplicate or invalid model measurements')
    if ((errors.domain=='xray')&(errors.q==10)&(errors.codec=='jpegxl')).any():raise ProtocolFailure('Excluded model cell present')
    if not np.allclose(errors[column],abs(errors['achieved_'+metric]-errors['target_'+metric]),rtol=0,atol=1e-12):raise ProtocolFailure('Model fidelity/error mismatch')
    return True
