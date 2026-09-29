"""Audit manuscript quantities that are outside the generated value macros."""
import pandas as pd
from .analysis import ROOT,OUT,read,write_json
from .reporting import DISPLAY_NAMES
from .formatting import format_number

def generate():
    rows=[]
    current=read('manifest.csv')[['domain','image_id']].sort_values(['domain','image_id']).reset_index(drop=True)
    published=pd.read_csv(ROOT/'data/manifest.csv',dtype={'image_id':str})[['domain','image_id']].sort_values(['domain','image_id']).reset_index(drop=True)
    published_scope=current.equals(published)
    def add(name,value,precision,unit,source,expected=None,domain='',q='',codec='',note=''):
        display=str(value) if isinstance(value,str) else format_number(value,precision)
        status=('not_present_in_workspace_manuscript' if expected is None else 'match' if display==expected else 'mismatch') if published_scope else 'subset_not_compared'
        rows.append(dict(name=name,domain=domain,q=q,codec=codec,value=value,display=display,unit=unit,manuscript_display=expected or '',status=status,source=source,note=note))
    ranking=read('ranking.csv');f=ranking[(ranking.domain=='natural')&(ranking.metric=='ssim')]
    add('NaturalSsimJxlMajorityPct',float(f.iloc[0].win_pct),'.1f','percent','ranking.csv','53.5','natural',50,'jpegxl')
    diffs=read('ssim_model_differences.csv')
    for codec,expected in [('jpeg2000',['14.2','13.2','15.4']),('jpegxl',['2.0','1.7','2.3'])]:
        row=diffs[diffs.codec==codec].iloc[0]
        for key,printed in zip(['difference','lo','hi'],expected):
            add('SsimT1MinusC1_'+key,float(row[key]),'.1f','10^-3 SSIM','ssim_model_evaluation.csv',printed,codec=codec)
    r2=read('ssim_complexity_r2.csv');f=r2[(r2.codec=='jpegxl')&(r2.q==50)];row=f.loc[f.r2.idxmin()]
    add('LowestSsimDistanceRsqDomain',DISPLAY_NAMES[row.domain],'','display name','ssim_complexity_r2.csv','aerial imagery',row.domain,50,'jpegxl')
    add('LowestSsimDistanceRsqUnrounded',float(row.r2),'.17g','R-squared','ssim_complexity_r2.csv',None,row.domain,50,'jpegxl','The manuscript displays 0.00 and describes no explanatory power; this row preserves the unrounded value.')
    fits=read('reviewer_quantizer_fits.csv');f=fits[fits.codec=='jpeg2000'];row=f.loc[f.absolute_difference.idxmax()]
    add('ReferenceTableMaxRatioRsqChange',float(row.absolute_difference),'.17g','absolute R-squared change','reviewer_quantizer_fits.csv',None,row.domain,int(row.reference_quality),'jpeg2000','Requested audit quantity; no numerical statement found in the supplied workspace manuscript.')
    slopes=read('model_parameters.csv');slopes=slopes[(slopes.codec=='jpeg2000')&(slopes.predictor=='C1')]
    for row in slopes.itertuples():
        sign='negative' if row.slope<0 else 'positive' if row.slope>0 else 'zero'
        add('RatioC1SlopeSign',sign,'','sign','model_parameters.csv',None,row.domain,int(row.q),'jpeg2000',f'slope={row.slope:.17g}; no explicit slope-sign statement in workspace manuscript')
    add('AllRatioC1SlopesNegative',str(bool(slopes.slope.lt(0).all())).lower(),'','boolean','model_parameters.csv',None)
    sanity=read('sanity_mct.csv');ratio=sanity.j2k_defaults_bpp/sanity.j2k_both_fixed_bpp;row=sanity.loc[ratio.idxmax()]
    add('PillowDefaultsMaxFactor',float(ratio.max()),'.1f','ratio','sanity_mct.csv','2.7',row.domain,int(row.jpeg_quality),'jpeg2000','Maximum of domain-quality mean default bpp / corrected bpp; nearest one-decimal rounding.')
    raw=read('sanity_mct_per_image.csv');ratio=raw.j2k_defaults_bpp/raw.j2k_both_fixed_bpp;row=raw.loc[ratio.idxmax()]
    add('PillowDefaultsIndividualMaxFactor',float(ratio.max()),'.17g','ratio','sanity_mct_per_image.csv',None,row.domain,int(row.jpeg_quality),'jpeg2000','Individual-image maximum is a different estimand from the domain-quality means.')
    # Explicit method constants and experimental design counts also occur outside
    # the numerical value macros. These are distinguished by their source column.
    from .codecs import BOUNDS,QUALITIES
    for codec,(low,high) in BOUNDS.items():
        for tag,value in [('Lower',low),('Upper',high)]:
            add('SearchBound'+tag,float(value),'.4g','parameter','src/codecs.py',format_number(value,'.4g'),codec=codec)
    for q in QUALITIES:add('JpegQualityLevel',q,'d','quality','src/codecs.py',str(q),q=q)
    for name,value,unit,source in [
        ('CropWidth',256,'pixels','data/manifest.csv'),('CropHeight',256,'pixels','data/manifest.csv'),
        ('XrayDownsampleFactor',2,'integer factor','data/manifest.csv'),('MriSlicesPerSubject',5,'slices','data/manifest.csv'),
        ('ImageBitDepth',8,'bits','src/codecs.py'),('DctBlockSide',8,'pixels','src/features.py'),
        ('ReferenceJpegQuality',50,'quality','src/features.py'),('BootstrapReplicates',10000,'replicates','src/analysis.py'),
        ('ConfidenceLevel',95,'percent','src/analysis.py'),('TrainingFraction',80,'percent','src/models.py'),
        ('TestFraction',20,'percent','src/models.py'),('JxlEffort',7,'effort','src/codecs.py'),
        ('NonmonotoneScanPoints',64,'evaluations','src/search.py'),('ErrorTailPercentile',90,'percentile','src/models.py'),
        ('ResidualPercentile',99,'percentile','src/reporting.py'),('UnreachableFigureThreshold',5,'percent','src/figures.py')]:
        add(name,value,'d',unit,source,str(value))
    add('ScanPsnrTolerance',.1,'.1f','dB','src/search.py','0.1')
    add('ScanSsimTolerance',.001,'.3f','SSIM','src/search.py','0.001')
    pd.DataFrame(rows).to_csv(OUT/'paper_numbers.csv',index=False)
    mismatches=[x for x in rows if x['status']=='mismatch']
    write_json('paper_numbers_check.json',dict(status='mismatches_reported' if mismatches else 'passed',mismatches=mismatches,not_present_in_workspace_manuscript=sorted({x['name'] for x in rows if x['status']=='not_present_in_workspace_manuscript'})))
    for x in mismatches:print(f"Manuscript mismatch: {x['name']}: stated {x['manuscript_display']}, calculated {x['display']} ({x['value']})",flush=True)
