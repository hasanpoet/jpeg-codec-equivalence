"""Exact manuscript values and a complete, explicitly denominated run summary."""


import json


import numpy as np


import pandas as pd


from .analysis import ROOT, OUT, GEN, DOMAINS, LABELS, CODECS, accepted, read, write_json


from .payload import VALUE_NAMES, run_payload


from .formatting import format_number


REQUIRED = '''HasPathology Ndomains NimgTotal NgroupsTotal NpairsTotal
VerPillow VerLibjpeg VerOpenjpeg VerLibjxl VerImagecodecs
JtkResidMedian JtkResidMax JxlResidMax HeaderPctMax
UnreachJtkPsnr UnreachJxlPsnr UnreachJtkSsim UnreachJxlSsim
UnreachWorstDomain UnreachWorstQ UnreachWorstPct
RatioR RatioRlo RatioRhi RatioD RatioDlo RatioDhi IqrRmin IqrRmax IqrDmin IqrDmax
SaveJtkPsnrMin SaveJtkPsnrMax SaveJxlPsnrMin SaveJxlPsnrMax
SaveJtkSsimMin SaveJtkSsimMax SaveJxlSsimMin SaveJxlSsimMax
JxlWinsPsnr JxlWinsSsim JxlWinsSsimDomainsMajority
ErrTzeroJtk ErrToneJtk ErrCzeroJtk ErrConeJtk ErrFloorJtk
ErrTzeroJxl ErrToneJxl ErrCzeroJxl ErrConeJxl ErrFloorJxl
DiffToneConeJtk DiffToneConeJtkLo DiffToneConeJtkHi
DiffToneConeJxl DiffToneConeJxlLo DiffToneConeJxlHi
DiffToneCzeroJtk DiffToneCzeroJtkLo DiffToneCzeroJtkHi
DiffToneCzeroJxl DiffToneCzeroJxlLo DiffToneCzeroJxlHi
RsqRmin RsqRmax RsqDmin RsqDmax ErrConeBppJtk ErrConeBppJxl
NonmonoPct UnmatchedPct HeaderPctQfifty JtkResidPnn JxlResidPnn
RsqDminDomain SaveJtkSsimMinDomain'''.split()


REQUIRED += VALUE_NAMES


DISPLAY_NAMES = dict(natural='natural photographs', xray='chest radiographs',
                     mri='brain MRI', aerial='aerial imagery')


COUNTS = {'HasPathology', 'Ndomains', 'NimgTotal', 'NgroupsTotal', 'NpairsTotal',
          'UnreachWorstQ', 'JxlWinsSsimDomainsMajority', 'HdrJpegBytes', 'HdrJtkBytes', 'HdrJxlBytes'}


def format_value(name, value):
    if isinstance(value, str):
        return value.replace('_', r'\_')
    if value is None or not np.isfinite(value):
        return 'NA'
    if name in COUNTS:
        return format_number(int(value), ',').replace(',', '{,}')
    percent = name.startswith(('Save', 'Unreach', 'JxlWins')) or name in ['HeaderPctMax', 'HeaderPctQfifty', 'NonmonoPct', 'UnmatchedPct']
    return format_number(value, '.1f' if percent else '.2f')


SOURCES = {'natural': 'BSDS500', 'xray': 'Montgomery + Shenzhen', 'mri': 'IXI', 'aerial': 'UC Merced'}

def table1(frame):
    rows = []
    for domain in DOMAINS:
        d = frame[frame.domain == domain]
        channels = ', '.join(format_number(c, 'd') for c in sorted(d.channels.unique()))
        unit = 'subjects' if domain == 'mri' else 'patients' if domain == 'xray' else 'images'
        n = format_number(len(d), ',').replace(',', '{,}')
        groups = format_number(d.group.nunique(), ',')
        rows.append(f'{LABELS[domain]} & {SOURCES[domain]} & {channels} & {groups} {unit} & {n} ' + r'\\')
    return '\n'.join(rows) + '\n'


def regenerate():
    """Regenerate every published artifact from saved numerical measurements only."""
    import time
    from .search import counts
    from .analysis import analyze_pairs
    from .models import summarize_psnr
    from .robustness import summarize_ssim,dispersion,table3,VALUE_NAMES as SSIM_NAMES
    from .figures import generate_figures
    from .paper_numbers import generate
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True);GEN.mkdir(parents=True,exist_ok=True)
    manifest,pairs=read('manifest.csv'),read('pairs.csv')
    if manifest.sha256.duplicated().any() or pairs.duplicated(['domain','image_id','q']).any():raise RuntimeError('Duplicate measurement identity')
    counts(pairs)
    analyze_pairs()
    print('Search summaries and group-bootstrap intervals regenerated',flush=True)
    values=json.loads((OUT/'analysis_values.json').read_text())
    values.update(summarize_psnr(read('model_evaluation.csv')))
    values.update(summarize_ssim(read('ssim_model_evaluation.csv')))
    values.update(dispersion(pairs))
    versions=json.loads((ROOT/'results/encoder_versions.json').read_text())
    values.update(HasPathology=0,Ndomains=int(manifest.domain.nunique()),NimgTotal=len(manifest),NgroupsTotal=int(manifest.groupby('domain').group.nunique().sum()),NpairsTotal=len(pairs),VerPillow=versions['Pillow'],VerLibjpeg=versions['LibjpegTurbo'],VerOpenjpeg=versions['Openjpeg'],VerLibjxl=versions['Libjxl'].removeprefix('libjxl '),VerImagecodecs=versions['Imagecodecs'])
    jtk=accepted(pairs,'jpeg2000').jpeg2000_psnr_residual;jxl=accepted(pairs,'jpegxl').jpegxl_psnr_residual
    h,r2,rates=read('header_measurements.csv'),read('complexity_r2.csv'),read('rate_savings.csv')
    d=r2[r2.codec=='jpegxl'];rates=rates[(rates.codec=='jpeg2000')&(rates.metric=='ssim')&(rates.q==50)]
    values.update(JtkResidMedian=float(jtk.median()),JtkResidMax=float(jtk.max()),JxlResidMax=float(jxl.max()),JtkResidPnn=float(jtk.quantile(.99)),JxlResidPnn=float(jxl.quantile(.99)),HeaderPctMax=float(h.header_pct.max()),HeaderPctQfifty=float(h.loc[h.q==50,'header_pct'].max()),RsqDminDomain=DISPLAY_NAMES[d.loc[d.r2.idxmin(),'domain']],SaveJtkSsimMinDomain=DISPLAY_NAMES[rates.loc[rates['median'].idxmin(),'domain']])
    values.update(run_payload())
    d=read('ssim_complexity_r2.csv');d=d[(d.codec=='jpegxl')&(d.q==50)]
    values['RsqDSsimMinDomain']=DISPLAY_NAMES[d.loc[d.r2.idxmin(),'domain']]
    names=REQUIRED+SSIM_NAMES+['RsqDSsimMinDomain']
    if set(names)!=set(values):raise RuntimeError('Manuscript value schema mismatch')
    (GEN/'values.tex').write_text('\n'.join(r'\setval{'+k+'}{'+format_value(k,values[k])+'}' for k in names)+'\n')
    (GEN/'table1_body.tex').write_text(table1(manifest))
    (GEN/'table3_body.tex').write_text(table3(read('model_error_summary.csv')))
    write_json('manuscript_values.json',values)
    # Publish all nine SSIM levels, using the same accepted-target definitions.
    rows=[]
    for codec in CODECS:
        for (domain,q),cell in pairs.groupby(['domain','q']):
            f=accepted(cell,codec,'ssim');q1,med,q3=f[codec+'_ssim_parameter'].quantile([.25,.5,.75])
            rows.append(dict(domain=domain,q=int(q),codec=codec,n=len(f),total=len(cell),q1=q1,median=med,q3=q3))
    pd.DataFrame(rows).to_csv(OUT/'equivalence_ssim_full.csv',index=False)
    generate_figures();generate()
    write_json('regeneration_check.json',dict(status='passed',seconds=time.monotonic()-start,value_count=len(values),encoding=False))
    print(f'Regenerated {len(values)} values, three tables, three figures and paper_numbers.csv in {time.monotonic()-start:.1f}s',flush=True)
