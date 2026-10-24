"""Measured supplement table bodies, figures, values and worked example."""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .revision import ROOT, DATA, OUT, SUBSET, read, baseline, CODECS, METRICS, save_json

GEN = Path(os.environ.get('EQUIVALENCE_REVISION_PAPER', ROOT / 'paper/revision')).resolve()
# The superseded lowest-index MRI selection survives only as this frozen audit.
LOWEST_INDEX_AUDIT = ROOT / 'results/revision/mri_representative_bias.csv'


def tex(value):
    labels={'jpeg2000':'JPEG~2000','jpegxl':'JPEG~XL','natural':'Natural','xray':'X-ray','mri':'MRI','aerial':'Aerial',
        'baseline':'Selected','effort3':'Effort 3','effort9':'Effort 9','pillow_transforms':'Default transforms','jpeg444':'JPEG 4:4:4',
        'psnr':'PSNR','ssim':'SSIM','crop256':'Original 256','crop256_new':'New 256','larger':'Larger',
        'domain':'Domain','config':'Configuration','codec':'Codec','metric':'Metric','variant':'Size',
        'parameter_median':'Median setting','parameter_iqr_ratio':'Q3/Q1','accepted':'Accepted','total':'All',
        'file_bpp_median':'File bpp','payload_bpp_median':'Payload bpp','header_pct_median':'Header (%)',
        'between_domain_ratio':'Max/min','n_domains':'Domains','seed':'Seed','predictor':'Model',
        'mean':'Mean','p90':'P90','left':'Left','right':'Right','difference':'Difference','lo':'95% low','hi':'95% high'}
    return labels.get(str(value),str(value)).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')


def body(frame, columns, digits=3):
    lines=[]
    for row in frame[columns].itertuples(index=False,name=None):
        values=[r'--' if pd.isna(v) else format(v,f'.{digits}f') if isinstance(v,(float,np.floating)) else tex(v) for v in row]
        lines.append(' & '.join(values)+r'\\')
    return '\n'.join(lines)+'\n'


def coefficient_body(summary):
    short=dict(jpeg2000='J2K',jpegxl='JXL')
    def cell(f,name):
        return f"{f[name+'_median']:.3f} [{f[name+'_min']:.3f},{f[name+'_max']:.3f}]; CI [{f[name+'_bootstrap_low']:.3f},{f[name+'_bootstrap_high']:.3f}]"
    lines=[f"{tex(f['domain'])} & {short[f['codec']]} & {f['metric']} & {int(f['q'])} & {cell(f,'intercept')} & {cell(f,'slope')}"+r'\\'
           for f in summary.to_dict('records')]
    return '\n'.join(lines)+'\n'


def figures(config,size,pred):
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
    colors=dict(natural='#0072B2',xray='#D55E00',mri='#009E73',aerial='#CC79A7')
    GEN.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3))
    labels=['effort3','baseline','effort9']
    for ax,metric in zip(axes,METRICS):
        f=config[(config.q==50)&(config.codec=='jpegxl')&(config.metric==metric)]
        for domain,cell in f.groupby('domain'):
            y=cell.set_index('config').reindex(labels).parameter_iqr_ratio
            ax.plot([3,7,9],y,'o-',label=domain,color=colors[domain])
        ax.set(title=metric.upper()+' matching',ylabel='Equivalent distance Q3/Q1',xlabel='JPEG XL effort',xticks=[3,7,9])
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(GEN/'figS1_configuration.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3))
    for ax,metric in zip(axes,METRICS):
        f=size[(size.q==50)&(size.metric==metric)]
        for (domain,codec),cell in f.groupby(['domain','codec']):
            cell=cell.copy()
            cell['variant']=cell.variant.replace({'crop256_new':'crop256'})
            y=cell.set_index('variant').reindex(['crop256','larger']).header_pct_median
            ax.plot(['256 crop','Larger source'],y,'o-' if codec=='jpeg2000' else 's--',label=f'{domain} / {codec}',color=colors[domain])
        ax.set(title=metric.upper()+' matching',ylabel='Median header share (%)')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(GEN/'figS2_size.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3))
    for ax,metric in zip(axes,METRICS):
        f=pred[pred.metric==metric]
        scale=1000 if metric=='ssim' else 1
        for codec in CODECS:
            for predictor,style in [('T1','--'),('C1','-'),('floor',':')]:
                d=f[(f.codec==codec)&(f.predictor==predictor)].sort_values('seed')
                ax.plot(d.seed,d['mean']*scale,style,marker='o',label=f'{codec} / {predictor}',color='#0072B2' if codec=='jpeg2000' else '#D55E00')
        ax.set(title=metric.upper()+' matching',xlabel='Grouped split seed',ylabel='Mean error ('+('1e-3 SSIM' if scale==1000 else 'dB')+')',xticks=range(5))
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(GEN/'figS3_prediction.pdf');plt.close(fig)


def render():
    config=read(OUT/'configuration_summary.csv')
    sizes=read(OUT/'size_summary.csv')
    pred=read(OUT/'prediction_equal_domain.csv')
    differences=read(OUT/'prediction_differences.csv')
    coefficients=read(OUT/'C1_coefficient_summary.csv')
    figures(config,sizes,pred)
    # Use a baseline held-out example so the illustration relates to the full study.
    evaluation=read(baseline('model_evaluation.csv'))
    feature=read(baseline('complexity.csv'))
    models=read(baseline('model_parameters.csv'))
    sample=evaluation[(evaluation.domain=='natural')&(evaluation.codec=='jpeg2000')&(evaluation.q==50)&(evaluation.predictor=='C1')].sort_values('image_id').iloc[0]
    c=float(feature[(feature.domain==sample.domain)&(feature.image_id==sample.image_id)].C_DCT.iloc[0])
    model=models[(models.domain=='natural')&(models.codec=='jpeg2000')&(models.q==50)&(models.predictor=='C1')].iloc[0]
    floor=evaluation[(evaluation.domain==sample.domain)&(evaluation.image_id==sample.image_id)&(evaluation.codec==sample.codec)&(evaluation.q==50)&(evaluation.predictor=='floor')].iloc[0]
    example=dict(domain=sample.domain,image_id=sample.image_id,q=50,C_DCT=c,intercept=float(model.intercept),slope=float(model.slope),
        predicted_parameter=float(sample.parameter),target_psnr=float(sample.target_psnr),achieved_psnr=float(sample.achieved_psnr),error_db=float(sample.error_db),search_residual=float(floor.error_db))
    save_json(OUT/'worked_example.json',example)
    manifest=read(baseline('manifest.csv'))
    eligible=read(DATA/'size_eligible.csv')
    size_counts=eligible.groupby('domain').size()
    # Compute between-domain dispersion separately at each sampled JPEG quality.
    dispersion=[]
    for (config_name,q,codec,metric),cell in config.groupby(['config','q','codec','metric']):
        cell=cell[cell.accepted>0]
        medians=cell.set_index('domain').parameter_median
        ratio=float(medians.max()/medians.min()) if len(medians) else np.nan
        dispersion.append(dict(config=config_name,q=int(q),codec=codec,metric=metric,
            n_domains=len(medians),subset_max_min=ratio,full_dataset_q50_max_min=np.nan))
    dispersion=pd.DataFrame(dispersion)
    full_pairs=read(baseline('pairs.csv'))
    for codec in CODECS:
        for metric in METRICS:
            col=f'{codec}_{metric}_parameter'
            full_medians=full_pairs[full_pairs.q==50].groupby('domain')[col].median()
            value=float(full_medians.max()/full_medians.min())
            ix=(dispersion.config=='baseline')&(dispersion.q==50)&(dispersion.codec==codec)&(dispersion.metric==metric)
            dispersion.loc[ix,'full_dataset_q50_max_min']=value
    dispersion.to_csv(OUT/'configuration_between_domain_by_q.csv',index=False)
    # Middle-slice representativeness audit for the corrected MRI subset.
    subset=read(SUBSET)
    features=read(baseline('complexity.csv'))
    mri_all=manifest[manifest.domain=='mri'].merge(
        features[['domain','image_id','C_DCT']],on=['domain','image_id'],validate='one_to_one')
    subject=mri_all.groupby('group').agg(subject_median_C_DCT=('C_DCT','median'),
        subject_min_slice=('slice_index_canonical','min'),subject_max_slice=('slice_index_canonical','max'),
        subject_n_slices=('image_id','size')).reset_index()
    mri=subset[subset.domain=='mri'].merge(subject,on='group',validate='one_to_one')
    mri['entropy_ratio_to_subject_median']=mri.C_DCT/mri.subject_median_C_DCT
    mri['at_subject_slice_edge']=(mri.slice_index_canonical.eq(mri.subject_min_slice)|
        mri.slice_index_canonical.eq(mri.subject_max_slice))
    mri.to_csv(OUT/'mri_middle_slice_representativeness.csv',index=False)
    old_ratio=float(read(LOWEST_INDEX_AUDIT).entropy_ratio_to_subject_median.median())
    mri_ratio=float(mri.entropy_ratio_to_subject_median.median())
    q50=full_pairs[full_pairs.q==50]
    mri_settings={codec:float(q50[q50.domain=='mri'][f'{codec}_psnr_parameter'].median()) for codec in CODECS}
    config50=config[(config.q==50)&(config.config=='baseline')&(config.metric=='psnr')]
    subset_settings={codec:float(config50[(config50.domain=='mri')&(config50.codec==codec)].parameter_median.iloc[0]) for codec in CODECS}
    setting_change={codec:100*(subset_settings[codec]/mri_settings[codec]-1) for codec in CODECS}
    def ratio(config_name,q,codec,metric='psnr'):
        return float(dispersion[(dispersion.config==config_name)&(dispersion.q==q)&(dispersion.codec==codec)&(dispersion.metric==metric)].subset_max_min.iloc[0])
    def full(codec):
        return float(dispersion[(dispersion.config=='baseline')&(dispersion.q==50)&(dispersion.codec==codec)&(dispersion.metric=='psnr')].full_dataset_q50_max_min.iloc[0])
    name=dict(jpeg2000='Jtk',jpegxl='Jxl')
    values=dict(RevSubsetJtkPsnrMaxMin=f"{ratio('baseline',50,'jpeg2000'):.3f}",RevSubsetJxlPsnrMaxMin=f"{ratio('baseline',50,'jpegxl'):.3f}",
        RevFullJtkPsnrMaxMin=f"{full('jpeg2000'):.3f}",RevFullJxlPsnrMaxMin=f"{full('jpegxl'):.3f}",
        RevLowestIndexEntropyRatio=f'{old_ratio:.3f}',RevMiddleSliceEntropyRatio=f'{mri_ratio:.3f}',
        RevMriJtkSettingDiffPct=str(int(np.ceil(abs(setting_change['jpeg2000'])))),RevMriJxlSettingDiffPct=str(int(np.ceil(abs(setting_change['jpegxl'])))))
    for codec in CODECS:
        for q,label in [(50,'Fifty'),(90,'Ninety')]:
            values[f'RevFourFourFour{name[codec]}Psnr{label}']=f"{ratio('jpeg444',q,codec):.3f}"
    exact=read(DATA/'size_exact_original_exclusions.csv')
    values.update(RevEligibleNatural=str(int(size_counts.get('natural',0))),RevTotalNatural=str(int((manifest.domain=='natural').sum())),
        RevEligibleXray=str(int(size_counts.get('xray',0))),RevTotalXray=str(int((manifest.domain=='xray').sum())),
        RevExactOriginalXrayExcluded=str(len(exact)))
    new256=sizes[(sizes.domain=='xray')&(sizes.variant=='crop256_new')&(sizes.q==50)&(sizes.codec=='jpeg2000')&(sizes.metric=='psnr')]
    if len(new256):
        values['RevXrayNewJtkPsnrMedian']=f'{float(new256.parameter_median.iloc[0]):.1f}'
    values['RevXrayBaselineJtkPsnrMedian']=f"{float(q50[q50.domain=='xray'].jpeg2000_psnr_parameter.median()):.1f}"
    for codec in CODECS:
        for metric in METRICS:
            d=differences[(differences.codec==codec)&(differences.metric==metric)&(differences.left=='T1')]
            scale=1000 if metric=='ssim' else 1
            key=f'Rev{name[codec]}{metric.capitalize()}Gain'
            values.update({key+'Min':f'{d.difference.min()*scale:.3f}',key+'Max':f'{d.difference.max()*scale:.3f}',key+'Positive':str(int((d.lo>0).sum()))})
        c1=pred[(pred.codec==codec)&(pred.metric=='psnr')&(pred.predictor=='C1')]['mean']
        values.update({f'Rev{name[codec]}COnePsnrMeanMin':f'{c1.min():.2f}',f'Rev{name[codec]}COnePsnrMeanMax':f'{c1.max():.2f}'})
    values.update(RevExampleImage=tex(example['image_id']),RevExampleC=f'{c:.6f}',RevExampleIntercept=f'{model.intercept:.6f}',
        RevExampleSlope=f'{model.slope:.6f}',RevExampleRatio=f'{sample.parameter:.6f}',RevExampleTarget=f'{sample.target_psnr:.4f}',
        RevExampleAchieved=f'{sample.achieved_psnr:.4f}',RevExampleError=f'{sample.error_db:.4f}',RevExampleResidual=f'{floor.error_db:.4f}')
    GEN.mkdir(parents=True,exist_ok=True)
    (GEN/'values.tex').write_text(''.join(r'\setval{'+k+'}{'+v+'}'+'\n' for k,v in values.items()))
    tables={'tableS1_body.tex':body(config[config.q==50],['domain','config','codec','metric','parameter_median','parameter_iqr_ratio','accepted','total']),
        'tableS2_body.tex':body(dispersion,['config','q','codec','metric','n_domains','subset_max_min','full_dataset_q50_max_min']),
        'tableS3_body.tex':body(sizes[sizes.q==50],['domain','variant','codec','metric','parameter_median','file_bpp_median','payload_bpp_median','header_pct_median']),
        'tableS4_body.tex':body(pred[pred.predictor.isin(['T1','C1','floor'])],['seed','codec','metric','predictor','mean','p90'],digits=5),
        'tableS5_body.tex':body(differences,['seed','codec','metric','left','right','difference','lo','hi'],digits=5),
        'tableS6_body.tex':coefficient_body(coefficients)}
    for file,text in tables.items():
        (GEN/file).write_text(text)
