"""Bounded sensitivity experiments, isolated from the published baseline outputs."""
import gzip
import hashlib
import io
import json
import os
import time
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import imagecodecs
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, features
from scipy.fft import dctn

from .models import Q50
from .search import match, STATUSES
from .headers import jpeg_header, jp2_header, jxl_header
from .codecs import BOUNDS, EncoderConfig, ProtocolFailure, encode, luma
from .config import ROOT, CROPS

# Published baseline (read only) and frozen revision inputs.
BASE = ROOT / 'results'
DATA = ROOT / 'data/revision'
OUT = Path(os.environ.get('EQUIVALENCE_REVISION_RESULTS', ROOT / 'results/revision')).resolve()
SUBSET = Path(os.environ.get('EQUIVALENCE_REVISION_SUBSET', DATA / 'subset.csv')).resolve()
SIZE_MANIFEST = Path(os.environ.get('EQUIVALENCE_REVISION_SIZE_MANIFEST', DATA / 'size_manifest.csv')).resolve()
SPLITS = os.environ.get('EQUIVALENCE_REVISION_SPLITS')
REVISION_CROPS = Path(os.environ.get('EQUIVALENCE_REVISION_CROPS', ROOT / 'data/revision_crops')).resolve()
VERSION = 'reviewer-revision-2-middle-mri'
DOMAINS = ['natural', 'xray', 'mri', 'aerial']
QUALITIES = [10, 50, 90]
CODECS = ['jpeg2000', 'jpegxl']
METRICS = ['psnr', 'ssim']
ACCEPTED = ['matched', 'matched_scan']
CONFIGS = dict(baseline=EncoderConfig(), effort3=EncoderConfig(effort=3),
               effort9=EncoderConfig(effort=9),
               pillow_transforms=EncoderConfig(irreversible=False, color_transform=False),
               jpeg444=EncoderConfig(subsampling=0))


def read(path):
    return pd.read_csv(path, dtype={'image_id': str, 'group': str}, float_precision='round_trip')


def baseline(name):
    return ROOT / 'data/manifest.csv' if name == 'manifest.csv' else BASE / name


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n', encoding='utf-8')
    for attempt in range(30):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt==29:
                raise
            time.sleep(.1)


def require(condition, message):
    if not condition:
        raise ProtocolFailure(message)


def locate(path):
    """Map a repository-relative image path onto the configured crop folders."""
    path = Path(path).as_posix()
    for prefix, folder in [('data/crops/', CROPS), ('data/revision_crops/', REVISION_CROPS)]:
        if path.startswith(prefix):
            return folder / path[len(prefix):]
    raise ProtocolFailure('Unknown image location: '+path)


def pixels(record):
    a = np.asarray(Image.open(locate(record['path'])))
    require(hashlib.sha256(a.tobytes()).hexdigest() == record['sha256'], 'Source pixels changed')
    return a


def entropy(a):
    """Same q50 statistic, using complete native 8x8 blocks (no padding)."""
    y = luma(a) - 128
    h, w = y.shape
    h, w = h // 8 * 8, w // 8 * 8
    blocks = y[:h, :w].reshape(h//8, 8, w//8, 8).transpose(0, 2, 1, 3)
    counts = np.unique(np.rint(dctn(blocks, axes=(-2, -1), norm='ortho') / Q50), return_counts=True)[1]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def select_subset(manifest, complexity, count=40):
    joined = manifest.merge(complexity[['domain', 'image_id', 'C_DCT']], on=['domain', 'image_id'], validate='one_to_one')
    selected = []
    for domain, f in joined.groupby('domain', sort=True):
        groups = f.groupby('group', sort=True).C_DCT.median().reset_index(name='group_complexity')
        groups = groups.sort_values(['group_complexity', 'group']).reset_index(drop=True)
        groups['quartile'] = pd.qcut(np.arange(len(groups)), 4, labels=False)
        rng = np.random.default_rng(0)
        for quartile, cell in groups.groupby('quartile', sort=True):
            require(len(cell) >= count//4, f'Insufficient eligible groups: {domain}/{quartile}')
            for group in cell.iloc[rng.choice(len(cell), count//4, replace=False)].itertuples():
                members=f[f.group == group.group]
                if domain == 'mri':
                    members=members.sort_values(['slice_index_canonical','image_id'],kind='mergesort')
                    r=members.iloc[(len(members)-1)//2].to_dict()
                    rule='middle_slice_by_canonical_index_lower_middle'
                else:
                    r=members.sort_values('image_id').iloc[0].to_dict()
                    rule='lexicographically_first_image_in_selected_group'
                r.update(quartile=int(quartile), group_complexity=group.group_complexity, selection_rule=rule)
                selected.append(r)
    return pd.DataFrame(selected).sort_values(['domain', 'image_id']).reset_index(drop=True)


def protocol():
    return dict(version=VERSION, configurations={k:asdict(v) for k,v in CONFIGS.items()},
        qualities=QUALITIES, groups_per_domain=40, selection_seed=0, prediction_seeds=list(range(5)),
        coefficient_bootstraps=1000, error_bootstraps=10000, bounds=BOUNDS,
        search='src.search.match: unchanged bisection and 64-point nonmonotone fallback',
        selection_rule='40 seed-0 quartile-stratified groups/domain; MRI uses lower-middle retained slice by canonical slice index; other domains use lexicographically first image; pixel hashes stored in subset manifest',
        size_complexity='Complete native 8x8 blocks; discard incomplete edge blocks, no padding',
        versions=json.loads((BASE/'encoder_versions.json').read_text()))


def protected_baseline():
    files = [p for p in [*BASE.glob('*'), *(ROOT/'data').glob('*')] if p.is_file() and p.suffix in ['.csv', '.json', '.txt']]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = protected_baseline()
    frozen = OUT / 'baseline_hashes.json'
    if frozen.exists():
        old = json.loads(frozen.read_text())
        require(all(digest(ROOT/p) == h for p, h in old.items()), 'Frozen baseline changed')
    else:
        save_json(frozen, hashes)
    manifest, complexity, pairs = [read(baseline(n)) for n in ['manifest.csv', 'complexity.csv', 'pairs.csv']]
    splits = read(baseline('model_splits.csv'))
    require(len(manifest) == 5970 and not manifest.sha256.duplicated().any(), 'Baseline image count/hash issue')
    require(not pairs.duplicated(['domain', 'image_id', 'q']).any() and len(pairs) == 9*len(manifest), 'Baseline pair coverage')
    require(splits.groupby(['domain', 'group']).split.nunique().max() == 1, 'Baseline group leakage')
    linked = complexity.merge(splits, on=['domain', 'group'], suffixes=('', '_saved'), validate='many_to_one')
    require(linked.split.eq(linked.split_saved).all(), 'Feature split mismatch')
    subset = select_subset(manifest, complexity)
    published = read(DATA/'subset.csv')
    try:
        pd.testing.assert_frame_equal(subset[published.columns], published, check_dtype=False)
    except AssertionError as exc:
        raise ProtocolFailure('Reselected subset differs from data/revision/subset.csv') from exc
    probes = []
    for r in subset.groupby('domain', sort=True).head(1).to_dict('records'):
        a = pixels(r)
        require(abs(entropy(a)-r['C_DCT']) < 1e-12, 'DCT baseline mismatch')
        pair = pairs[(pairs.domain == r['domain']) & (pairs.image_id == r['image_id']) & (pairs.q == 50)].iloc[0]
        for codec in ['jpeg', *CODECS]:
            prefix = codec+'_psnr_'
            parameter = 50 if codec == 'jpeg' else pair[prefix+'parameter']
            measured = encode(a, codec, parameter)
            expected = pair.jpeg_bytes if codec == 'jpeg' else pair[prefix+'encoded_bytes']
            require(measured['encoded_bytes'] == expected, 'Default encoder byte count changed')
            require(abs(measured['psnr'] - (pair.jpeg_psnr if codec == 'jpeg' else pair[prefix+'psnr'])) < 1e-9, 'Default fidelity changed')
            require(abs(measured['ssim'] - (pair.jpeg_ssim if codec == 'jpeg' else pair[prefix+'ssim'])) < 1e-12, 'Default SSIM changed')
            probes.append(dict(domain=r['domain'], codec=codec, bytes=int(expected)))
    eval_checks = []
    for name, error, predictors in [('model_evaluation.csv', 'error_db', ['T0','T1','C0','C1','C1_BPP','floor']),
                                     ('ssim_model_evaluation.csv', 'error_ssim', ['T1','C1'])]:
        f = read(baseline(name))
        joined = f.merge(splits, on=['domain','group'], validate='many_to_one')
        require(joined.split.eq('test').all(), 'Evaluation contains training group')
        require(not ((f.domain == 'xray') & (f.q == 10) & (f.codec == 'jpegxl')).any(), 'Excluded cell present')
        keys = ['domain','image_id','group','q','codec']
        require(not f.duplicated(keys+['predictor']).any(), 'Duplicate model evaluation')
        require(f.groupby(keys).predictor.agg(set).map(lambda x: x == set(predictors)).all(), 'Models not paired')
        eval_checks.append(dict(file=name, rows=len(f), paired=True, held_out=True))
    # Original implementation hashes are audit history; bump VERSION for any semantic protocol change.
    previous = json.loads((DATA/'protocol.json').read_text())
    require({k:v for k,v in previous.items() if k!='implementation_hashes'}==json.loads(json.dumps(protocol())),'Revision protocol changed')
    save_json(OUT/'audit.json', dict(status='passed', baseline_images=len(manifest), subset_images=len(subset),
        probes=probes, evaluation=eval_checks, confidence_intervals='Existing analysis and models: within-domain group bootstrap; paired model differences; equal domain weights',
        r2='complexity_r2.csv and ssim_complexity_r2.csv: existing held-out split',
        ranking_denominator='Intersection of accepted codec targets, separately by metric and file/payload basis'))
    print('Audit passed; frozen baseline and selected 160 independent groups', flush=True)


def checkpoint_path(stage, record, tag=''):
    return OUT/'checkpoints'/stage/record['domain']/(record['image_id']+tag+'.json.gz')


def load_checkpoint(path, signature):
    if not path.exists():
        return None
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    require(data['fingerprint'] == signature, f'Incompatible checkpoint: {path}')
    return data


def store_checkpoint(path, signature, rows, calls):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8') as f:
        json.dump(dict(fingerprint=signature, rows=rows, actual_encode_calls=calls), f)
    temporary.replace(path)
    return path


def experiment_image(task):
    record, stage = task[:2]
    saved_pairs = task[2] if len(task)>2 else []
    signature = fingerprint([VERSION, digest(DATA/'protocol.json'), record, stage])
    dest = checkpoint_path(stage, record)
    if load_checkpoint(dest, signature) is not None:
        return str(dest)
    a = pixels(record)
    cache, files, rows = {}, {}, []
    configs = ['baseline'] if stage == 'size' else list(CONFIGS)
    def evaluate(codec, parameter, config_name):
        config = CONFIGS[config_name]
        # Include the complete configuration in the key, even for irrelevant fields.
        key = codec, float(parameter), config_name
        if key not in cache:
            result, data = encode(a, codec, parameter, config, return_data=True)
            cache[key], files[key] = result, data
        return cache[key]
    for q in QUALITIES:
        for name in configs:
            if name == 'jpeg444' and a.ndim == 2:
                continue
            target_name = 'jpeg444' if name == 'jpeg444' else 'baseline'
            target = evaluate('jpeg', q, target_name)
            active = CODECS if name in ['baseline','jpeg444'] else ['jpegxl'] if name.startswith('effort') else ['jpeg2000']
            for codec in active:
                for metric in METRICS:
                    saved = next((p for p in saved_pairs if p['q']==q),None) if name=='baseline' else None
                    if saved is not None:
                        prefix=f'{codec}_{metric}_'
                        result={k[len(prefix):]:v for k,v in saved.items() if k.startswith(prefix)}
                        result.pop('new_encode_calls',None)
                        measured=evaluate(codec,result['parameter'],name)
                        require(measured['encoded_bytes']==result['encoded_bytes'] and abs(measured[metric]-result[metric])<1e-9,
                                'Frozen subset search measurement failed reproduction')
                    else:
                        result, trials = match(lambda p: evaluate(codec, p, name), codec, target[metric], metric)
                    data = files[codec, float(result['parameter']), name]
                    header = dict(jpeg2000=jp2_header, jpegxl=jxl_header)[codec](data)
                    target_header = jpeg_header(files['jpeg', float(q), target_name])
                    rows.append(dict(domain=record['domain'], image_id=record['image_id'], group=record['group'],
                        source_sha256=record['sha256'], variant=record.get('variant','crop256'), config=name,
                        width=a.shape[1], height=a.shape[0], C_DCT=entropy(a), q=q, codec=codec, metric=metric,
                        target=target[metric], jpeg_bytes=target['encoded_bytes'], jpeg_header_bytes=target_header,
                        header_bytes=header, payload_bytes=result['encoded_bytes']-header,
                        payload_bpp=8*(result['encoded_bytes']-header)/a.shape[0]/a.shape[1],
                        header_pct=100*header/result['encoded_bytes'], **result))
                    rows[-1]['measurement_origin']='frozen_search_reproduced' if saved is not None else 'fresh_search'
    return str(store_checkpoint(dest, signature, rows, len(cache)))


def parallel(tasks, function, stage, workers):
    rows, calls = [], 0
    # Native codec calls release the GIL. Threads also work in restricted Windows
    # environments that prohibit multiprocessing named-pipe creation.
    from .headers import jxl_library
    jxl_library()
    pool=ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(function, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            path = future.result()
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            rows.extend(data['rows'])
            calls += data['actual_encode_calls']
            print(f'{stage}: {i}/{len(tasks)} checkpoints; {calls} recorded actual encodes', flush=True)
            save_json(OUT/(stage+'_progress.json'), dict(completed=i, total=len(tasks), recorded_encode_calls=calls))
    except BaseException as exc:
        save_json(OUT/(stage+'_failure.json'),dict(error=repr(exc),completed_rows=len(rows),stage=stage))
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    frame = pd.DataFrame(rows)
    if 'measurement_origin' in frame:
        frame['measurement_origin']=frame.measurement_origin.fillna('fresh_search')
    frame=frame.sort_values([key for key in ['seed','domain','image_id','q','codec','metric','config','predictor'] if key in frame]).reset_index(drop=True)
    frame.to_csv(OUT/(stage+'_results.csv'), index=False)
    return frame


def frozen_pairs():
    pairs=read(baseline('pairs.csv'))
    return {(d,i):f[f.q.isin(QUALITIES)].to_dict('records') for (d,i),f in pairs.groupby(['domain','image_id'])}


def configuration(workers):
    grouped=frozen_pairs()
    return parallel([(r,'configuration',grouped[r['domain'],r['image_id']]) for r in read(SUBSET).to_dict('records')], experiment_image, 'configuration', workers)


def larger_xray(record, dataset_root, landmarks):
    """Recover original image/mask; find a fully lung-covered 512 crop containing the frozen crop."""
    archive_path, name = record['source'].split('::')
    with zipfile.ZipFile(Path(dataset_root)/archive_path.removeprefix('data/')) as archive:
        im = Image.open(io.BytesIO(archive.read(name))).convert('L')
        if 'Montgomery' in archive_path:
            masks = [np.asarray(Image.open(io.BytesIO(archive.read('MontgomerySet/ManualMask/'+side+'/'+Path(name).name)))) > 0 for side in ['leftMask','rightMask']]
            mask = Image.fromarray((masks[0]|masks[1]).astype(np.uint8)*255)
        else:
            yy, xx = np.where(np.asarray(im)>1)
            x0,y0,x1,y1 = xx.min(),yy.min(),xx.max()+1,yy.max()+1
            width,height = x1-x0,y1-y0
            side = max(width,height)
            px,py = (side-width)//2,(side-height)//2
            mask = Image.new('L',im.size)
            draw = ImageDraw.Draw(mask)
            for lung in ['LL','RL']:
                points = np.load(Path(landmarks)/lung/(record['image_id']+'.npy'), allow_pickle=False)
                points = points*side/1024-np.array([px,py])+np.array([x0,y0])
                draw.polygon([tuple(x) for x in points], fill=255)
    small = im.resize((im.width//2,im.height//2),Image.Resampling.BOX)
    m = np.asarray(mask.resize(small.size,Image.Resampling.NEAREST)) > 0
    x,y = int(record['crop_x']),int(record['crop_y'])
    require(np.array_equal(np.asarray(small)[y:y+256,x:x+256],pixels(record)), 'X-ray source conversion differs')
    integral = np.pad(m.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    sums = integral[512:,512:]-integral[:-512,512:]-integral[512:,:-512]+integral[:-512,:-512]
    candidates = np.zeros(sums.shape, dtype=bool)
    candidates[max(0,y-256):min(y+1,sums.shape[0]),max(0,x-256):min(x+1,sums.shape[1])] = True
    yy,xx = np.where((sums == 512**2) & candidates)
    fallback = not len(xx)
    if fallback:
        # Approved fallback: a fresh centered 256 crop inside an eligible
        # 512 crop from the same patient, never a relaxation of mask coverage.
        yy,xx = np.where(sums == 512**2)
        if not len(xx):
            return None
    # Closest available centre to frozen crop, then row/column order.
    index = np.argmin((xx-(x-128))**2+(yy-(y-128))**2)
    nx,ny = int(xx[index]),int(yy[index])
    return np.asarray(small)[ny:ny+512,nx:nx+512], dict(larger_crop_x=nx,larger_crop_y=ny,lung_fraction=1.,new_nested_pair=fallback)


def select_size(dataset_root, landmarks):
    """Re-derive size eligibility and selection; compare with the frozen size manifests.

    Needs the X-ray lung masks/landmarks, which are not redistributed. Results are
    written to OUT, never over data/revision.
    """
    from .crops import source_bytes
    manifest, complexity = read(baseline('manifest.csv')), read(baseline('complexity.csv'))
    eligible, excluded = [], []
    for record in manifest[manifest.domain.isin(['natural','xray'])].to_dict('records'):
        if record['domain'] == 'natural':
            a = np.asarray(Image.open(io.BytesIO(source_bytes(record['source'],dataset_root))).convert('RGB'))
            x,y = int(record['crop_x']),int(record['crop_y'])
            require(np.array_equal(a[y:y+256,x:x+256],pixels(record)), 'Natural source pixels differ')
            meta = dict(larger_crop_x=0,larger_crop_y=0)
            path = f"data/revision_crops/natural/{record['image_id']}.png"
        else:
            result = larger_xray(record, dataset_root, landmarks)
            if result is None:
                excluded.append(dict(domain='xray',image_id=record['image_id'],group=record['group'],reason='No fully lung-covered 512 crop even with approved new nested pair'))
                continue
            a,meta = result
            if meta['new_nested_pair']:
                small = a[128:384,128:384]
                record.update(original_baseline_sha256=record['sha256'],original_baseline_path=record['path'],
                              path=f"data/revision_crops/xray256/{record['image_id']}.png",sha256=hashlib.sha256(small.tobytes()).hexdigest(),
                              baseline_crop_x=record['crop_x'],baseline_crop_y=record['crop_y'],
                              crop_x=meta['larger_crop_x']+128,crop_y=meta['larger_crop_y']+128)
            path = f"data/revision_crops/xray/{record['image_id']}.png"
        eligible.append(dict(record, larger_path=path, larger_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                             larger_width=a.shape[1],larger_height=a.shape[0],**meta))
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(excluded,columns=['domain','image_id','group','reason']).to_csv(OUT/'size_exclusions.csv',index=False)
    pool = pd.DataFrame(eligible)
    pool.to_csv(OUT/'size_eligible.csv',index=False)
    require(set(pool.domain)=={'natural','xray'},'No eligible larger images for a required domain')
    selected = select_subset(pool,complexity)
    records = []
    for r in selected.to_dict('records'):
        records.append(dict(r,variant='crop256_new' if r['domain']=='xray' and r.get('new_nested_pair',False) else 'crop256'))
        records.append(dict(r,image_id=r['image_id']+'_large', original_image_id=r['image_id'],variant='larger',
                            path=r['larger_path'],sha256=r['larger_sha256'],width=r['larger_width'],height=r['larger_height']))
    result = pd.DataFrame(records)
    result.to_csv(OUT/'size_manifest.csv',index=False)
    frozen = read(DATA/'size_manifest.csv')
    keys = ['domain','image_id','variant','sha256','crop_x','crop_y','larger_crop_x','larger_crop_y']
    require(read(OUT/'size_manifest.csv')[keys].equals(frozen[keys]),'Re-derived size manifest differs from data/revision/size_manifest.csv')
    require(read(OUT/'size_exclusions.csv').equals(read(DATA/'size_exclusions.csv')),'Re-derived size exclusions differ')
    print(f'Size selection reproduced: {len(pool)} eligible, {len(excluded)} excluded, {len(result)} manifest rows',flush=True)
    return result


def rebuild_size(dataset_root):
    """Rebuild size-study images from frozen coordinates and verify every pixel hash."""
    from .crops import source_bytes
    frame = read(SIZE_MANIFEST)
    for i,record in enumerate(frame.to_dict('records')):
        if record['path'].startswith('data/crops/'):
            pixels(record)
            continue
        im = Image.open(io.BytesIO(source_bytes(record['source'],dataset_root))).convert('RGB' if record['domain']=='natural' else 'L')
        factor = int(record['integer_factor'])
        if factor > 1:
            im = im.resize((im.width//factor,im.height//factor),Image.Resampling.BOX)
        a = np.asarray(im)
        if record['domain'] == 'xray':
            x,y = int(record['larger_crop_x']),int(record['larger_crop_y'])
            a = a[y:y+512,x:x+512]
            if record['variant'] == 'crop256_new':
                a = a[128:384,128:384]
        a = np.ascontiguousarray(a)
        require(hashlib.sha256(a.tobytes()).hexdigest()==record['sha256'],'Rebuilt size image hash mismatch: '+record['image_id'])
        path = locate(record['path'])
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(a).save(path)
    print(f'All {len(frame)} size-study images match manifest SHA-256.',flush=True)


def size(workers):
    manifest = read(SIZE_MANIFEST)
    grouped=frozen_pairs()
    return parallel([(r,'size',grouped.get((r['domain'],r['image_id']),[]) if r['domain']=='natural' and r['variant']=='crop256' else []) for r in manifest.to_dict('records')],experiment_image,'size',workers)


def eligible(f,codec,metric):
    f = f[f[f'{codec}_{metric}_status'].isin(ACCEPTED) & (f.C_DCT > 0)]
    if codec == 'jpegxl':
        f = f[~((f.domain == 'xray') & (f.q == 10))]
    return f.copy()


def fit(f,codec,metric,predictor):
    y = np.log(f[f'{codec}_{metric}_parameter'].to_numpy())
    if predictor.startswith('T'):
        return float(np.median(y)),0.
    x = np.log(f.C_DCT.to_numpy())
    a,b = np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
    return float(a),float(b)


def prediction_splits(subset):
    if SPLITS:
        return read(SPLITS)
    assignments = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        for domain in DOMAINS:
            groups = sorted(subset[subset.domain == domain].group.unique())
            train = set(rng.permutation(groups)[:int(.8*len(groups))])
            assignments.extend(dict(seed=seed,domain=domain,group=g,split='train' if g in train else 'test') for g in groups)
    splits = pd.DataFrame(assignments)
    frozen = read(DATA/'prediction_splits.csv')
    if set(zip(splits.domain,splits.group))==set(zip(frozen.domain,frozen.group)):
        require(splits.equals(frozen),'Group-separated split assignments changed')
    return splits


def fit_prediction():
    subset = read(SUBSET)
    pairs = read(baseline('pairs.csv'))
    data = pairs[pairs.q.isin(QUALITIES)].merge(subset[['domain','image_id','C_DCT']],on=['domain','image_id'],validate='many_to_one')
    all_splits = prediction_splits(subset)
    models = []
    rng_boot = np.random.default_rng(0)
    for seed in range(5):
        splits = all_splits[all_splits.seed == seed].drop(columns='seed')
        d = data.merge(splits,on=['domain','group'],validate='many_to_one')
        for codec in CODECS:
            for metric in METRICS:
                e = eligible(d,codec,metric)
                for predictor in ['T0','T1','C0','C1']:
                    grouping = ['q'] if predictor in ['T0','C0'] else ['domain','q']
                    for key,f in e[e.split == 'train'].groupby(grouping):
                        key = key if isinstance(key,tuple) else (key,)
                        domain,q = ('all',key[0]) if len(key)==1 else key
                        a,b = fit(f,codec,metric,predictor)
                        row = dict(seed=seed,domain=domain,q=int(q),codec=codec,metric=metric,predictor=predictor,
                                   intercept=a,slope=b,train_n=len(f))
                        if predictor in ['C0','C1']:
                            # One image/group in this subset: rows are independent groups.
                            boot = np.array([fit(f.iloc[rng_boot.integers(0,len(f),len(f))],codec,metric,predictor) for _ in range(1000)])
                            row.update(intercept_lo=float(np.quantile(boot[:,0],.025)),intercept_hi=float(np.quantile(boot[:,0],.975)),
                                       slope_lo=float(np.quantile(boot[:,1],.025)),slope_hi=float(np.quantile(boot[:,1],.975)))
                        models.append(row)
    all_splits.to_csv(OUT/'prediction_splits.csv',index=False)
    pd.DataFrame(models).to_csv(OUT/'prediction_coefficients.csv',index=False)
    return data


def prediction_image(task):
    record,seed,pairs,models = task
    signature = fingerprint([VERSION,digest(DATA/'protocol.json'),record,seed,pairs,models])
    dest = checkpoint_path('prediction',record,f'_seed{seed}')
    if load_checkpoint(dest,signature) is not None:
        return str(dest)
    a = pixels(record)
    cache,rows = {},[]
    for pair in pairs:
        for codec in CODECS:
            if codec == 'jpegxl' and record['domain']=='xray' and pair['q']==10:
                continue
            for metric in METRICS:
                prefix = f'{codec}_{metric}_'
                if pair[prefix+'status'] not in ACCEPTED or record['C_DCT']<=0:
                    continue
                target = pair['jpeg_'+metric]
                base = dict(seed=seed,domain=record['domain'],image_id=record['image_id'],group=record['group'],source_sha256=record['sha256'],q=pair['q'],codec=codec,metric=metric,target=target)
                rows.append(dict(base,predictor='floor',parameter=pair[prefix+'parameter'],achieved=pair[prefix+metric],error=pair[prefix+'residual'],measurement_origin='frozen_full_dataset_search'))
                for predictor in ['T0','T1','C0','C1']:
                    domain = 'all' if predictor in ['T0','C0'] else record['domain']
                    model = next(m for m in models if m['domain']==domain and m['q']==pair['q'] and m['codec']==codec and m['metric']==metric and m['predictor']==predictor)
                    lp = model['intercept']+model['slope']*np.log(record['C_DCT'])
                    parameter = float(np.exp(np.clip(lp,*np.log(BOUNDS[codec]))))
                    key = codec,parameter
                    if key not in cache:
                        cache[key] = encode(a,codec,parameter)
                    achieved = cache[key][metric]
                    rows.append(dict(base,predictor=predictor,parameter=parameter,achieved=achieved,error=abs(achieved-target),
                                     C_DCT=record['C_DCT'],intercept=model['intercept'],slope=model['slope'],clipped=bool(lp<np.log(BOUNDS[codec][0]) or lp>np.log(BOUNDS[codec][1])),measurement_origin='fresh_encode'))
    return str(store_checkpoint(dest,signature,rows,len(cache)))


def prediction(workers):
    subset = read(SUBSET)
    fit_prediction()
    splits,models,pairs = read(OUT/'prediction_splits.csv'),read(OUT/'prediction_coefficients.csv'),read(baseline('pairs.csv'))
    tasks=[]
    for s in splits[splits.split=='test'].to_dict('records'):
        r = subset[(subset.domain==s['domain']) & (subset.group==s['group'])].iloc[0].to_dict()
        p = pairs[(pairs.domain==r['domain']) & (pairs.image_id==r['image_id']) & pairs.q.isin(QUALITIES)]
        tasks.append((r,s['seed'],p.to_dict('records'),models[models.seed==s['seed']].to_dict('records')))
    fresh=parallel(tasks,prediction_image,'prediction',workers)
    fresh.groupby(['domain','measurement_origin']).size().rename('rows').reset_index().to_csv(OUT/'prediction_measurement_origins.csv',index=False)
    return fresh


def summarize_experiment(f):
    rows=[]
    keys=['domain','config','variant','q','codec','metric']
    for key,cell in f.groupby(keys):
        a=cell[cell.status.isin(ACCEPTED)]
        row=dict(zip(keys,key),total=len(cell),accepted=len(a),unreachable=int(cell.status.str.startswith('unreachable').sum()),unmatched=int(cell.status.eq('unmatched').sum()))
        if len(a):
            row.update(parameter_median=a.parameter.median(),parameter_iqr_ratio=a.parameter.quantile(.75)/a.parameter.quantile(.25),
                residual_median=a.residual.median(),residual_p90=a.residual.quantile(.9),file_bpp_median=a.bpp.median(),payload_bpp_median=a.payload_bpp.median(),
                file_saving_median=(100*(1-a.encoded_bytes/a.jpeg_bytes)).median(),
                payload_saving_median=(100*(1-a.payload_bytes/(a.jpeg_bytes-a.jpeg_header_bytes))).median(),header_pct_median=a.header_pct.median())
        rows.append(row)
    return pd.DataFrame(rows)


def check_asymmetry():
    """Create measured per-quality dispersion and codec-asymmetry gate tables."""
    config=summarize_experiment(read(OUT/'configuration_results.csv'))
    config.to_csv(OUT/'configuration_summary.csv',index=False)
    comparisons=[
        ('selected','baseline','baseline'),
        ('selected J2K / JXL effort 3','baseline','effort3'),
        ('selected J2K / JXL effort 9','baseline','effort9'),
        ('J2K default transforms / selected JXL','pillow_transforms','baseline'),
        ('JPEG 4:4:4 reference','jpeg444','jpeg444')]
    rows=[]
    for q in QUALITIES:
        for metric in METRICS:
            for label,j2k_config,jxl_config in comparisons:
                a=config[(config.q==q)&(config.metric==metric)&(config.codec=='jpeg2000')&(config.config==j2k_config)&(config.accepted>0)]
                b=config[(config.q==q)&(config.metric==metric)&(config.codec=='jpegxl')&(config.config==jxl_config)&(config.accepted>0)]
                common=sorted(set(a.domain)&set(b.domain))
                am=a[a.domain.isin(common)].set_index('domain').parameter_median
                bm=b[b.domain.isin(common)].set_index('domain').parameter_median
                ar=float(am.max()/am.min()) if len(am) else np.nan
                br=float(bm.max()/bm.min()) if len(bm) else np.nan
                rows.append(dict(comparison=label,q=q,metric=metric,jpeg2000_config=j2k_config,jpegxl_config=jxl_config,
                    n_domains=len(common),jpeg2000_max_min=ar,jpegxl_max_min=br,
                    jpeg2000_more_domain_dispersion=bool(ar>br) if len(common) else False))
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'configuration_asymmetry_by_q.csv',index=False)
    selected=table[(table.comparison=='selected')&(table.q==50)&(table.metric=='psnr')].iloc[0]
    full=read(baseline('pairs.csv'))
    full_q50=full[full.q==50]
    medians={}
    for codec in CODECS:
        vals=full_q50.groupby('domain')[f'{codec}_psnr_parameter'].median()
        medians[codec]=float(vals.max()/vals.min())
    decision=dict(protocol_version=VERSION,subset_q50_psnr_jpeg2000_max_min=float(selected.jpeg2000_max_min),
        subset_q50_psnr_jpegxl_max_min=float(selected.jpegxl_max_min),full_dataset_q50_psnr=medians,
        asymmetry_holds=bool(selected.jpeg2000_more_domain_dispersion),q50_gate='pass' if selected.jpeg2000_more_domain_dispersion else 'stop')
    save_json(OUT/'asymmetry_gate.json',decision)
    return decision


def coefficient_summary(coefficients):
    """Median and range of C1 coefficients across splits, with the envelope of their bootstrap intervals."""
    g=coefficients[coefficients.predictor=='C1'].groupby(['domain','codec','metric','q'])
    summary=pd.DataFrame(dict(intercept_median=g.intercept.median(),intercept_min=g.intercept.min(),intercept_max=g.intercept.max(),
        intercept_bootstrap_low=g.intercept_lo.min(),intercept_bootstrap_high=g.intercept_hi.max(),
        slope_median=g.slope.median(),slope_min=g.slope.min(),slope_max=g.slope.max(),
        slope_bootstrap_low=g.slope_lo.min(),slope_bootstrap_high=g.slope_hi.max(),n_splits=g.size())).reset_index()
    summary['metric']=summary.metric.str.upper()
    return summary


def prediction_statistics(f):
    keys=['seed','domain','codec','metric','predictor']
    summary=f.groupby(keys).error.agg(n='size',mean='mean',p90=lambda x:x.quantile(.9)).reset_index()
    summary.to_csv(OUT/'prediction_summary.csv',index=False)
    summary.groupby(['seed','codec','metric','predictor'])[['mean','p90']].mean().reset_index().to_csv(OUT/'prediction_equal_domain.csv',index=False)
    differences=[]
    rng=np.random.default_rng(0)
    for (seed,codec,metric),cell in f.groupby(['seed','codec','metric']):
        wide=cell.pivot(index=['domain','group','image_id','q'],columns='predictor',values='error').reset_index()
        require(not wide[['T0','T1','C0','C1','floor']].isna().any().any(),'Unpaired prediction records')
        for left,right in [('T1','C1'),('C1','floor')]:
            values=[]
            points=[]
            for domain,d in wide.groupby('domain'):
                groups=d.assign(delta=d[left]-d[right]).groupby('group').delta.agg(['sum','count'])
                indices=rng.integers(0,len(groups),(10000,len(groups)))
                values.append(groups['sum'].to_numpy()[indices].sum(axis=1)/groups['count'].to_numpy()[indices].sum(axis=1))
                points.append(groups['sum'].sum()/groups['count'].sum())
            boot=np.mean(values,axis=0)
            differences.append(dict(seed=seed,codec=codec,metric=metric,left=left,right=right,difference=float(np.mean(points)),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975))))
    pd.DataFrame(differences).to_csv(OUT/'prediction_differences.csv',index=False)
    coefficients=read(OUT/'prediction_coefficients.csv')
    stability=coefficients.groupby(['domain','q','codec','metric','predictor'])[['intercept','slope']].agg(['mean','std','min','max'])
    stability.columns=['_'.join(column) for column in stability.columns]
    stability.reset_index().to_csv(OUT/'coefficient_stability.csv',index=False)
    coefficient_summary(coefficients).to_csv(OUT/'C1_coefficient_summary.csv',index=False)


def reporting():
    """Summaries, paired tables, supplement table bodies and figures; no encoding or image access."""
    for stage in ['configuration','size']:
        summarize_experiment(read(OUT/(stage+'_results.csv'))).to_csv(OUT/(stage+'_summary.csv'),index=False)
    config=read(OUT/'configuration_results.csv')
    keys=['domain','image_id','q','codec','metric']
    comparison=config[config.config!='baseline'].merge(config[config.config=='baseline'],on=keys,suffixes=('','_baseline'),validate='many_to_one')
    comparison['both_accepted']=comparison.status.isin(ACCEPTED)&comparison.status_baseline.isin(ACCEPTED)
    comparison['parameter_ratio']=comparison.parameter/comparison.parameter_baseline
    comparison['file_rate_ratio']=comparison.encoded_bytes/comparison.encoded_bytes_baseline
    comparison['target_difference']=comparison.target-comparison.target_baseline
    comparison.to_csv(OUT/'configuration_paired.csv',index=False)
    size_data=read(OUT/'size_results.csv')
    paired=size_data[size_data.variant.str.startswith('crop256')].merge(size_data[size_data.variant=='larger'],on=['domain','group','q','codec','metric'],suffixes=('_small','_large'),validate='one_to_one')
    paired['both_accepted']=paired.status_small.isin(ACCEPTED)&paired.status_large.isin(ACCEPTED)
    paired['parameter_ratio']=paired.parameter_large/paired.parameter_small
    paired['complexity_ratio']=paired.C_DCT_large/paired.C_DCT_small
    paired['file_bpp_ratio']=paired.bpp_large/paired.bpp_small
    paired['payload_bpp_ratio']=paired.payload_bpp_large/paired.payload_bpp_small
    paired.to_csv(OUT/'size_paired.csv',index=False)
    paired[paired.both_accepted].groupby(['domain','q','codec','metric'])[['parameter_ratio','complexity_ratio','file_bpp_ratio','payload_bpp_ratio']].median().reset_index().to_csv(OUT/'size_paired_summary.csv',index=False)
    prediction_statistics(read(OUT/'prediction_results.csv'))
    check_asymmetry()
    from .revision_report import render
    render()


def validation():
    hashes=json.loads((OUT/'baseline_hashes.json').read_text())
    require(all(digest(ROOT/p)==h for p,h in hashes.items()),'Baseline output changed')
    subset=read(SUBSET)
    require(subset.groupby('domain').size().nunique()==1 and subset.domain.nunique()==4,'Unbalanced subset')
    require(not subset.duplicated(['domain','group']).any(),'Subset group duplicate')
    require(subset[subset.domain=='mri'].selection_rule.eq('middle_slice_by_canonical_index_lower_middle').all(),'MRI middle-slice rule missing from manifest')
    require(subset.sha256.notna().all(),'Subset pixel hashes missing')
    config=read(OUT/'configuration_results.csv')
    size_data=read(OUT/'size_results.csv')
    keys=['domain','image_id','q','codec','metric','config']
    require(not config.duplicated(keys).any(),'Configuration duplicates')
    color=int((subset.channels==3).sum())
    require(len(config)==len(subset)*3*(4+4+2)+color*3*4,'Configuration coverage mismatch')
    require(config.measurement_origin.notna().all(),'Configuration measurement provenance missing')
    cfg_sources=subset[['domain','image_id','sha256']].merge(config[['domain','image_id','source_sha256']].drop_duplicates(),on=['domain','image_id'],validate='one_to_one')
    require(cfg_sources.sha256.eq(cfg_sources.source_sha256).all(),'Configuration source hash differs from current manifest')
    large=read(SIZE_MANIFEST)
    require(len(size_data)==len(large)*3*4,'Size coverage mismatch')
    for data in [config,size_data]:
        require(data.status.isin(STATUSES).all(),'Unknown search status')
        require((data.payload_bytes>0).all() and (data.header_bytes>0).all(),'Invalid payload/header')
    # A fresh prediction run records its splits; published results use the frozen assignments.
    split_file=OUT/'prediction_splits.csv' if (OUT/'prediction_splits.csv').exists() else DATA/'prediction_splits.csv'
    splits=read(split_file)
    require(splits.groupby(['seed','domain','group']).split.nunique().max()==1,'Prediction group leakage')
    if not SPLITS:
        per_domain=subset.groupby('domain').size().iloc[0]
        require(splits[splits.split=='test'].groupby(['seed','domain']).size().eq(per_domain-int(.8*per_domain)).all(),'Prediction split size')
    f=read(OUT/'prediction_results.csv')
    require(f.measurement_origin.notna().all() and f.source_sha256.notna().all(),'Prediction provenance missing')
    pred_sources=subset[['domain','image_id','sha256']].merge(f[['domain','image_id','source_sha256']].drop_duplicates(),on=['domain','image_id'],validate='one_to_one')
    require(pred_sources.sha256.eq(pred_sources.source_sha256).all(),'Prediction source hash differs from current manifest')
    require(not ((f.domain=='xray') & (f.q==10) & (f.codec=='jpegxl')).any(),'Excluded cell evaluated')
    require(f.groupby(['seed','domain','image_id','q','codec','metric']).predictor.agg(set).map(lambda s:s=={'T0','T1','C0','C1','floor'}).all(),'Prediction pairing')
    require(f.merge(splits,on=['seed','domain','group'],validate='many_to_one').split.eq('test').all(),'Training evaluation leakage')
    independent_baseline_checks()
    gate=check_asymmetry()
    require(gate['q50_gate']=='pass','Subset contradicts the q=50 between-domain asymmetry')
    for domain in ['natural','xray']:
        subset_size=large[large.domain==domain]
        require(len(subset_size)==2*subset_size.group.nunique(),'Size group coverage')
        for group,cell in subset_size.groupby('group'):
            small=cell[cell.variant.str.startswith('crop256')].iloc[0].to_dict()
            big=cell[cell.variant=='larger'].iloc[0].to_dict()
            a,b=pixels(small),pixels(big)
            x=int(small['crop_x'])-int(big['larger_crop_x'])
            y=int(small['crop_y'])-int(big['larger_crop_y'])
            require(np.array_equal(a,b[y:y+256,x:x+256]),'Size pair is not pixel-nested')
    checks=dict(status='passed',baseline_files_unchanged=len(hashes),configuration_rows=len(config),size_rows=len(size_data),prediction_rows=len(f),
        input_hashes={p.name:digest(p) for p in [SUBSET,DATA/'protocol.json',SIZE_MANIFEST,split_file,OUT/'prediction_coefficients.csv']})
    save_json(OUT/'validation.json',checks)
    print('Revision data validation passed',flush=True)


def codec_details():
    from .headers import jxl_library
    return dict(
        runtime_versions=dict(Pillow=Image.__version__,Libjpeg=features.version('jpg'),LibjpegTurbo=features.version('libjpeg_turbo'),
            Openjpeg=features.version('jpg_2000'),Libjxl=imagecodecs.jpegxl_version(),Imagecodecs=imagecodecs.__version__,
            header_decoder_version=jxl_library().JxlDecoderVersion()),
        input=dict(dtype='uint8',channels='L or RGB; replicated grayscale channels collapsed',range=[0,255]),
        jpeg=dict(container='JPEG file',quality=QUALITIES,progressive=False,optimize=False,subsampling='2 (4:2:0), alternative 0 (4:4:4); grayscale has no chroma',color_conversion='Pillow/libjpeg-turbo version-specific RGB to YCbCr'),
        jpeg2000=dict(container='JP2',quality_mode='rates',quality_layers='one ratio per search',tile_size='full image',
            irreversible='true; transform-default alternative false',mct='1 for RGB, 0 for grayscale; alternative always 0',
            progression='Pillow default LRCP',cinema_mode='Pillow default no',signed=False,plt=False,
            num_resolutions='Pillow forwards 0: OpenJPEG default',codeblock_size='Pillow None: OpenJPEG default',precinct_size='Pillow None: OpenJPEG default'),
        jpegxl=dict(distance='positive lossy control; bounds 0.05 to 25',effort=[3,7,9],numthreads=1,
            other_parameters='imagecodecs 2026.3.6 defaults: level, lossless, decodingspeed, photometric, bitspersample, planar, usecontainer are None'),
        metrics=dict(psnr='all-channel MSE, peak=255',ssim='Pillow YCbCr luma or L; skimage structural_similarity default window 7, data_range=255'),
        scope='Declared explicit settings plus version-pinned wrapper defaults; no equal-runtime claim')


def independent_baseline_checks():
    """Recompute held-out R2 and paired codec ranking populations from raw tables."""
    original_pairs=read(baseline('pairs.csv'))
    original_features=read(baseline('complexity.csv'))
    joined=original_pairs.merge(original_features.drop(columns='group'),on=['domain','image_id'],validate='many_to_one')
    fit_checks=[]
    for metric,name in [('psnr','model_parameters.csv'),('ssim','ssim_model_parameters.csv')]:
        for model in read(baseline(name)).to_dict('records'):
            f=joined[(joined.split=='train')&(joined.q==model['q'])&joined[f"{model['codec']}_{metric}_status"].isin(ACCEPTED)]
            if model['codec']=='jpegxl':
                f=f[~((f.domain=='xray')&(f.q==10))]
            if model['domain']!='all':
                f=f[f.domain==model['domain']]
            feature=model.get('feature','C_DCT')
            if model['predictor'].startswith('C'):
                f=f[f[feature]>0]
            y=np.log(f[f"{model['codec']}_{metric}_parameter"].to_numpy())
            if model['predictor'].startswith('T'):
                a,b=float(np.median(y)),0.
            else:
                x=np.log(f[feature].to_numpy())
                a,b=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
            require(len(f)==model['train_n'] and abs(a-model['intercept'])<1e-10 and abs(b-model['slope'])<1e-10,
                    'Baseline coefficients fail train-only reconstruction: '+str(model))
            fit_checks.append(dict(metric=metric,domain=model['domain'],q=model['q'],codec=model['codec'],predictor=model['predictor'],train_n=len(f),intercept_difference=a-model['intercept'],slope_difference=b-model['slope']))
    pd.DataFrame(fit_checks).to_csv(OUT/'baseline_train_only_audit.csv',index=False)
    r2_rows=[]
    for metric,name in [('psnr','model_evaluation.csv'),('ssim','ssim_model_evaluation.csv')]:
        evaluation=read(baseline(name))
        saved=read(baseline('complexity_r2.csv' if metric=='psnr' else 'ssim_complexity_r2.csv'))
        for (domain,codec),cell in evaluation[(evaluation.q==50)&(evaluation.predictor=='C1')].groupby(['domain','codec']):
            y=np.log(cell.exact_parameter.to_numpy())
            value=1-np.sum((y-cell.predicted_log_parameter.to_numpy())**2)/np.sum((y-y.mean())**2)
            old=saved[(saved.domain==domain)&(saved.codec==codec)].iloc[0]
            require(len(cell)==old.n and abs(value-old.r2)<1e-12,'Held-out R2 differs from baseline report')
            r2_rows.append(dict(domain=domain,codec=codec,metric=metric,n=len(cell),r2=value))
    pd.DataFrame(r2_rows).to_csv(OUT/'baseline_r2_audit.csv',index=False)
    payload=read(baseline('payload_measurements.csv'))
    pairs=read(baseline('pairs.csv'))
    expected=read(baseline('payload_ranking.csv'))
    counts=[]
    for metric in METRICS:
        a=payload[(payload.q==50)&(payload.codec=='jpeg2000')&(payload.metric==metric)]
        b=payload[(payload.q==50)&(payload.codec=='jpegxl')&(payload.metric==metric)]
        both=a.merge(b,on=['domain','image_id','q'],suffixes=('_jtk','_jxl'),validate='one_to_one')
        valid=pairs[(pairs.q==50)&pairs[f'jpeg2000_{metric}_status'].isin(ACCEPTED)&pairs[f'jpegxl_{metric}_status'].isin(ACCEPTED)]
        require(set(zip(both.domain,both.image_id))==set(zip(valid.domain,valid.image_id)),'Ranking population differs from accepted intersection')
        for domain in [*DOMAINS,'overall']:
            cell=both if domain=='overall' else both[both.domain==domain]
            for basis,column in [('file','encoded_bytes'),('payload','payload_bytes')]:
                old=expected[(expected.domain==domain)&(expected.metric==metric)&(expected.basis==basis)].iloc[0]
                wins=int((cell[column+'_jxl']<cell[column+'_jtk']).sum())
                require(len(cell)==old.n and wins==old.wins,'Codec ranking count differs from report')
                counts.append(dict(domain=domain,metric=metric,basis=basis,n=len(cell),wins=wins))
    pd.DataFrame(counts).to_csv(OUT/'baseline_ranking_audit.csv',index=False)
    require(json.loads(json.dumps(codec_details()))==json.loads((DATA/'codec_details.json').read_text()),'Codec settings or runtime versions differ from data/revision/codec_details.json')
    import platform
    from importlib.metadata import version
    packages={name:version(name) for name in ['numpy','pandas','scipy','scikit-image','matplotlib','Pillow','imagecodecs','pillow-jxl-plugin','pytest']}
    save_json(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages=packages))


def run_stage(stage,workers=4):
    require(workers>=1,'workers must be positive')
    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'status.json',dict(stage=stage,status='running',pid=os.getpid()))
    started=time.time()
    try:
        if stage=='audit': audit()
        elif stage=='configuration': configuration(workers)
        elif stage=='size': size(workers)
        elif stage=='prediction': prediction(workers)
        elif stage=='reporting': reporting()
        elif stage=='validation': validation()
        else: raise ValueError(stage)
    except Exception as exc:
        save_json(OUT/'status.json',dict(stage=stage,status='failed',error=str(exc),seconds=time.time()-started))
        raise
    save_json(OUT/'status.json',dict(stage=stage,status='complete',seconds=time.time()-started))
