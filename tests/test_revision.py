"""Guard the revision's experimental isolation and statistical contracts."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import imagecodecs
import numpy as np
import pandas as pd
from PIL import Image

from src import revision as r
from src.models import complexity
from src.codecs import EncoderConfig, ProtocolFailure, encode


def published_encoder_bytes(a, codec, parameter):
    """The encoder calls of the published protocol, written out literally."""
    if codec == 'jpegxl':
        return imagecodecs.jpegxl_encode(a, distance=float(parameter), effort=7, numthreads=1)
    f = io.BytesIO()
    if codec == 'jpeg':
        Image.fromarray(a).save(f, format='JPEG', quality=int(parameter), progressive=False, subsampling=2, optimize=False)
    else:
        Image.fromarray(a).save(f, format='JPEG2000', quality_mode='rates', quality_layers=[float(parameter)],
                                irreversible=True, mct=1 if a.ndim == 3 else 0, tile_size=(a.shape[1], a.shape[0]))
    return f.getvalue()


class RevisionTests(unittest.TestCase):
    def test_default_config_is_byte_identical_to_published_encoder(self):
        rng=np.random.default_rng(7)
        for shape in [(48,40),(48,40,3)]:
            a=rng.integers(0,256,shape,dtype=np.uint8)
            for codec,p in [('jpeg',10),('jpeg',90),('jpeg2000',3.),('jpeg2000',40.),('jpegxl',.5),('jpegxl',4.)]:
                result,data=encode(a,codec,p,return_data=True)
                self.assertEqual(data,published_encoder_bytes(a,codec,p))
                self.assertEqual(result,{**encode(a,codec,p),'seconds':result['seconds']})

    def test_default_encoder_reproduces_explicit_configuration(self):
        a=np.random.default_rng(4).integers(0,256,(32,32,3),dtype=np.uint8)
        for codec,p in [('jpeg',50),('jpeg2000',10),('jpegxl',2)]:
            left,data=encode(a,codec,p,return_data=True)
            right,other=encode(a,codec,p,EncoderConfig(),return_data=True)
            self.assertEqual(data,other)
            self.assertEqual(left['psnr'],right['psnr'])

    def test_material_configuration_changes_encoding(self):
        a=np.random.default_rng(4).integers(0,256,(64,64,3),dtype=np.uint8)
        _,baseline=encode(a,'jpeg',50,return_data=True)
        _,other=encode(a,'jpeg',50,EncoderConfig(subsampling=0),return_data=True)
        self.assertNotEqual(baseline,other)

    def test_entropy_native_edges_and_baseline(self):
        a=np.random.default_rng(4).integers(0,256,(256,256,3),dtype=np.uint8)
        self.assertEqual(r.entropy(a),complexity(a))
        b=np.pad(a,((0,3),(0,5),(0,0)),constant_values=0)
        self.assertEqual(r.entropy(a),r.entropy(b))

    def test_stratified_subset_independent_and_deterministic(self):
        manifest=pd.DataFrame([dict(domain='mri',group=f'g{i:03}',image_id=f'i{i:03}_{j}',path='x',slice_index_canonical=j) for i in range(80) for j in range(2)])
        features=manifest[['domain','image_id']].assign(C_DCT=np.repeat(np.arange(80),2)+1.)
        a=r.select_subset(manifest,features)
        b=r.select_subset(manifest.sample(frac=1,random_state=9),features)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(len(a)==40 and a.group.nunique()==40)
        self.assertEqual(a.groupby('quartile').size().tolist(),[10]*4)
        self.assertTrue(a.image_id.str.endswith('_0').all())  # lower-middle of two retained slices

    def test_checkpoint_rejects_configuration_and_seed_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.gz'
            baseline=r.fingerprint([r.VERSION,EncoderConfig().__dict__,0,'source'])
            r.store_checkpoint(path,baseline,[{'error':1}],3)
            self.assertEqual(r.load_checkpoint(path,baseline)['actual_encode_calls'],3)
            for variant in [r.fingerprint([r.VERSION,EncoderConfig(effort=3).__dict__,0,'source']),r.fingerprint([r.VERSION,EncoderConfig().__dict__,1,'source'])]:
                with self.assertRaisesRegex(ProtocolFailure,'Incompatible checkpoint'):
                    r.load_checkpoint(path,variant)

    def test_codec_metric_specific_eligibility_and_exclusion(self):
        f=pd.DataFrame(dict(domain=['xray','natural','natural'],q=[10,50,50],C_DCT=[1,2,0],
            jpegxl_psnr_status=['matched','unreachable_low','matched'],jpeg2000_psnr_status=['matched']*3))
        self.assertTrue(r.eligible(f,'jpegxl','psnr').empty)
        self.assertEqual(len(r.eligible(f,'jpeg2000','psnr')),2)

    def test_summary_denominators_include_unreachable(self):
        base=dict(domain='natural',config='baseline',variant='crop256',q=50,codec='jpegxl',metric='psnr',
            parameter=2.,residual=.01,bpp=1.,payload_bpp=.9,encoded_bytes=100,jpeg_bytes=200,payload_bytes=90,jpeg_header_bytes=20,header_pct=10.)
        f=pd.DataFrame([dict(base,status='matched'),dict(base,status='unreachable_low')])
        summary=r.summarize_experiment(f).iloc[0]
        self.assertTrue(summary.total==2 and summary.accepted==1 and summary.unreachable==1)
        self.assertEqual(summary.file_saving_median,50)

    def test_coefficient_summary_envelopes_split_intervals(self):
        f=pd.DataFrame(dict(seed=[0,1,2],domain='mri',q=50,codec='jpegxl',metric='psnr',predictor='C1',
            intercept=[1.,3.,2.],slope=[-1.,-.5,-.7],intercept_lo=[.5,2.,1.],intercept_hi=[1.5,4.,2.5],slope_lo=[-2.,-1.,-.9],slope_hi=[0.,.1,-.1]))
        s=r.coefficient_summary(pd.concat([f,f.assign(predictor='T1')])).iloc[0]
        self.assertEqual((s.metric,s.intercept_median,s.intercept_min,s.intercept_max,s.n_splits),('PSNR',2.,1.,3.,3))
        self.assertEqual((s.intercept_bootstrap_low,s.intercept_bootstrap_high,s.slope_bootstrap_low,s.slope_bootstrap_high),(.5,4.,-2.,.1))

    def test_image_paths_are_repository_relative(self):
        self.assertEqual(r.locate('data/crops/mri/x.png'),r.CROPS/'mri/x.png')
        self.assertEqual(r.locate('data/revision_crops/xray256/x.png'),r.REVISION_CROPS/'xray256/x.png')
        with self.assertRaises(ProtocolFailure):
            r.locate('C:/elsewhere/x.png')

    def test_xray_fallback_preserves_coverage_without_claiming_original_crop(self):
        # A narrow upper lung extension permits the first 256 crop, while only a
        # lower, wider region permits 512. No 512 square can contain the old crop.
        mask=np.zeros((768,768),np.uint8)
        mask[0:384,128:384]=255
        mask[128:640,128:640]=255
        small=np.random.default_rng(9).integers(0,256,(768,768),dtype=np.uint8)
        image=np.repeat(np.repeat(small,2,axis=0),2,axis=1)
        large_mask=np.repeat(np.repeat(mask,2,axis=0),2,axis=1)
        def png(a):
            stream=io.BytesIO();Image.fromarray(a).save(stream,format='PNG');return stream.getvalue()
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'NLM-MontgomeryCXRSet.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('MontgomerySet/CXR_png/test.png',png(image))
                z.writestr('MontgomerySet/ManualMask/leftMask/test.png',png(large_mask))
                z.writestr('MontgomerySet/ManualMask/rightMask/test.png',png(np.zeros_like(large_mask)))
            with patch.object(r,'pixels',lambda record:small[0:256,128:384]):
                a,meta=r.larger_xray(dict(source='data/'+archive.name+'::MontgomerySet/CXR_png/test.png',image_id='test',crop_x=128,crop_y=0),folder,None)
        self.assertTrue(meta['new_nested_pair'])
        self.assertEqual(a.shape,(512,512))
        x,y=meta['larger_crop_x'],meta['larger_crop_y']
        self.assertEqual(mask[y:y+512,x:x+512].min(),255)
        self.assertTrue(np.array_equal(a[128:384,128:384],small[y+128:y+384,x+128:x+384]))

    def test_progress_replace_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'progress.json'
            target.write_text('{"completed":1}')
            original=r.Path.replace
            calls=[]
            def locked_once(path,destination):
                calls.append(1)
                if len(calls)==1:
                    self.assertEqual(json.loads(target.read_text())['completed'],1)
                    raise PermissionError('simulated reader lock')
                return original(path,destination)
            with patch.object(r.Path,'replace',locked_once):
                r.save_json(target,dict(completed=2))
            self.assertTrue(json.loads(target.read_text())['completed']==2 and len(calls)==2)

    def test_parallel_failure_is_reported_and_raised(self):
        from src import headers
        def failed(task):
            raise ProtocolFailure('test worker failure')
        with tempfile.TemporaryDirectory() as folder, patch.object(r,'OUT',Path(folder)), patch.object(headers,'jxl_library',lambda:None):
            with self.assertRaisesRegex(ProtocolFailure,'test worker failure'):
                r.parallel([1,2,3],failed,'test_stage',1)
            report=json.loads((Path(folder)/'test_stage_failure.json').read_text())
            self.assertIn('test worker failure',report['error'])
            self.assertFalse((Path(folder)/'test_stage_results.csv').exists())


if __name__ == '__main__':
    unittest.main()
