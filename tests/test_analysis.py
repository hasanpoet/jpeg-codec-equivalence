"""Scientific invariants: no test-target leakage and known entropy/CI cases."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features import complexity
from src.models import fit_models, difference_bootstrap


class AnalysisTests(unittest.TestCase):
    def test_pooled_dct_entropy_known_constant(self):
        self.assertEqual(complexity(np.full((256,256), 128, dtype=np.uint8)), 0.)
        p = 1 / 64
        expected = -p * np.log2(p) - (1-p) * np.log2(1-p)
        self.assertAlmostEqual(complexity(np.zeros((256,256), dtype=np.uint8)), expected, places=12)

    def test_test_targets_cannot_change_fits(self):
        pairs, features = [], []
        for domain in ['natural', 'mri']:
            for i in range(6):
                image = domain + str(i)
                features.append(dict(domain=domain,image_id=image,group=image,split='train' if i < 4 else 'test',C_DCT=i+1.,C_BPP=2*(i+1.)))
                pairs.append(dict(domain=domain,image_id=image,group=image,q=50,
                    jpeg2000_psnr_parameter=10/(i+1.),jpegxl_psnr_parameter=2/(i+1.),
                    jpeg2000_psnr_status='matched',jpegxl_psnr_status='matched_scan'))
        pairs, features = pd.DataFrame(pairs), pd.DataFrame(features)
        with tempfile.TemporaryDirectory() as folder, patch('src.models.OUT',Path(folder)):
            _, first = fit_models(pairs,features)
            test = pairs.image_id.isin(features.loc[features.split.eq('test'),'image_id'])
            pairs.loc[test,['jpeg2000_psnr_parameter','jpegxl_psnr_parameter']] *= 100000
            _, second = fit_models(pairs,features)
        pd.testing.assert_frame_equal(first,second)

    def test_paired_cluster_ci_constant_difference(self):
        rows = []
        for domain in ['natural','xray','mri','aerial']:
            for group, size in [('one',1),('many',5)]:
                for image in range(size):
                    for predictor,error in [('T1',3.),('C1',1.)]:
                        rows.append(dict(domain=domain,group=group,image_id=str(image),q=50,
                                         codec='jpeg2000',predictor=predictor,error_db=error))
        with patch('src.models.REPLICATES',100):
            result = difference_bootstrap(pd.DataFrame(rows),'jpeg2000','C1',np.random.default_rng(0))
        self.assertEqual(result['difference'],2.)
        self.assertEqual(result['lo'],2.)
        self.assertEqual(result['hi'],2.)

    def test_equal_domain_weights_despite_unequal_image_counts(self):
        rows = []
        for domain, size, difference in [('natural', 1, 1.), ('xray', 2, 2.),
                                         ('mri', 20, 3.), ('aerial', 3, 4.)]:
            for group in ['a', 'b']:
                for image in range(size):
                    for predictor, error in [('T1', 5. + difference), ('C1', 5.)]:
                        rows.append(dict(domain=domain, group=group, image_id=str(image), q=50,
                                         codec='jpegxl', predictor=predictor, error_db=error))
        with patch('src.models.REPLICATES', 100):
            result = difference_bootstrap(pd.DataFrame(rows), 'jpegxl', 'C1', np.random.default_rng(0))
        self.assertEqual(result['difference'], 2.5)
        self.assertEqual(result['lo'], 2.5)
        self.assertEqual(result['hi'], 2.5)

    def test_exclusion_removes_cell_from_domain_and_global_fits(self):
        pairs, features = [], []
        for domain in ['natural', 'xray']:
            for i in range(3):
                image = domain + str(i)
                features.append(dict(domain=domain, image_id=image, group=image,
                                     split='train', C_DCT=i+1., C_BPP=i+1.))
                pairs.append(dict(domain=domain, image_id=image, group=image, q=10,
                    jpeg2000_psnr_parameter=10., jpegxl_psnr_parameter=20. if domain == 'xray' else 2.,
                    jpeg2000_psnr_status='matched', jpegxl_psnr_status='matched'))
        with tempfile.TemporaryDirectory() as folder, patch('src.models.OUT', Path(folder)):
            _, models = fit_models(pd.DataFrame(pairs), pd.DataFrame(features))
        jxl = models[models.codec == 'jpegxl']
        self.assertFalse((jxl.domain == 'xray').any())
        self.assertTrue((jxl.train_n == 3).all())
        self.assertTrue((models[(models.codec == 'jpeg2000') & (models.domain == 'all')].train_n == 6).all())


if __name__ == '__main__':
    unittest.main()
