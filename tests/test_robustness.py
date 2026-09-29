"""Scientific invariants for the SSIM robustness extension."""
import unittest

import numpy as np
import pandas as pd

from src.robustness import fit_ssim, table3


class RobustnessTests(unittest.TestCase):
    def test_known_log_linear_relation_and_test_target_independence(self):
        rows = []
        for domain in ['natural', 'xray']:
            for q in [10, 50]:
                for i in range(1, 7):
                    rows.append(dict(domain=domain, q=q, split='train' if i < 5 else 'test', C_DCT=float(i),
                        jpeg2000_ssim_status='matched', jpegxl_ssim_status='matched_scan',
                        jpeg2000_ssim_parameter=12/i**2, jpegxl_ssim_parameter=2/i))
        data = pd.DataFrame(rows)
        models = fit_ssim(data)
        changed = data.copy()
        changed.loc[changed.split == 'test', ['jpeg2000_ssim_parameter', 'jpegxl_ssim_parameter']] *= 1e6
        pd.testing.assert_frame_equal(models, fit_ssim(changed))
        jtk = models[(models.codec == 'jpeg2000') & (models.predictor == 'C1')]
        np.testing.assert_allclose(jtk.slope, -2, atol=1e-14)
        np.testing.assert_allclose(jtk.intercept, np.log(12), atol=1e-14)
        self.assertFalse(((models.domain == 'xray') & (models.q == 10) & (models.codec == 'jpegxl')).any())
        self.assertTrue(models.train_n.eq(4).all())

    def test_table_bolds_best_mean_not_best_tail_or_floor(self):
        rows = []
        for domain in ['natural', 'xray', 'mri', 'aerial']:
            for codec in ['jpeg2000', 'jpegxl']:
                for p, mean, p90 in [('T0', 3., 4.), ('T1', 2., 3.), ('C0', 1.5, 2.), ('C1', 1., 5.), ('floor', .1, .2)]:
                    rows.append(dict(domain=domain, codec=codec, predictor=p, mean=mean, p90=p90))
        text = table3(pd.DataFrame(rows))
        lines = text.splitlines()
        self.assertEqual(lines.count(r'\midrule'), 3)
        rows = [line for line in lines if '&' in line]
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(len(row.split('&')), 12)
            self.assertEqual(row.count(r'\textbf{'), 1)
            self.assertIn(r'\textbf{1.00} & 5.00 & 0.10 & 0.20', row)


if __name__ == '__main__':
    unittest.main()
