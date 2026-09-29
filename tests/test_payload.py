"""Known payload arithmetic, eligibility, and strict-win ranking cases."""
import unittest

import pandas as pd

from src.payload import payload_measurements, payload_statistics


def fixture():
    pairs, headers = [], []
    for domain in ['natural', 'xray', 'mri', 'aerial']:
        for i in range(3):
            base = dict(domain=domain, image_id=str(i), q=50)
            pair = dict(base, jpeg_bytes=100)
            headers.append(dict(base, codec='jpeg', metric='target', encoded_bytes=100, header_bytes=20))
            for codec, size, header in [('jpeg2000', 90-10*i, 40), ('jpegxl', 80-10*i, 10)]:
                for metric in ['psnr', 'ssim']:
                    pair[f'{codec}_{metric}_status'] = 'matched_scan' if i else 'matched'
                    pair[f'{codec}_{metric}_encoded_bytes'] = size
                    headers.append(dict(base, codec=codec, metric=metric, encoded_bytes=size, header_bytes=header))
            pairs.append(pair)
    return pd.DataFrame(headers), pd.DataFrame(pairs)


class PayloadTests(unittest.TestCase):
    def test_payload_savings_and_ranking_reversal(self):
        headers, pairs = fixture()
        data = payload_measurements(headers, pairs)
        values, rates, ranking, _ = payload_statistics(data)
        self.assertEqual(values['SaveJtkPsnrPayMin'], 50.)
        self.assertEqual(values['SaveJtkPsnrPayMax'], 50.)
        self.assertEqual(values['SaveJxlPsnrPayMin'], 25.)
        self.assertEqual(values['JxlWinsPsnrPay'], 0.)
        self.assertTrue(ranking[ranking.basis == 'file'].win_pct.eq(100).all())
        self.assertTrue(ranking[ranking.basis == 'payload'].majority.eq('JPEG2000').all())
        self.assertEqual(values['HdrJpegBytes'], 20)
        self.assertEqual(len(data), 60)

    def test_ties_are_not_wins_and_remain_in_denominator(self):
        headers, pairs = fixture()
        headers.loc[headers.codec == 'jpegxl', 'header_bytes'] = 30
        data = payload_measurements(headers, pairs)
        _, _, ranking, _ = payload_statistics(data)
        pay = ranking[ranking.basis == 'payload']
        self.assertTrue(pay.win_pct.eq(0).all())
        self.assertTrue(pay.ties.eq(pay.n).all())
        self.assertTrue(pay.majority.eq('no majority').all())

    def test_missing_header_is_integrity_error(self):
        headers, pairs = fixture()
        with self.assertRaisesRegex(RuntimeError, 'coverage'):
            payload_measurements(headers.iloc[1:], pairs)

    def test_unaccepted_target_is_excluded_only_for_its_metric(self):
        headers, pairs = fixture()
        pairs.loc[0, 'jpegxl_ssim_status'] = 'unmatched'
        keep = ~((headers.domain == 'natural') & (headers.image_id == '0') &
                 (headers.codec == 'jpegxl') & (headers.metric == 'ssim'))
        data = payload_measurements(headers[keep], pairs)
        _, _, ranking, _ = payload_statistics(data)
        full = ranking[(ranking.domain == 'overall') & (ranking.basis == 'payload')].set_index('metric')
        self.assertEqual(full.loc['psnr', 'n'], 12)
        self.assertEqual(full.loc['ssim', 'n'], 11)


if __name__ == '__main__':
    unittest.main()
