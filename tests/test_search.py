"""Check endpoint direction, closest matches, and monotonicity classification."""
import unittest
import numpy as np
from src.search import match, scan


def evaluation(p, value):
    return dict(parameter=p, psnr=value, ssim=value / 100, encoded_bytes=100, bpp=1.)


class SearchTests(unittest.TestCase):
    def test_log_bisection_match(self):
        r, _ = match(lambda p: evaluation(p, 50 - np.log(p)), 'jpeg2000', 46., 'psnr')
        self.assertEqual(r['status'], 'matched')
        self.assertLessEqual(r['residual'], .02)

    def test_unreachable_direction(self):
        for target, status in [(100, 'unreachable_high'), (0, 'unreachable_low')]:
            r, trials = match(lambda p: evaluation(p, 50 - np.log(p)), 'jpeg2000', target, 'psnr')
            self.assertEqual(r['status'], status)
            self.assertEqual(len(trials['bisection']), 2)

    def test_reversal(self):
        r, _ = match(lambda p: evaluation(p, np.log(p)), 'jpeg2000', 2., 'psnr')
        self.assertEqual(r['status'], 'matched_scan')
        self.assertTrue(r['initially_nonmonotone'])
        self.assertEqual(r['scan_calls'], 64)

    def test_quantized_closest(self):
        r, trials = match(lambda p: evaluation(p, round(50 - np.log(p))), 'jpeg2000', 46.4, 'psnr')
        self.assertEqual(r['status'], 'matched')
        self.assertFalse(r['tolerance_met'])
        self.assertEqual(r['residual'], min(t['residual'] for t in trials['bisection']))

    def test_scan_preserves_better_bisection_observation(self):
        previous = dict(evaluation(2., 30.01), residual=.01, calls=5, status='nonmonotone', max_inversion=.2)
        visited = []
        def evaluate(p):
            visited.append(p)
            return evaluation(p, 31.)
        result, _ = scan(evaluate, 'jpeg2000', 30., 'psnr', previous, [])
        self.assertEqual(result['parameter'], 2.)
        self.assertEqual(result['status'], 'matched_scan')
        self.assertEqual(len(visited), 64)
        self.assertEqual(visited[0], 1.5)
        self.assertEqual(visited[-1], 1000.)
        self.assertEqual(result['calls'], 69)

    def test_scan_unmatched(self):
        previous = dict(evaluation(2., 32.), residual=2., calls=5, status='nonmonotone', max_inversion=.2)
        result, _ = scan(lambda p: evaluation(p, 31.), 'jpeg2000', 30., 'psnr', previous, [])
        self.assertEqual(result['status'], 'unmatched')
        self.assertEqual(result['path'], 'unmatched')


if __name__ == '__main__':
    unittest.main()
