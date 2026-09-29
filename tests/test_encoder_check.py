import unittest
from unittest.mock import patch

import numpy as np

from src import encoder_check


class RerunSanityTests(unittest.TestCase):
    def test_replicated_gray_has_identical_codec_size_and_quality(self):
        gray = np.random.default_rng(0).integers(0, 256, (32, 32), dtype=np.uint8)
        replicated = np.repeat(gray[..., None], 3, axis=2)
        for codec in ['jpeg', 'jpegxl', *encoder_check.CONFIGS]:
            parameter = 50 if codec == 'jpeg' else 2.
            self.assertEqual(encoder_check.encode(gray, codec, parameter),
                             encoder_check.encode(replicated, codec, parameter))

    def test_search_retains_closest_and_reports_unreachable(self):
        def encode(a, codec, p):
            value = 40. if p < 10. else 30.
            return dict(psnr=value, bpp=1., encoded_bytes=1)
        with patch.object(encoder_check, 'encode', encode):
            result = encoder_check.search(None, 'j2k_both_fixed', 37., max_calls=12)
            self.assertEqual(result['psnr'], 40.)
            self.assertEqual(result['residual_db'], 3.)
            self.assertEqual(result['encoder_calls'], 12)
            self.assertFalse(result['tolerance_met'])
            self.assertFalse(result['out_of_range'])
            outside = encoder_check.search(None, 'j2k_both_fixed', 45.)
            self.assertTrue(outside['out_of_range'])
            self.assertEqual(outside['encoder_calls'], 2)


if __name__ == '__main__':
    unittest.main()
