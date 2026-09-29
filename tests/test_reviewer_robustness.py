import unittest
import numpy as np

from src.reference_robustness import quantizer, Q50, entropy


class ReferenceQuantizerTests(unittest.TestCase):
    def test_ijg_integer_scaling(self):
        np.testing.assert_array_equal(quantizer(50),Q50)
        np.testing.assert_array_equal(quantizer(30)[0],[27,18,17,27,40,66,85,101])
        np.testing.assert_array_equal(quantizer(70)[0],[10,7,6,10,14,24,31,37])
        # IJG's integer scale 5000//30 differs from floating 5000/30.
        self.assertEqual(quantizer(30)[5,7],153)

    def test_entropy_of_known_coefficients(self):
        coefficients = np.zeros((2,8,8))
        self.assertEqual(entropy(coefficients,quantizer(30)),0.)
        coefficients[1,:,:] = quantizer(70)
        self.assertEqual(entropy(coefficients,quantizer(70)),1.)


if __name__ == '__main__':
    unittest.main()
