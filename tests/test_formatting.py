"""Negative-zero suppression acts on display rounding, not computed values."""
import unittest

from src.formatting import format_number
from src.reporting import format_value


class FormattingTests(unittest.TestCase):
    def test_two_decimal_rounding(self):
        for value, expected in [(-.004, '0.00'), (-.006, '-0.01'), (0., '0.00'),
                                (-.05, '-0.05'), (-0., '0.00')]:
            with self.subTest(value=value):
                self.assertEqual(format_number(value, '.2f'), expected)
                self.assertEqual(format_value('AnyNumericValue', value), expected)

    def test_other_display_precisions(self):
        self.assertEqual(format_number(-.04, '.1f'), '0.0')
        self.assertEqual(format_number(-.06, '.1f'), '-0.1')
        self.assertEqual(format_number(-0., '.3g'), '0')
        self.assertEqual(format_number(-.004, '.3g'), '-0.004')
        self.assertEqual(format_number(-0., ',.0f'), '0')


if __name__ == '__main__':
    unittest.main()
