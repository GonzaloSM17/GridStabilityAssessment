"""Behavioral checks for pre-contingency RMS-voltage initialization."""

import math
import unittest

from stability.initialization import assess_initialization


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.time = [index / 1000 for index in range(1201)]

    def assess(self, voltage):
        return assess_initialization(self.time, [voltage(t) for t in self.time])

    def test_constant_and_slow_motion_pass_at_different_voltage_scales(self):
        for scale in (1.0, 220.0, 220000.0):
            for slope in (0.0, 0.004, -0.004):
                with self.subTest(scale=scale, slope=slope):
                    self.assertTrue(self.assess(lambda t: scale * (1 + slope * t)).is_well_initialized)

    def test_reference_is_the_median_of_the_analysis_window(self):
        result = self.assess(lambda t: 1.0 + t)
        self.assertAlmostEqual(result.reference_value, 1.7495)

    def test_fast_drift_is_allowed_when_it_has_no_fast_oscillation(self):
        result = self.assess(lambda t: 1 + 0.02 * t)
        self.assertTrue(result.is_well_initialized)

    def test_persistent_fast_oscillations_fail(self):
        for frequency in (12, 40):
            with self.subTest(frequency=frequency):
                result = self.assess(lambda t: 1 + 0.015 * math.sin(2 * math.pi * frequency * t))
                self.assertFalse(result.is_well_initialized)
                self.assertIn("fast_oscillation", result.rejection_reasons)

    def test_slow_oscillation_is_allowed(self):
        self.assertTrue(self.assess(
            lambda t: 1 + 0.015 * math.sin(2 * math.pi * 2 * t)
        ).is_well_initialized)

    def test_small_ripple_with_slow_motion_passes(self):
        self.assertTrue(self.assess(
            lambda t: 1 + 0.003 * t + 0.001 * math.sin(2 * math.pi * 40 * t)
        ).is_well_initialized)

    def test_early_transient_is_allowed(self):
        self.assertTrue(self.assess(
            lambda t: 1 + 0.1 * math.exp(-25 * t) * math.cos(2 * math.pi * 20 * t)
        ).is_well_initialized)

    def test_short_final_excursion_is_rejected(self):
        result = self.assess(lambda t: 1.02 if 0.985 <= t < 0.988 else 1.0)
        self.assertIn("fast_oscillation", result.rejection_reasons)

    def test_contingency_does_not_affect_initialization(self):
        self.assertTrue(self.assess(lambda t: 1.0 if t < 1.0 else 0.3).is_well_initialized)

    def test_incomplete_or_invalid_time_is_not_classified_as_initialized(self):
        for times in (self.time[:950], self.time[700:], list(reversed(self.time)),
                      [t for t in self.time if not 0.7 < t < 0.8]):
            with self.subTest(length=len(times)):
                with self.assertRaises(ValueError):
                    assess_initialization(times, [1.0] * len(times))

    def test_invalid_voltage_is_rejected(self):
        for value in (0.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.assess(lambda t: value)


if __name__ == "__main__":
    unittest.main()
