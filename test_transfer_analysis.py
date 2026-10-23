"""Tests for electromechanical-mode transfer damping assessment."""

from math import exp, pi, sqrt
import unittest

from transfer_analysis import assess_transfer_damping


class TransferDampingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.time = [index / 1000 for index in range(5001)]

    def power_with_mode(
        self,
        *,
        mode_frequency_hz: float = 1.0,
        damping_ratio: float = 0.10,
        starts_negative: bool = False,
        high_frequency_ripple: bool = False,
    ) -> list[float]:
        decay = (
            2.0
            * pi
            * mode_frequency_hz
            * damping_ratio
            / sqrt(1.0 - damping_ratio**2)
        )
        polarity = -1.0 if starts_negative else 1.0
        values = []
        for time in self.time:
            if time < 1.0:
                values.append(100.0)
                continue
            mode = (
                polarity
                * 20.0
                * exp(-decay * (time - 1.12))
                * __import__("math").cos(2.0 * pi * mode_frequency_hz * (time - 1.12))
            )
            ripple = (
                5.0 * __import__("math").cos(2.0 * pi * 12.0 * (time - 1.12))
                if high_frequency_ripple
                else 0.0
            )
            values.append(90.0 + mode + ripple)
        return values

    def test_identifies_the_dominant_electromechanical_mode(self):
        result = assess_transfer_damping(
            self.time,
            self.power_with_mode(high_frequency_ripple=True),
            run_id=1,
            transfer_name="NC_TEST_C1",
        )

        self.assertEqual(result.status, "compliant")
        self.assertAlmostEqual(result.dominant_frequency_hz, 1.0, delta=0.3)
        self.assertIsNotNone(result.selected_damping_pair)
        self.assertAlmostEqual(result.damping_ratio, 0.10, delta=0.03)
        self.assertGreaterEqual(result.selected_damping_pair.period_s, 0.5)

    def test_negative_initial_excursion_uses_same_sign_valleys(self):
        result = assess_transfer_damping(
            self.time,
            self.power_with_mode(starts_negative=True),
            run_id=1,
            transfer_name="NC_TEST_C1",
        )

        self.assertEqual(result.status, "compliant")
        self.assertEqual(result.selected_damping_pair.polarity, "negative")

    def test_positive_initial_excursion_uses_same_sign_peaks(self):
        result = assess_transfer_damping(
            self.time,
            self.power_with_mode(starts_negative=False),
            run_id=1,
            transfer_name="NC_TEST_C1",
        )

        self.assertEqual(result.status, "compliant")
        self.assertEqual(result.selected_damping_pair.polarity, "positive")

    def test_fast_ripple_is_not_an_electromechanical_mode(self):
        values = []
        for time in self.time:
            values.append(
                100.0
                if time < 1.0
                else 90.0 + 20.0 * __import__("math").cos(2.0 * pi * 12.0 * (time - 1.12))
            )

        result = assess_transfer_damping(
            self.time,
            values,
            run_id=1,
            transfer_name="NC_TEST_C1",
        )

        self.assertIsNone(result.damping_ratio)
        self.assertIn(
            result.status,
            {"not_evaluated"},
        )

    def test_classifies_positive_but_low_damping_as_under_damped(self):
        result = assess_transfer_damping(
            self.time,
            self.power_with_mode(damping_ratio=0.02),
            run_id=1,
            transfer_name="NC_TEST_C1",
        )

        self.assertEqual(result.status, "under_damped")
        self.assertGreater(result.damping_ratio, 0.0)
        self.assertLess(result.damping_ratio, 0.05)

    def test_opened_transfer_is_not_a_damping_warning(self):
        values = [100.0 if time < 1.0 else 0.0 for time in self.time]
        result = assess_transfer_damping(
            self.time,
            values,
            run_id=1,
            transfer_name="NC_TEST_C1",
        )

        self.assertEqual(result.status, "opened")
        self.assertIsNone(result.damping_ratio)


if __name__ == "__main__":
    unittest.main()
