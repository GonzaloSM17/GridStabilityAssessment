"""Unit tests for active-power transfer damping assessment."""

from math import exp, pi, sqrt
import unittest

from transfer_analysis import assess_transfer_damping


class TransferDampingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.time = [index / 1000 for index in range(5001)]

    def damped_power(self, *, starts_negative: bool = False):
        damping_ratio = 0.10
        frequency_hz = 2.0
        decay = 2.0 * pi * frequency_hz * damping_ratio / sqrt(1.0 - damping_ratio**2)
        polarity = -1.0 if starts_negative else 1.0
        values = []
        for time in self.time:
            if time < 1.0:
                values.append(100.0)
            else:
                values.append(
                    90.0
                    + polarity
                    * 20.0
                    * exp(-decay * (time - 1.12))
                    * __import__("math").cos(2.0 * pi * frequency_hz * (time - 1.12))
                )
        return values

    def test_records_all_reliable_pairs_and_the_lowest_damping(self):
        result = assess_transfer_damping(
            self.time,
            self.damped_power(),
            run_id=1,
            transfer_name="NC_TEST_C1",
        )

        self.assertEqual(result.status, "pass")
        self.assertGreater(len(result.damping_pairs), 2)
        self.assertAlmostEqual(result.damping_ratio, 0.10, places=2)
        self.assertEqual(
            result.damping_ratio,
            min(pair.damping_ratio for pair in result.damping_pairs),
        )

    def test_negative_initial_excursion_uses_negative_minima(self):
        result = assess_transfer_damping(
            self.time,
            self.damped_power(starts_negative=True),
            run_id=1,
            transfer_name="NC_TEST_C1",
        )

        self.assertEqual(result.status, "pass")
        self.assertTrue(any(pair.polarity == "negative" for pair in result.damping_pairs))

    def test_opened_transfer_is_not_a_damping_failure(self):
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
