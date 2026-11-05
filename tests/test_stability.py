"""Unit tests for the voltage response and recovery assessment."""

import unittest

from stability.voltage import StabilityCriteria, assess_stability


class StabilityAssessmentTests(unittest.TestCase):
    """Verify the active voltage criteria and event-response areas."""

    def setUp(self) -> None:
        self.time = [index / 1000 for index in range(5001)]
        self.criteria = StabilityCriteria()

    def assess(self, voltage):
        return assess_stability(self.time, voltage, self.criteria)

    def test_idd_covers_the_full_relative_evaluation_window(self):
        voltage = [1.0] * len(self.time)
        for index, time in enumerate(self.time):
            if 1.0 <= time <= 1.12:
                voltage[index] = 0.90
            elif 1.12 < time <= 2.0:
                voltage[index] = 0.98

        result = self.assess(voltage)

        self.assertAlmostEqual(result.fault_on_area_pu_s, 0.012, places=8)
        self.assertAlmostEqual(result.recovery_area_pu_s, 0.01764, places=8)
        self.assertGreater(
            result.idd_area_pu_s,
            result.fault_on_area_pu_s + result.recovery_area_pu_s,
        )
        self.assertAlmostEqual(result.idd_area_pu_s, 0.02965, places=8)

    def test_idd_window_moves_with_fault_time(self):
        standard_voltage = [1.0] * len(self.time)
        shifted_time = [index / 1000 for index in range(6001)]
        shifted_voltage = [1.0] * len(shifted_time)
        for values, times, fault_time in (
            (standard_voltage, self.time, 1.0),
            (shifted_voltage, shifted_time, 2.0),
        ):
            for index, time in enumerate(times):
                relative_time = time - fault_time
                if 0.0 <= relative_time <= 0.12:
                    values[index] = 0.90
                elif 0.12 < relative_time <= 1.0:
                    values[index] = 0.98

        standard = self.assess(standard_voltage)
        shifted = assess_stability(
            shifted_time,
            shifted_voltage,
            StabilityCriteria(
                fault_time_s=2.0,
                clearance_time_s=2.12,
                response_end_time_s=3.0,
            ),
        )
        self.assertAlmostEqual(standard.idd_area_pu_s, shifted.idd_area_pu_s)

    def test_idd_requires_the_complete_evaluation_window(self):
        short_time = [index / 1000 for index in range(2111)]
        criteria = StabilityCriteria(
            minimum_simulation_end_time_s=2.11,
            minimum_final_reference_start_time_s=0.0,
            final_reference_duration_s=0.1,
            final_settlement_duration_s=0.1,
        )
        with self.assertRaisesRegex(ValueError, "IDD evaluation endpoint"):
            assess_stability(short_time, [1.0] * len(short_time), criteria)

    def test_low_voltage_duration_accumulates_separate_intervals(self):
        voltage = [1.0] * len(self.time)
        for index, time in enumerate(self.time):
            if 1.0 <= time < 1.5 or 2.0 <= time < 2.6:
                voltage[index] = 0.70

        result = self.assess(voltage)

        self.assertAlmostEqual(result.cumulative_low_voltage_duration_s, 1.099, places=3)
        self.assertFalse(result.is_low_voltage_duration_compliant)

    def test_late_080_pu_reentry_fails_the_existing_low_voltage_criterion(self):
        voltage = [1.0] * len(self.time)
        for index, time in enumerate(self.time):
            if 2.3 <= time < 2.31:
                voltage[index] = 0.79

        result = self.assess(voltage)

        self.assertLess(result.cumulative_low_voltage_duration_s, 1.0)
        self.assertFalse(result.is_low_voltage_duration_compliant)
        self.assertIn("Cumulative time below 0.80 pu", result.failed_criteria)

    def test_crisis_voltage_uses_the_final_post_clearance_upward_crossing(self):
        voltage = [1.0] * len(self.time)
        for index, time in enumerate(self.time):
            if 1.0 <= time < 1.16:
                voltage[index] = 0.60

        result = self.assess(voltage)
        self.assertTrue(result.is_crisis_voltage_compliant)
        self.assertEqual(result.crisis_decision_event, "recovery_crossing")
        self.assertAlmostEqual(
            result.crisis_decision_duration_s,
            0.03925,
            places=3,
        )

        for index, time in enumerate(self.time):
            if 1.2 <= time < 1.28:
                voltage[index] = 0.69
        result = self.assess(voltage)
        self.assertFalse(result.is_crisis_voltage_compliant)
        self.assertAlmostEqual(
            result.crisis_decision_duration_s,
            0.0799677,
            places=3,
        )

    def test_late_critical_contact_fails_even_when_the_signal_recovers(self):
        voltage = [1.0] * len(self.time)
        voltage[int(1.2 * 1000)] = 0.70

        result = self.assess(voltage)

        self.assertFalse(result.is_crisis_voltage_compliant)
        self.assertAlmostEqual(result.crisis_decision_time_s, 1.2, places=6)

    def test_recovery_at_the_50_ms_deadline_is_not_strictly_compliant(self):
        voltage = [1.0] * len(self.time)
        for index, time in enumerate(self.time):
            if 1.0 <= time < 1.17:
                voltage[index] = 0.60
        voltage[int(1.17 * 1000)] = 0.70

        result = self.assess(voltage)

        self.assertFalse(result.is_crisis_voltage_compliant)
        self.assertAlmostEqual(result.crisis_decision_time_s, 1.17, places=6)

    def test_final_settlement_uses_the_last_second_of_a_five_second_run(self):
        voltage = [1.0] * len(self.time)
        for index, time in enumerate(self.time):
            if time >= 4.5:
                voltage[index] = 1.12

        result = self.assess(voltage)
        self.assertFalse(result.is_settled)
        self.assertIn("Final settlement", result.failed_criteria)

    def test_short_simulation_is_not_evaluable(self):
        short_time = [index / 1000 for index in range(4999)]
        with self.assertRaisesRegex(ValueError, "minimum stability evaluation time"):
            assess_stability(short_time, [1.0] * len(short_time), self.criteria)


if __name__ == "__main__":
    unittest.main()
