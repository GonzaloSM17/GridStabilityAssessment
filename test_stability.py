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

    def test_response_areas_partition_the_unified_event_response(self):
        voltage = [1.0] * len(self.time)
        for index, time in enumerate(self.time):
            if 1.0 <= time <= 1.12:
                voltage[index] = 0.90
            elif 1.12 < time <= 2.0:
                voltage[index] = 0.98

        result = self.assess(voltage)

        self.assertAlmostEqual(result.fault_on_area_pu_s, 0.012, places=8)
        self.assertAlmostEqual(result.recovery_area_pu_s, 0.01764, places=8)
        self.assertAlmostEqual(
            result.unified_response_area_pu_s,
            result.fault_on_area_pu_s + result.recovery_area_pu_s,
            places=8,
        )

    def test_low_voltage_duration_accumulates_separate_intervals(self):
        voltage = [1.0] * len(self.time)
        for index, time in enumerate(self.time):
            if 1.0 <= time < 1.5 or 2.0 <= time < 2.6:
                voltage[index] = 0.70

        result = self.assess(voltage)

        self.assertAlmostEqual(result.cumulative_low_voltage_duration_s, 1.099, places=3)
        self.assertFalse(result.is_low_voltage_duration_compliant)

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
