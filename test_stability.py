"""Behavioral checks for post-clearance voltage analysis."""

import math
import unittest

from signal_analysis import StabilityCriteria, assess_stability


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.time = [index / 1000 for index in range(3001)]
        self.criteria = StabilityCriteria(
            clearance_time_s=0.5,
            evaluation_duration_s=1.0,
            final_reference_duration_s=1.0,
            minimum_simulation_end_time_s=3.0,
            minimum_final_reference_start_time_s=2.0,
            post_fault_persistence_start_time_s=2.0,
            post_fault_persistence_max_end_time_s=3.0,
            maximum_relative_error_area_s=0.03,
        )

    def assess(self, voltage, criteria=None, pre_fault_reference_value=None):
        return assess_stability(
            self.time,
            [voltage(time) for time in self.time],
            criteria or self.criteria,
            pre_fault_reference_value=pre_fault_reference_value,
        )

    def test_idd_uses_the_supplied_pre_fault_reference_over_the_fault_window(self):
        criteria = StabilityCriteria(
            fault_time_s=0.5,
            dynamic_performance_index_end_time_s=0.62,
            maximum_dynamic_performance_index_pu_s=0.02,
            clearance_time_s=0.5,
            evaluation_duration_s=1.0,
            final_reference_duration_s=1.0,
            minimum_simulation_end_time_s=3.0,
            minimum_final_reference_start_time_s=2.0,
            post_fault_persistence_start_time_s=2.0,
            post_fault_persistence_max_end_time_s=3.0,
        )

        result = self.assess(
            lambda time: 0.95 if 0.5 <= time <= 0.62 else 1.0,
            criteria,
            pre_fault_reference_value=1.05,
        )

        self.assertAlmostEqual(result.pre_fault_reference_value, 1.05)
        self.assertAlmostEqual(result.dynamic_performance_index_pu_s, 0.012, places=8)
        self.assertTrue(result.is_dynamic_performance_index_compliant)

    def test_unified_response_area_is_the_sum_of_fault_and_recovery_areas(self):
        criteria = StabilityCriteria(
            fault_time_s=0.5,
            dynamic_performance_index_end_time_s=0.62,
            response_area_end_time_s=1.0,
            clearance_time_s=0.5,
            evaluation_duration_s=1.0,
            final_reference_duration_s=1.0,
            minimum_simulation_end_time_s=3.0,
            minimum_final_reference_start_time_s=2.0,
            post_fault_persistence_start_time_s=2.0,
            post_fault_persistence_max_end_time_s=3.0,
        )

        result = self.assess(
            lambda time: (
                0.90
                if 0.5 <= time <= 0.62
                else 0.98 if 0.62 < time <= 1.0 else 1.0
            ),
            criteria,
            pre_fault_reference_value=1.0,
        )

        self.assertAlmostEqual(result.dynamic_performance_index_pu_s, 0.012, places=8)
        self.assertAlmostEqual(result.recovery_area_pu_s, 0.00764, places=8)
        self.assertAlmostEqual(
            result.unified_response_area_pu_s,
            result.dynamic_performance_index_pu_s + result.recovery_area_pu_s,
            places=8,
        )

    def test_post_fault_persistence_area_is_separate_from_unified_response(self):
        criteria = StabilityCriteria(
            fault_time_s=0.5,
            dynamic_performance_index_end_time_s=0.62,
            response_area_end_time_s=1.0,
            post_fault_persistence_start_time_s=1.5,
            post_fault_persistence_max_end_time_s=2.0,
            clearance_time_s=0.5,
            evaluation_duration_s=1.0,
            final_reference_duration_s=1.0,
            minimum_simulation_end_time_s=3.0,
            minimum_final_reference_start_time_s=2.0,
        )

        baseline = self.assess(lambda time: 1.0, criteria, 1.0)
        persistent = self.assess(
            lambda time: 0.95 if 1.5 <= time < 2.0 else 1.0,
            criteria,
            1.0,
        )

        self.assertAlmostEqual(
            persistent.unified_response_area_pu_s,
            baseline.unified_response_area_pu_s,
            places=8,
        )
        self.assertGreater(persistent.post_fault_persistence_area_pu_s, 0.0)

    def test_post_fault_persistence_area_uses_simulation_end_capped_at_ten_seconds(self):
        time = [index / 1000 for index in range(12001)]
        criteria = StabilityCriteria(
            fault_time_s=0.5,
            dynamic_performance_index_end_time_s=0.62,
            response_area_end_time_s=1.0,
            post_fault_persistence_start_time_s=2.0,
            post_fault_persistence_max_end_time_s=10.0,
            clearance_time_s=0.5,
            evaluation_duration_s=1.0,
            final_reference_duration_s=1.0,
            minimum_simulation_end_time_s=12.0,
            minimum_final_reference_start_time_s=11.0,
        )

        result = assess_stability(
            time,
            [0.9 if 2.0 <= instant <= 10.0 else 1.0 for instant in time],
            criteria,
            pre_fault_reference_value=1.0,
        )

        self.assertAlmostEqual(result.post_fault_persistence_end_time_s, 10.0)
        self.assertAlmostEqual(result.post_fault_persistence_area_pu_s, 0.8, places=8)

    def test_area_is_the_integral_of_absolute_relative_error(self):
        result = self.assess(
            lambda time: 1.02 if 0.5 <= time <= 1.5 else 1.0
        )

        self.assertAlmostEqual(result.final_reference_value, 1.0)
        self.assertAlmostEqual(result.relative_error_area_s, 0.02, places=8)
        self.assertAlmostEqual(
            result.post_clear_peak_absolute_relative_error, 0.02, places=8
        )
        self.assertTrue(result.is_area_within_limit)
        self.assertTrue(result.is_settled)

    def test_area_above_limit_is_unsettled(self):
        result = self.assess(
            lambda time: 1.04 if 0.5 <= time <= 1.5 else 1.0
        )

        self.assertFalse(result.is_area_within_limit)

    def test_limit_is_closed(self):
        provisional = self.assess(
            lambda time: 1.02 if 0.5 <= time <= 1.5 else 1.0
        )
        criterion_at_measurement = StabilityCriteria(
            clearance_time_s=0.5,
            evaluation_duration_s=1.0,
            final_reference_duration_s=1.0,
            minimum_simulation_end_time_s=3.0,
            minimum_final_reference_start_time_s=2.0,
            post_fault_persistence_start_time_s=2.0,
            post_fault_persistence_max_end_time_s=3.0,
            maximum_relative_error_area_s=provisional.relative_error_area_s,
        )

        result = self.assess(
            lambda time: 1.02 if 0.5 <= time <= 1.5 else 1.0,
            criterion_at_measurement,
        )

        self.assertTrue(result.is_area_within_limit)

    def test_short_simulation_is_not_classified(self):
        short_time = [index / 1000 for index in range(2500)]
        with self.assertRaises(ValueError):
            assess_stability(short_time, [1.0] * len(short_time), self.criteria)

    def test_default_reference_requires_a_complete_second_from_time_five(self):
        time = [index / 1000 for index in range(6000)]
        with self.assertRaises(ValueError):
            assess_stability(time, [1.0] * len(time))

    def test_recovery_group_uses_the_signal_after_the_area_window(self):
        recovered = self.assess(lambda time: 1.0)
        later_undervoltage = self.assess(
            lambda time: 0.79 if 1.6 <= time < 1.7 else 1.0
        )

        self.assertTrue(recovered.remains_above_voltage_threshold_after_evaluation)
        self.assertAlmostEqual(recovered.minimum_voltage_after_evaluation_pu, 1.0)
        self.assertFalse(
            later_undervoltage.remains_above_voltage_threshold_after_evaluation
        )
        self.assertAlmostEqual(
            later_undervoltage.minimum_voltage_after_evaluation_pu, 0.79
        )

    def test_final_settlement_requires_a_quiet_last_second(self):
        settled = self.assess(lambda time: 1.0)
        unsettled = self.assess(
            lambda time: 1.0 + 0.06 * math.sin(20 * time)
        )

        self.assertTrue(settled.is_settled)
        self.assertFalse(unsettled.is_settled)
        self.assertLessEqual(
            unsettled.final_settlement_peak_absolute_relative_error, 0.10
        )
        self.assertGreater(
            unsettled.final_settlement_peak_to_peak_relative_range, 0.10
        )


if __name__ == "__main__":
    unittest.main()
