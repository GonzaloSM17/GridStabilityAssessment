"""Tests for explicit contingency scenarios and their active criteria profile."""

import unittest

from scenario import ContingencyScenario, EvaluationProfile


class EvaluationProfileTests(unittest.TestCase):
    """Verify that scenario timing is shared by all active analyses."""

    def test_profile_inherits_fault_and_clearance_times(self):
        scenario = ContingencyScenario(
            name="Example CC-2PH-G",
            severity=4,
            fault_type="two_phase_to_ground",
            fault_time_s=2.0,
            clearance_time_s=2.15,
            location="EXAMPLE_220",
        )

        profile = EvaluationProfile.for_scenario(scenario)

        self.assertEqual(profile.initialization_criteria.evaluation_start_time_s, 1.5)
        self.assertEqual(profile.initialization_criteria.evaluation_end_time_s, 2.0)
        self.assertEqual(profile.voltage_criteria.fault_time_s, 2.0)
        self.assertEqual(profile.voltage_criteria.clearance_time_s, 2.15)
        self.assertEqual(profile.transfer_damping_criteria.fault_time_s, 2.0)
        self.assertEqual(profile.transfer_damping_criteria.clearance_time_s, 2.15)
        self.assertEqual(profile.transfer_damping_criteria.pre_fault_start_time_s, 1.9)


if __name__ == "__main__":
    unittest.main()
