"""Tests for terminal P/Q operating-point recovery."""

import unittest

from stability.plant_recovery import PlantRecoveryCriteria, assess_plant_recovery


class PlantRecoveryTests(unittest.TestCase):
    """Verify the explicit 0.5 s reference and tmax decision."""

    def test_uses_interpolated_pre_fault_reference_and_terminal_values(self):
        result = assess_plant_recovery(
            (0.0, 0.4, 0.6, 2.12, 3.0),
            (0.0, 100.0, 120.0, 80.0, 116.0),
            (0.0, 10.0, 20.0, 6.0, 16.0),
            plant_name="TEST_PV",
            technology="PV",
            clearance_time_s=1.12,
        )

        self.assertAlmostEqual(result.pre_fault_active_power_mw, 110.0)
        self.assertAlmostEqual(result.pre_fault_reactive_power_mvar, 15.0)
        self.assertEqual(result.evaluation_start_time_s, 2.12)
        self.assertEqual(result.final_time_s, 3.0)
        self.assertAlmostEqual(result.active_power_error_mw, 6.0)
        self.assertAlmostEqual(result.reactive_power_error_mvar, 1.0)
        self.assertAlmostEqual(result.maximum_active_power_absolute_error_mw, 30.0)
        self.assertAlmostEqual(result.maximum_active_power_error_mw, -30.0)
        self.assertAlmostEqual(result.maximum_active_power_error_time_s, 2.12)
        self.assertAlmostEqual(result.maximum_reactive_power_absolute_error_mvar, 9.0)
        self.assertAlmostEqual(result.maximum_reactive_power_error_mvar, -9.0)
        self.assertAlmostEqual(result.maximum_reactive_power_error_time_s, 2.12)
        self.assertFalse(result.is_active_power_recovered)
        self.assertTrue(result.is_reactive_power_recovered)
        self.assertEqual(result.status, "P not recovered")

    def test_requires_the_full_post_clearance_observation_start(self):
        with self.assertRaisesRegex(ValueError, "evaluation period"):
            assess_plant_recovery(
                (0.0, 0.5, 2.0),
                (0.0, 10.0, 10.0),
                (0.0, 2.0, 2.0),
                plant_name="TEST_BESS",
                technology="BESS",
                clearance_time_s=1.12,
            )

    def test_tolerances_are_absolute_mw_and_mvar(self):
        criteria = PlantRecoveryCriteria(
            active_power_tolerance_mw=5.0,
            reactive_power_tolerance_mvar=5.0,
        )
        result = assess_plant_recovery(
            (0.0, 0.5, 2.12, 3.0),
            (0.0, 0.0, 0.0, -5.0),
            (0.0, 0.0, 0.0, 5.0),
            plant_name="TEST_WF",
            technology="WF",
            clearance_time_s=1.12,
            criteria=criteria,
        )

        self.assertTrue(result.is_recovered)


if __name__ == "__main__":
    unittest.main()
