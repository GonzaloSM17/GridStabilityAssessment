"""Decision-table checks for Grid Code compliance and dynamic stability."""

import unittest

from stability.dynamic_stability import assess_dynamic_stability


class GridCodeAssessmentTests(unittest.TestCase):
    def test_all_criteria_must_pass_for_grid_code_compliance(self):
        compliant = assess_dynamic_stability(1, "Stable", "Damped", "Stable")
        self.assertEqual(compliant.grid_code_status, "Compliant")
        self.assertEqual(compliant.status, "Stable")

        voltage = assess_dynamic_stability(2, "Unstable", "Damped", "Stable")
        damping = assess_dynamic_stability(3, "Stable", "Damped under", "Stable")
        angle = assess_dynamic_stability(4, "Stable", "Damped", "Unstable")
        for result in (voltage, damping, angle):
            self.assertEqual(result.grid_code_status, "Non-compliant")

    def test_positive_under_damping_is_stable_but_non_compliant(self):
        result = assess_dynamic_stability(1, "Stable", "Damped under", "Stable")
        self.assertEqual(result.grid_code_status, "Non-compliant")
        self.assertEqual(result.status, "Stable")

    def test_growing_mode_is_non_compliant_and_dynamically_unstable(self):
        result = assess_dynamic_stability(1, "Stable", "Growing", "Stable")
        self.assertEqual(result.grid_code_status, "Non-compliant")
        self.assertEqual(result.status, "Unstable")

    def test_missing_required_evidence_is_not_evaluated_and_inconclusive(self):
        result = assess_dynamic_stability(1, "Stable", "Damped", None)
        self.assertEqual(result.grid_code_status, "Not evaluated")
        self.assertEqual(result.status, "Inconclusive")


if __name__ == "__main__":
    unittest.main()
