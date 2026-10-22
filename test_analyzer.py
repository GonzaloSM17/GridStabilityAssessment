"""Focused tests for Monte Carlo analyzer presentation helpers."""

import unittest

from analyzer import AnalysisPipeline, SignalStudyAnalysis, _round_up_to_nice_number


class NiceNumberTests(unittest.TestCase):
    def test_rounds_severity_upper_limit_up_to_a_nice_number(self):
        self.assertEqual(_round_up_to_nice_number(4.07), 5.0)
        self.assertEqual(_round_up_to_nice_number(0.101), 0.2)
        self.assertEqual(_round_up_to_nice_number(5.0), 5.0)

    def test_rejects_non_positive_scale_limits(self):
        for value in (0.0, -1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _round_up_to_nice_number(value)


class NoStabilityAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.analysis = SignalStudyAnalysis(
            signal_name="test/voltage",
            run_results=(),
            skipped_run_ids=(),
        )

    def test_analysis_exposes_the_absence_of_stability_assessments(self):
        self.assertFalse(self.analysis.has_stability_assessments)

    def test_controller_area_plots_report_a_clear_error_without_assessments(self):
        with self.assertRaisesRegex(ValueError, "No voltage-stability assessments"):
            AnalysisPipeline._stability_criteria_from(self.analysis)


if __name__ == "__main__":
    unittest.main()
