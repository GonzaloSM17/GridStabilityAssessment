"""Focused tests for Monte Carlo analyzer presentation helpers."""

import unittest

from analyzer import AnalysisPipeline, SignalStudyAnalysis, _round_up_to_nice_number
from transfer_analysis import (
    DampingCriteria,
    DampingPair,
    TransferRunAssessment,
    TransferStudyAnalysis,
)


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


class TransferDampingPlotTests(unittest.TestCase):
    def test_negative_damping_values_keep_the_five_percent_axis_limit(self):
        pair = DampingPair(
            polarity="positive",
            first_time_s=1.2,
            second_time_s=2.2,
            first_amplitude=3.0,
            second_amplitude=4.0,
            period_s=1.0,
            damping_ratio=-0.037,
        )
        assessment = TransferRunAssessment(
            run_id=1,
            transfer_name="NC_TEST_C1",
            pre_fault_power=100.0,
            final_power=90.0,
            status="warning_growing_mode",
            dominant_frequency_hz=1.0,
            warning="The selected dominant-mode envelope grows.",
            damping_pairs=(pair,),
            selected_damping_pair=pair,
            criteria=DampingCriteria(),
        )
        figure = object.__new__(AnalysisPipeline).plot_transfer_damping_summary(
            TransferStudyAnalysis((assessment,))
        )
        try:
            self.assertEqual(figure.axes[0].get_xlim()[1], 5.0)
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
