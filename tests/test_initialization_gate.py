"""Tests for the shared strict initialization gate."""

from types import SimpleNamespace
import unittest

from montecarlo.batch_analysis import (
    AnalysisPipeline,
    BusStudyAnalysis,
    RunSignalAnalysis,
    SignalStudyAnalysis,
    StudyAnalysis,
)
from stability.initialization import InitializationAssessment, InitializationCriteria
from stability.power_transfer import TransferStudyAnalysis


def _initialization(*, is_well_initialized: bool) -> InitializationAssessment:
    return InitializationAssessment(
        reference_value=1.0,
        fast_relative_error_percentile_99=0.0,
        fast_rms_relative_error=0.0,
        is_well_initialized=is_well_initialized,
        rejection_reasons=() if is_well_initialized else ("fast_oscillation",),
        criteria=InitializationCriteria(),
    )


class InitializationGateTests(unittest.TestCase):
    def setUp(self):
        passing_stability = SimpleNamespace(is_stable=True, failed_criteria=())
        analysis = SignalStudyAnalysis(
            signal_name="KIMAL_220/MTR/u_rms",
            run_results=(
                RunSignalAnalysis(1, _initialization(is_well_initialized=True), passing_stability),
                # This mimics an explicitly requested diagnostic evaluation: its
                # local value exists, but it remains invalid for formal results.
                RunSignalAnalysis(2, _initialization(is_well_initialized=False), passing_stability),
            ),
            skipped_run_ids=(),
        )
        self.study = StudyAnalysis((BusStudyAnalysis("KIMAL_220", analysis),))

    def test_only_well_initialized_runs_are_downstream_eligible(self):
        self.assertEqual(self.study.initialization_eligible_run_ids, (1,))
        self.assertEqual(self.study.initialization_blocked_run_ids, (2,))

    def test_diagnostic_voltage_value_does_not_overrule_initialization_gate(self):
        self.assertEqual(self.study.status_for_run(1), "Stable")
        self.assertEqual(self.study.status_for_run(2), "Not evaluated")

    def test_global_conclusion_is_not_evaluated_for_blocked_run(self):
        result = AnalysisPipeline.analyze_contingency_dynamic_stability(
            self.study,
            TransferStudyAnalysis(()),
        )
        blocked = result.assessment_for(2)
        self.assertEqual(blocked.grid_code_status, "Not evaluated")
        self.assertEqual(blocked.status, "Inconclusive")


if __name__ == "__main__":
    unittest.main()
