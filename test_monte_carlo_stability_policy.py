"""Monte Carlo labels are voltage OR Growing, with low positive damping retained."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from stability.dynamic_stability import assess_monte_carlo_stability, assess_dynamic_stability
from stability.scenario import DEFAULT_CONTINGENCY_SCENARIO, EvaluationProfile
from montecarlo.batch_analysis import AnalysisPipeline, ContingencyDynamicStabilityAnalysis
from montecarlo.spatial_stability.analysis import SpatialStabilityAnalyzer, assess_run
from montecarlo.spatial_stability.distributions import load_saved_analysis
from montecarlo.spatial_stability.models import SpatialCaseAssessment
from montecarlo.workbench import MonteCarloWorkbench
from test_spatial_stability import make_assets, write_parameters, write_parquet, components


class MonteCarloStabilityPolicyTests(unittest.TestCase):
    def test_only_voltage_failure_or_growing_is_unstable_including_partial_evidence(self):
        for voltage, damping, expected in (
            ("Stable", "Damped", "Stable"),
            ("Stable", "Damped under", "Stable"),
            ("Stable", "Growing", "Unstable"),
            ("Unstable", "Damped", "Unstable"),
            ("Unstable", "Damped under", "Unstable"),
            ("Unstable", "Not measurable", "Unstable"),
            ("Not evaluated", "Growing", "Unstable"),
            ("Not evaluated", "Damped", "Not evaluated"),
            ("Stable", "Not measurable", "Not evaluated"),
            ("Stable", "Not applicable", "Not evaluated"),
        ):
            with self.subTest(voltage=voltage, damping=damping):
                result = assess_monte_carlo_stability(1, voltage, damping)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.modal_damping_status, damping)
                self.assertIsNone(result.angular_status)

    def test_low_positive_damping_detail_is_kept_and_single_run_policy_is_independent(self):
        result = assess_monte_carlo_stability(1, "Stable", "Damped under")
        self.assertEqual(result.status, "Stable")
        self.assertIn("Damped under", result.reason)
        self.assertEqual(assess_dynamic_stability(1, "Stable", "Damped under", "Stable").status,
                         "Unstable")
        self.assertEqual(assess_dynamic_stability(1, "Stable", "Damped", "Unstable").status,
                         "Unstable")

    def test_batch_pipeline_uses_the_same_monte_carlo_rule(self):
        study = NS(run_ids=(1, 2, 3), status_for_run=lambda run_id: {
            1: "Stable", 2: "Not evaluated", 3: "Unstable"}[run_id])
        transfers = NS(modal_damping_status_for_run=lambda run_id: {
            1: "Damped under", 2: "Growing", 3: "Not measurable"}[run_id])
        result = AnalysisPipeline.analyze_contingency_dynamic_stability(study, transfers)
        self.assertEqual([a.status for a in result.run_assessments],
                         ["Stable", "Unstable", "Unstable"])
        self.assertTrue(all(a.angular_status is None for a in result.run_assessments))

    def test_monte_carlo_assessment_does_not_calculate_or_use_angle(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assets = make_assets(root / "assets.xlsx")
            run = write_parameters(root / "run_000001")
            known = components(angular_stable=False)
            profile = EvaluationProfile.for_scenario(DEFAULT_CONTINGENCY_SCENARIO)
            with patch("single_run.angular_stability.SingleRunAngularStability.analyze",
                       side_effect=AssertionError("Angular analysis must not run")):
                result = assess_run(run, assets, profile, **known, include_angle=False)
            self.assertEqual(result.assessment.status, "Stable")
            self.assertIsNone(result.assessment.angular_status)
            self.assertFalse(result.diagnostics)

    def test_cached_labels_require_complete_matching_monte_carlo_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assets = make_assets(root / "assets.xlsx")
            run = write_parameters(root / "run_000001")
            profile = EvaluationProfile.for_scenario(DEFAULT_CONTINGENCY_SCENARIO)
            analyzer = SpatialStabilityAnalyzer(assets, profile, include_angle=False)
            with self.assertRaisesRegex(ValueError, "Missing cached"):
                analyzer.analyze([run], case_assessments={})
            stale = assess_dynamic_stability(1, "Stable", "Damped under", "Stable")
            with self.assertRaisesRegex(ValueError, "differ from"):
                analyzer.analyze([run], case_assessments={1: stale})

    def test_opening_another_study_invalidates_cached_stability_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assets = make_assets(root / "assets.xlsx")
            paths = []
            for name in ("first", "second"):
                study = root / name
                study.mkdir()
                (study / "manifest.json").write_text("{}")
                write_parameters(study / "runs/run_000001")
                write_parquet(study / "runs/run_000001")
                paths.append(study)
            workbench = MonteCarloWorkbench(paths[0], assets.workbook_path)
            workbench.dynamic_stability_analysis = ContingencyDynamicStabilityAnalysis((
                assess_monte_carlo_stability(1, "Stable", "Damped"),))
            workbench._spatial_voltage_scope = "Selected Monte Carlo buses: OLD_BUS"
            workbench.open_study(paths[1], assets.workbook_path)
            self.assertIsNone(workbench.dynamic_stability_analysis)
            self.assertEqual(workbench._spatial_voltage_scope, "All available 220/500 kV buses")

    def test_workbench_reuses_selected_bus_labels_and_exports_reloadable_groups_without_angle(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            assets = make_assets(root / "assets.xlsx")
            study = root / "study"
            study.mkdir()
            (study / "manifest.json").write_text("{}")
            for i in (1, 2, 3):
                write_parameters(study / "runs" / f"run_{i:06d}")
                write_parquet(study / "runs" / f"run_{i:06d}")
            workbench = MonteCarloWorkbench(study, assets.workbook_path)
            expected = (
                assess_monte_carlo_stability(1, "Stable", "Damped under"),
                assess_monte_carlo_stability(2, "Stable", "Growing"),
                assess_monte_carlo_stability(3, "Not evaluated", "Damped"),
            )
            workbench.dynamic_stability_analysis = ContingencyDynamicStabilityAnalysis(expected)
            workbench._spatial_voltage_scope = "Selected Monte Carlo buses: KIMAL_220"
            with patch("montecarlo.spatial_stability.analysis.assess_run",
                       side_effect=AssertionError("Do not re-evaluate Parquet or angles")):
                result = workbench.create_spatial_stability()
            self.assertEqual([c.assessment for c in result.analysis.cases], list(expected))
            self.assertFalse(result.analysis.include_angle)
            metadata = json.loads((result.output_directory / "spatial_metadata.json").read_text())
            self.assertFalse(metadata["include_angle"])
            self.assertIn("KIMAL_220", metadata["voltage_scope"])
            self.assertEqual(metadata["counts"], {"Stable": 1, "Unstable": 1, "Not evaluated": 1})
            folder = result.output_directory / "xi_maps/by_stability/stable"
            with (folder / "cases.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["case_id"] for r in rows], ["run_000001"])
            self.assertEqual(rows[0]["modal_damping_status"], "Damped under")
            self.assertEqual(rows[0]["angular_status"], "")
            loaded = load_saved_analysis(result.output_directory)
            self.assertFalse(loaded.include_angle)
            self.assertEqual(loaded.voltage_scope, workbench._spatial_voltage_scope)
            self.assertEqual([c.assessment for c in loaded.cases], list(expected))
            # A saved legacy policy is never silently substituted for Monte Carlo.
            metadata["include_angle"] = True
            (result.output_directory / "spatial_metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "angular criterion"):
                load_saved_analysis(result.output_directory)


if __name__ == "__main__":
    unittest.main()
