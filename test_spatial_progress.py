"""Long spatial studies retain useful summaries and report bounded progress."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from montecarlo.spatial_stability.analysis import SpatialStabilityAnalyzer
from montecarlo.spatial_stability.plotting import export_maps
from stability.scenario import DEFAULT_CONTINGENCY_SCENARIO, EvaluationProfile
from test_spatial_stability import make_assets, write_parameters, components


class SpatialProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = make_assets(self.root / "assets.xlsx")
        self.profile = EvaluationProfile.for_scenario(DEFAULT_CONTINGENCY_SCENARIO)
        self.runs = tuple(write_parameters(self.root / f"run_{i:06d}") for i in range(1, 4))
        self.components = {
            run.display_name: components(angular_stable=(i != 1))
            for i, run in enumerate(self.runs)
        }
        self.analyzer = SpatialStabilityAnalyzer(self.assets, self.profile)

    def test_long_analysis_reports_totals_without_per_run_identifiers(self):
        messages = []
        with patch("montecarlo.spatial_stability.analysis.monotonic",
                   side_effect=[0., 10., 61., 62.]):
            analysis = self.analyzer.analyze(
                self.runs, component_analyses=self.components, progress=messages.append)
        self.assertEqual([c.assessment.status for c in analysis.cases],
                         ["Stable", "Unstable", "Stable"])
        self.assertEqual(len(messages), 3)
        self.assertIn("evaluated 2/3 runs", messages[1])
        self.assertIn("2 stable, 1 unstable, 0 not evaluated", messages[-1])
        self.assertFalse(any(run.display_name in msg for msg in messages for run in self.runs))

    def test_quick_analysis_only_reports_stage_start_and_completion(self):
        messages = []
        with patch("montecarlo.spatial_stability.analysis.monotonic",
                   side_effect=[0., 1., 2., 3.]):
            self.analyzer.analyze(
                self.runs, component_analyses=self.components, progress=messages.append)
        self.assertEqual(len(messages), 2)

    def test_distributions_and_class_means_survive_interrupted_individual_maps(self):
        import matplotlib.pyplot as plt
        from montecarlo.spatial_stability import plotting
        analysis = self.analyzer.analyze(self.runs, component_analyses=self.components)
        out = self.root / "spatial"
        original_draw = plotting._draw
        def interrupt_individual(points, universe, metric, title, *args, **kwargs):
            if title.startswith("Run "):
                raise RuntimeError("Interrupted during a long individual-map export")
            return original_draw(points, universe, metric, title, *args, **kwargs)
        messages = []
        with patch.object(plotting, "_draw", side_effect=interrupt_individual), \
             patch("montecarlo.spatial_stability.clustering.export_clustering", return_value=()):
            with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                export_maps(analysis, out, self.assets, self.profile, self.runs,
                            progress=messages.append)
        self.assertEqual(len(list((out / "distributions_by_stability").glob("*.png"))), 5)
        for metric in ("xi", "fn"):
            for status in ("stable", "unstable"):
                self.assertTrue((out / f"{metric}_maps/by_stability/{status}/{metric}_mean_map.png").is_file())
        with (out / "case_assessments.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual([row["status"] for row in csv.DictReader(stream)],
                             ["Stable", "Unstable", "Stable"])
        self.assertTrue(any("summary figures are already saved" in msg for msg in messages))
        self.assertFalse(plt.get_fignums())


if __name__ == "__main__":
    unittest.main()
