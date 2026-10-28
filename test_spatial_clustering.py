"""Validate clustering inputs, label independence, diagnostics and saved exports."""
from dataclasses import replace
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from stability.dynamic_stability import assess_dynamic_stability
from montecarlo.spatial_stability.models import (
    SpatialControllerPoint, SpatialCaseAssessment, SpatialStabilityAnalysis,
)
from montecarlo.spatial_stability.clustering import (
    FEATURES, build_case_profiles, fit_profiles, export_clustering,
)


def sample(n=20):
    points, cases = [], []
    rng = np.random.default_rng(17)
    for i in range(n):
        identifier = f"run_{i:06d}"
        group = i >= n // 2
        for technology in ("BESS", "PE", "PFV", "PMGD"):
            for j, power in enumerate((1., 3.)):
                xi = (.9 if group else .6) + rng.normal(0, .005)
                fn = (15. if group else 3.) + rng.normal(0, .05)
                name = f"{technology}_{j}"
                ki = (2 * np.pi * fn) ** 2
                points.append(SpatialControllerPoint(
                    identifier, i, name, name, name, 0., 0., power,
                    2 * xi * np.sqrt(ki), ki, xi, fn))
        assessment = assess_dynamic_stability(
            i, "Stable", "Damped", "Unstable" if group else "Stable")
        cases.append(SpatialCaseAssessment(identifier, assessment))
    return SpatialStabilityAnalysis(tuple(points), tuple(cases))


class SpatialClusteringTests(unittest.TestCase):
    def test_ten_features_are_weighted_and_spreads_use_all_devices(self):
        analysis = sample()
        profiles = build_case_profiles(analysis)
        self.assertEqual(profiles.values.shape, (20, 10))
        self.assertEqual(len(FEATURES), 10)
        selected = [p for p in analysis.points if p.case_id == profiles.case_ids[0]]
        for index, technology in enumerate(("BESS", "PE", "PFV", "PMGD")):
            points = [p for p in selected if p.generator_name.startswith(technology + "_")]
            self.assertAlmostEqual(profiles.values[0, index],
                                   sum(p.xi * p.nominal_power_mva for p in points) / 4)
            self.assertAlmostEqual(profiles.values[0, index + 4],
                                   sum(p.fn_hz * p.nominal_power_mva for p in points) / 4)
        for index, metric in ((8, "xi"), (9, "fn_hz")):
            values = np.array([getattr(p, metric) for p in selected])
            weights = np.array([p.nominal_power_mva for p in selected])
            mean = np.average(values, weights=weights)
            self.assertAlmostEqual(profiles.values[0, index],
                                   np.sqrt(np.average((values - mean) ** 2, weights=weights)))

    def test_missing_technology_excludes_case_without_zero_imputation(self):
        analysis = sample()
        analysis = replace(analysis, points=tuple(
            p for p in analysis.points if not (
                p.case_id == "run_000000" and p.generator_name.startswith("PMGD_"))))
        profiles = build_case_profiles(analysis)
        self.assertEqual(len(profiles.case_ids), 19)
        self.assertEqual(profiles.excluded[0]["case_id"], "run_000000")
        self.assertIn("PMGD", profiles.excluded[0]["reason"])

    def test_stability_and_case_order_do_not_change_features(self):
        analysis = sample()
        changed = replace(analysis, points=tuple(reversed(analysis.points)),
                          cases=tuple(replace(c, assessment=assess_dynamic_stability(
                              c.assessment.run_id, "Not evaluated", "Damped", "Stable"))
                                      for c in reversed(analysis.cases)))
        a, b = build_case_profiles(analysis), build_case_profiles(changed)
        self.assertEqual(a.case_ids, b.case_ids)
        np.testing.assert_allclose(a.values, b.values)

    def test_separated_profiles_recovered_and_original_distance_used_for_pca(self):
        profiles = build_case_profiles(sample())
        fitted = fit_profiles(profiles.values, max_clusters=3, resamples=3)
        expected = np.arange(20) >= 10
        self.assertEqual(fitted["status"], "fitted")
        for info in fitted["methods"].values():
            self.assertEqual(info["k"], 2)
            self.assertAlmostEqual(adjusted_rand_score(expected, info["labels"]), 1.)
            self.assertEqual(info["conclusion"], "passes_descriptive_screen")
            self.assertAlmostEqual(info["silhouette_original"],
                silhouette_score(StandardScaler().fit_transform(profiles.values), info["labels"]))
        self.assertGreaterEqual(fitted["pca"]["retained_variance"], .9)

    def test_noise_is_reported_as_weak_and_repeatable(self):
        values = np.random.default_rng(21).normal(size=(80, 10))
        a = fit_profiles(values, max_clusters=3, resamples=3)
        b = fit_profiles(values, max_clusters=3, resamples=3)
        for method in a["methods"]:
            self.assertEqual(a["methods"][method], b["methods"][method])
            self.assertEqual(a["methods"][method]["conclusion"], "no_well_defined_groups")

    def test_small_constant_and_invalid_inputs_do_not_create_fake_clusters(self):
        for values in (np.zeros((1, 10)), np.ones((8, 10)), np.empty((0, 10))):
            result = fit_profiles(values, resamples=2)
            self.assertEqual(result["status"], "not_fitted")
            self.assertFalse(result["methods"])
        with self.assertRaisesRegex(ValueError, "ten"):
            fit_profiles(np.ones((8, 9)))
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_profiles(np.full((8, 10), np.nan))
        with self.assertRaises(ValueError):
            fit_profiles(np.ones((8, 10)), pca_variance=1.)

    def test_export_preserves_native_angular_labels_and_closes_figures(self):
        analysis = sample()
        first = analysis.cases[0]
        analysis = replace(analysis, cases=(replace(first, assessment=assess_dynamic_stability(
            first.assessment.run_id, "Not evaluated", "Damped", "Stable")), *analysis.cases[1:]))
        before = plt.get_fignums()
        with TemporaryDirectory() as directory:
            paths = export_clustering(analysis, directory, max_clusters=3, resamples=3)
            self.assertEqual({p.name for p in paths}, {
                "01_model_selection.png", "03_cluster_stability.png",
                "04_pca_projection.png", "05_pca_groups_and_stability.png",
                "06_weighted_pca_projection.png", "07_physical_instability.png",
                "08_technology_instability.png", "09_xi_groups_weighted_pca.png"})
            self.assertTrue(all(p.is_file() and p.suffix == ".png" for p in paths))
            out = Path(directory) / "clustering"
            with (out / "case_clusters.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["status"], "Not evaluated")
            self.assertEqual(rows[-1]["voltage_status"], "Stable")
            self.assertEqual(rows[-1]["angular_status"], "Unstable")
            self.assertEqual(rows[-1]["status"], "Unstable")
            metadata = json.loads((out / "clustering_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["feature_names"], list(FEATURES))
            self.assertFalse(list(out.glob("*.svg")))
            self.assertFalse((out / "02_cluster_profiles.png").exists())
            self.assertFalse((out / "06_pca_technology_shares.png").exists())
            self.assertTrue((out / "weighted_pca_coordinates.csv").is_file())
            description = (out / "PCA_DESCRIPTION.md").read_text(encoding="utf-8")
            self.assertIn("Coeficiente PC1", description)
            self.assertIn("xi medio BESS", description)
            # Simulate the old figure before regenerating an ineligible analysis.
            (out / "02_cluster_profiles.png").write_bytes(b"legacy generated plot")
            # Reusing a folder for an ineligible single run cannot leave stale clusters.
            single = replace(analysis, cases=analysis.cases[:1],
                             points=tuple(p for p in analysis.points if p.case_id == first.case_id))
            self.assertEqual(export_clustering(single, directory), ())
            metadata = json.loads((out / "clustering_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "not_fitted")
            self.assertFalse(list(out.glob("*.png")))
            self.assertFalse((out / "case_clusters.csv").exists())
        self.assertEqual(plt.get_fignums(), before)


if __name__ == "__main__":
    unittest.main()
