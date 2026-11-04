"""Validate xi-only groups, native stability display and preservation of graph 05."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from test_spatial_clustering import sample
from stability.dynamic_stability import assess_dynamic_stability
from montecarlo.spatial_stability.xi_groups import (
    xi_case_features, fit_xi_groups, export_xi_groups,
)


class XiGroupTests(unittest.TestCase):
    def test_two_groups_order_by_xi_and_report_actual_spread_direction(self):
        rng = np.random.default_rng(7)
        x = np.r_[rng.normal([.7, .12], [.001, .001], size=(20, 2)),
                  rng.normal([.9, .04], [.001, .001], size=(20, 2))]
        result = fit_xi_groups(x, resamples=2)
        self.assertEqual(result["status"], "fitted")
        self.assertEqual(result["labels"], [1]*20 + [2]*20)
        self.assertEqual(result["groups"][0]["xi_relation"], "menor")
        self.assertEqual(result["groups"][0]["spread_relation"], "mayor")
        self.assertEqual(result["groups"][1]["spread_relation"], "menor")
        self.assertGreater(result["silhouette"], .9)
        reversed_spread = x.copy()
        reversed_spread[:, 1] = .2-x[:, 1]
        result = fit_xi_groups(reversed_spread, resamples=2)
        self.assertEqual(result["groups"][0]["spread_relation"], "menor")

    def test_fn_and_stability_cannot_change_group_inputs(self):
        analysis = sample()
        ids = [c.case_id for c in analysis.cases]
        changed = replace(analysis,
            points=tuple(replace(p, fn_hz=100*p.fn_hz) for p in analysis.points),
            cases=tuple(replace(c, assessment=assess_dynamic_stability(
                c.assessment.run_id, "Not evaluated", "Damped", "Stable"))
                        for c in analysis.cases))
        for basis in ("units", "mva"):
            np.testing.assert_array_equal(xi_case_features(analysis, ids, basis),
                                          xi_case_features(changed, ids, basis))

    def test_weighted_mean_and_spread_are_both_computed_from_units(self):
        analysis = sample()
        first = analysis.cases[0].case_id
        analysis = replace(analysis, points=tuple(
            replace(p, xi=.6 if p.device_id == "BESS_0" else .9)
            if p.case_id == first else p for p in analysis.points))
        units = xi_case_features(analysis, [first], "units")[0]
        mva = xi_case_features(analysis, [first], "mva")[0]
        self.assertAlmostEqual(units[0], (.6+7*.9)/8)
        self.assertAlmostEqual(mva[0], (.6+15*.9)/16)
        self.assertAlmostEqual(units[1], np.std([.6]+[.9]*7))
        self.assertNotAlmostEqual(units[1], mva[1])

    def test_degenerate_data_does_not_invent_two_groups(self):
        for x in (np.zeros((8, 2)), np.zeros((1, 2))):
            self.assertEqual(fit_xi_groups(x, resamples=2)["status"], "not_fitted")
        with self.assertRaises(ValueError):
            fit_xi_groups(np.zeros((8, 3)))
        with self.assertRaises(ValueError):
            fit_xi_groups(np.full((8, 2), np.nan))

    def test_export_only_touches_new_outputs_and_cleans_stale_09(self):
        analysis = sample()
        ids = [c.case_id for c in analysis.cases]
        with TemporaryDirectory() as directory:
            out = Path(directory)
            preserved = out/"05_pca_groups_and_stability.png"
            preserved.write_bytes(b"existing graph must be preserved")
            paths = export_xi_groups(analysis, out, ids, resamples=2)
            self.assertEqual([p.name for p in paths], ["09_xi_groups_weighted_pca.png"])
            self.assertTrue(paths[0].is_file())
            self.assertEqual(preserved.read_bytes(), b"existing graph must be preserved")
            metadata = json.loads((out/"xi_groups_metadata.json").read_text())
            self.assertEqual(metadata["clustering"]["mva"]["k"], 2)
            self.assertEqual(metadata["case_ids"], ids)
            export_xi_groups(analysis, out, ids[:1], resamples=2)
            self.assertFalse((out/"09_xi_groups_weighted_pca.png").exists())
            self.assertTrue(preserved.is_file())


if __name__ == "__main__":
    unittest.main()
