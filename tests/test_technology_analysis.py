"""Check technology weighting, PCA coefficient shares and physical risk counts."""
from dataclasses import replace
import unittest
import numpy as np

from test_spatial_clustering import sample
from montecarlo.spatial_stability.technology_analysis import (
    technology_composition, physical_case_features,
    quantile_edges, instability_grid,
)


class TechnologyAnalysisTests(unittest.TestCase):
    def test_composition_counts_distinct_devices_and_equal_case_fractions(self):
        analysis = sample()
        analysis = replace(analysis, points=tuple(
            replace(p, nominal_power_mva=p.nominal_power_mva * 10)
            if p.generator_name.startswith("BESS_") else p for p in analysis.points))
        rows = {r["technology"]: r for r in technology_composition(
            analysis, [c.case_id for c in analysis.cases])}
        self.assertEqual(rows["BESS"]["unique_device_ids"], 2)
        self.assertEqual(rows["BESS"]["unit_case_observations"], 40)
        self.assertAlmostEqual(rows["BESS"]["unit_share"], .25)
        self.assertAlmostEqual(rows["BESS"]["mva_share"], 40/52)
        self.assertAlmostEqual(sum(r["unit_share"] for r in rows.values()), 1)
        self.assertAlmostEqual(sum(r["mva_share"] for r in rows.values()), 1)

    def test_only_requested_cases_define_composition(self):
        analysis = sample()
        selected = analysis.cases[0].case_id
        rows = technology_composition(analysis, [selected])
        self.assertEqual(sum(r["unit_case_observations"] for r in rows), 8)
        self.assertTrue(all(r["n_cases"] == 1 for r in rows))
        with self.assertRaises(ValueError):
            technology_composition(analysis, ["missing"])

    def test_low_xi_threshold_and_fn_weighting_use_same_basis(self):
        analysis = sample(4)
        first = analysis.cases[0].case_id
        points = tuple(replace(p, xi=.6 if p.device_id == "BESS_0" else .9,
                               fn_hz=2. if p.device_id == "BESS_0" else 10.)
                       if p.case_id == first else p for p in analysis.points)
        rows = physical_case_features(replace(analysis, points=points))
        rows = {r["basis"]: r for r in rows if r["case_id"] == first and r["scope"] == "BESS"}
        self.assertAlmostEqual(rows["units"]["low_xi_fraction"], .5)
        self.assertAlmostEqual(rows["mva"]["low_xi_fraction"], .25)
        self.assertAlmostEqual(rows["units"]["mean_fn_hz"], 6.)
        self.assertAlmostEqual(rows["mva"]["mean_fn_hz"], 8.)

    def test_unknown_labels_empty_cells_and_last_edge_are_handled(self):
        rows = [
            dict(low_xi_fraction=.2, mean_fn_hz=5., status=status)
            for status in ("Stable", "Unstable", "Not evaluated")]
        rows.append(dict(low_xi_fraction=.5, mean_fn_hz=20., status="Stable"))
        cells = instability_grid(rows, np.array([.1, .3, .5]), np.array([1., 10., 20.]))
        lookup = {(r["x_bin"], r["y_bin"]): r for r in cells}
        first = lookup[0, 0]
        self.assertEqual(first["evaluated"], 2)
        self.assertEqual(first["not_evaluated"], 1)
        self.assertEqual(first["instability_fraction"], .5)
        self.assertTrue(first["low_sample"])
        self.assertLess(first["wilson95_lower"], .5)
        self.assertGreater(first["wilson95_upper"], .5)
        self.assertIsNone(lookup[1, 0]["instability_fraction"])
        self.assertEqual(lookup[1, 1]["evaluated"], 1)
        self.assertEqual(sum(c["evaluated"] + c["not_evaluated"] for c in cells), 4)

    def test_constant_quantiles_form_one_bin_instead_of_fake_intervals(self):
        edges = quantile_edges([.2]*10)
        self.assertEqual(len(edges), 1)
        cells = instability_grid(
            [dict(low_xi_fraction=.2, mean_fn_hz=5., status="Unstable")],
            edges, quantile_edges([5.]*10))
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["instability_fraction"], 1.)


if __name__ == "__main__":
    unittest.main()
