"""Global tree features and two-panel distributions retain native Monte Carlo labels."""
from dataclasses import replace
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from stability.dynamic_stability import assess_monte_carlo_stability
from montecarlo.spatial_stability.models import SpatialCaseAssessment, SpatialStabilityAnalysis
from montecarlo.spatial_stability.tree_profiles import (
    GLOBAL_FEATURES, build_global_tree_inputs, export_global_descriptive_trees,
)
from montecarlo.spatial_stability.distributions import export_case_weighted_means
from montecarlo.spatial_stability.characterization import export_characterization
from test_spatial_distributions import point, sample


def dispersion_sample():
    points, cases = [], []
    for i in range(33):
        identifier = f"run_{i:06d}"
        half_range = 1. if i < 16 else 7.
        points.extend(replace(point(identifier, f"BESS_{j}", 1., .7, value), run_id=i)
                      for j, value in enumerate((10-half_range, 10+half_range)))
        voltage = "Not evaluated" if i == 32 else "Stable"
        damping = "Damped under" if i < 16 else "Growing"
        if i == 32:
            damping = "Damped"
        cases.append(SpatialCaseAssessment(identifier,
            assess_monte_carlo_stability(i, voltage, damping)))
    return SpatialStabilityAnalysis(tuple(points),tuple(cases))


class GlobalPhysicalFigureTests(unittest.TestCase):
    def test_means_and_population_dispersion_follow_mva_not_technology(self):
        analysis = sample()
        ids,y,values,excluded = build_global_tree_inputs(analysis)
        self.assertEqual(ids,("a","b"))
        np.testing.assert_allclose(values["mva"][0],
            [.75,12.5,np.sqrt(.0075),np.sqrt(18.75)])
        np.testing.assert_allclose(values["units"][0],[.7,10.,.1,5.])
        renamed = replace(analysis,points=tuple(
            replace(p,generator_name="ONE_TECH_"+str(i)) for i,p in enumerate(analysis.points)))
        ids2,y2,values2,excluded2 = build_global_tree_inputs(renamed)
        self.assertEqual(ids2,ids)
        for basis in values:
            np.testing.assert_allclose(values[basis],values2[basis])
        self.assertFalse(excluded)
        self.assertFalse(excluded2)

    def test_global_tree_can_split_on_fn_dispersion_without_technology_indicators(self):
        analysis = dispersion_sample()
        with TemporaryDirectory() as directory:
            out = Path(directory)
            paths = export_global_descriptive_trees(analysis,out)
            self.assertEqual(len(paths),3)
            self.assertTrue(all(path.is_file() for path in paths))
            rules = (out/"descriptive_tree_rules.txt").read_text(encoding="utf-8")
            self.assertIn("Disp. fn [Hz] <=",rules)
            self.assertNotIn("BESS",rules)
            meta = json.loads((out/"tree_profile_metadata.json").read_text())
            self.assertEqual(meta["features"],list(GLOBAL_FEATURES))
            self.assertFalse(meta["include_angle"])
            self.assertEqual(meta["counts"],{"Stable":16,"Unstable":16})
            self.assertEqual(meta["excluded_cases"],
                [{"case_id":"run_000032","reason":"Native stability not evaluated"}])
            with (out/"tree_leaf_profiles.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["feature"] for row in rows},set(GLOBAL_FEATURES))
            self.assertTrue(all(row["n_stable"]=="0" or row["n_unstable"]=="0" for row in rows))
            self.assertNotIn("\\n", (out/"TREE_PROFILES.md").read_text(encoding="utf-8"))

    def test_global_tree_does_not_depend_on_fitting_technology_models(self):
        with TemporaryDirectory() as directory:
            paths = export_characterization(dispersion_sample(),directory,
                families=("logistic",),repeats=2,outer_folds=2,inner_folds=2)
            self.assertEqual(len(paths),3)
            out = Path(directory)/"stability_characterization"
            meta = json.loads((out/"characterization_metadata.json").read_text())
            self.assertEqual(meta["status"],"not_fitted")
            self.assertEqual(meta["tree_n_cases"],32)
            self.assertEqual(len(meta["figures"]),3)

    def test_two_panels_use_the_same_weighted_case_means_and_keep_under_damped_stable(self):
        analysis = sample()
        analysis = replace(analysis,cases=(
            SpatialCaseAssessment("a",assess_monte_carlo_stability(1,"Stable","Damped under")),
            SpatialCaseAssessment("b",assess_monte_carlo_stability(2,"Stable","Growing"))))
        captured = {}
        original_save = Figure.savefig
        def inspect(figure,*args,**kwargs):
            captured["axes"] = len(figure.axes)
            captured["ys"] = [
                sorted(float(y) for collection in axis.collections
                       for _,y in collection.get_offsets())
                for axis in figure.axes]
            captured["titles"] = [label.get_text() for label in figure.axes[0].get_xticklabels()]
            captured["suptitle"] = figure._suptitle.get_text()
            return original_save(figure,*args,**kwargs)
        with TemporaryDirectory() as directory:
            folder=Path(directory)/"distributions_by_stability"
            folder.mkdir()
            preserved=folder/"01_case_parameter_distributions.png"
            preserved.write_bytes(b"preserve original complete figure")
            with patch.object(Figure,"savefig",autospec=True,side_effect=inspect):
                path=export_case_weighted_means(analysis,directory)
            self.assertTrue(path.is_file())
            self.assertEqual(captured["axes"],2)
            np.testing.assert_allclose(captured["ys"][0],[.6,.75])
            np.testing.assert_allclose(captured["ys"][1],[12.5,17.5])
            self.assertEqual(len(captured["titles"]),2)
            self.assertIn("\n",captured["suptitle"])
            meta=json.loads((folder/"05_case_weighted_means_metadata.json").read_text())
            self.assertEqual(meta["counts"],{"Stable":1,"Unstable":1,"Not evaluated":0})
            self.assertFalse(meta["include_angle"])
            self.assertEqual(preserved.read_bytes(),b"preserve original complete figure")
            self.assertFalse(plt.get_fignums())


if __name__=="__main__":
    unittest.main()
