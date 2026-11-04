"""Unevaluated cases remain auditable but never appear as plotted observations."""
from dataclasses import replace
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np

from stability.dynamic_stability import assess_monte_carlo_stability
from stability.scenario import DEFAULT_CONTINGENCY_SCENARIO, EvaluationProfile
from montecarlo.spatial_stability.models import SpatialCaseAssessment
from montecarlo.spatial_stability.distributions import export_distributions
from montecarlo.spatial_stability.plotting import export_maps
from montecarlo.spatial_stability.clustering import build_case_profiles, fit_profiles, _export_figures
from test_spatial_distributions import sample
from test_spatial_clustering import sample as cluster_sample


def with_unknown():
    analysis=sample()
    unknown=replace(analysis.points[0],case_id="unknown",run_id=3,xi=100.,fn_hz=1000.)
    cases=(
        SpatialCaseAssessment("a",assess_monte_carlo_stability(1,"Stable","Damped under")),
        SpatialCaseAssessment("b",assess_monte_carlo_stability(2,"Stable","Growing")),
        SpatialCaseAssessment("unknown",assess_monte_carlo_stability(3,"Not evaluated","Damped")),
    )
    return replace(analysis,points=(*analysis.points,unknown),cases=cases)


class UnevaluatedPlotTests(unittest.TestCase):
    def test_distribution_figures_hide_unknown_values_and_labels_but_keep_audit_rows(self):
        captured={}
        original=Figure.savefig
        def inspect(fig,path,*args,**kwargs):
            texts=[axis.get_title()+" ".join(t.get_text() for t in axis.get_xticklabels())+
                   " ".join(axis.get_legend_handles_labels()[1]) for axis in fig.axes]
            captured[Path(path).name]=texts
            for text in texts:
                self.assertNotIn("No evaluado",text)
                self.assertNotIn("no evaluado",text)
            if Path(path).name=="05_case_weighted_means.png":
                for axis in fig.axes:
                    self.assertEqual(sum(len(c.get_offsets()) for c in axis.collections),2)
            return original(fig,path,*args,**kwargs)
        with TemporaryDirectory() as directory:
            with patch.object(Figure,"savefig",autospec=True,side_effect=inspect):
                paths=export_distributions(with_unknown(),directory)
            self.assertEqual(len(captured),5)
            self.assertEqual(len(paths),5)
            folder=Path(directory)/"distributions_by_stability"
            with (folder/"case_distribution_features.csv").open(encoding="utf-8-sig",newline="") as f:
                self.assertIn("unknown",{row["case_id"] for row in csv.DictReader(f)})
            with (folder/"case_distribution_summary.csv").open(encoding="utf-8-sig",newline="") as f:
                self.assertIn("Not evaluated",{row["status"] for row in csv.DictReader(f)})
            meta=json.loads((folder/"distribution_metadata.json").read_text())
            self.assertEqual(meta["counts"]["Not evaluated"],1)
            self.assertEqual(meta["hidden_not_evaluated"],1)

    def test_pca_points_and_cluster_bars_contain_only_evaluated_cases(self):
        analysis=cluster_sample(32)
        first=analysis.cases[0]
        unknown=replace(first,assessment=assess_monte_carlo_stability(
            first.assessment.run_id,"Not evaluated","Damped"))
        analysis=replace(analysis,cases=(unknown,*analysis.cases[1:]))
        profiles=build_case_profiles(analysis)
        self.assertEqual(len(profiles.case_ids),32)
        result=fit_profiles(profiles.values,max_clusters=3,resamples=2)
        cases={case.case_id:case for case in analysis.cases}
        captured=[]
        original=Figure.savefig
        def inspect(fig,path,*args,**kwargs):
            name=Path(path).name
            if name in {"04_pca_projection.png","05_pca_groups_and_stability.png"}:
                for axis in fig.axes:
                    self.assertEqual(sum(len(c.get_offsets()) for c in axis.collections),31)
                    self.assertNotIn("No evaluado"," ".join(axis.get_legend_handles_labels()[1]))
                captured.append(name)
            if name=="03_cluster_stability.png":
                for axis in fig.axes:
                    self.assertEqual(sum(rect.get_height() for rect in axis.patches),31)
                    self.assertNotIn("no evaluados",axis.get_title())
            return original(fig,path,*args,**kwargs)
        with TemporaryDirectory() as directory:
            with patch.object(Figure,"savefig",autospec=True,side_effect=inspect):
                _export_figures(Path(directory),profiles,result,cases)
        self.assertEqual(len(captured),2)

    def test_stale_unknown_maps_are_removed_and_case_assessments_retained(self):
        analysis=with_unknown()
        profile=EvaluationProfile.for_scenario(DEFAULT_CONTINGENCY_SCENARIO)
        with TemporaryDirectory() as directory:
            out=Path(directory)
            stale=[]
            for metric in ("xi","fn"):
                for relative in (f"{metric}_maps/by_case/{metric}_map_unknown.png",
                                 f"{metric}_maps/by_stability/not_evaluated/{metric}_mean_map.png"):
                    path=out/relative
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(b"stale unknown map")
                    stale.append(path)
            with patch("montecarlo.spatial_stability.clustering.export_clustering",return_value=()):
                result=export_maps(analysis,out,NS(workbook_path="assets.xlsx"),profile,())
            self.assertTrue(all(not path.exists() for path in stale))
            self.assertTrue(all("unknown" not in path.name and "not_evaluated" not in path.parts
                                for path in result.figure_paths))
            with (out/"case_assessments.csv").open(encoding="utf-8-sig",newline="") as f:
                rows={r["case_id"]:r for r in csv.DictReader(f)}
            self.assertEqual(rows["unknown"]["status"],"Not evaluated")
            metadata=json.loads((out/"spatial_metadata.json").read_text())
            self.assertEqual(metadata["hidden_not_evaluated"],["unknown"])
            self.assertLess(metadata["xi_color_limits"][1],1.)


if __name__=="__main__":
    unittest.main()
