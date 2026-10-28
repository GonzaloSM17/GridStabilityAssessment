"""Compatibility of remote optional spatial runs with local extra studies."""
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
import csv
import io
import json
import runpy
import unittest

from test_spatial_clustering import sample
from stability.dynamic_stability import assess_dynamic_stability
from montecarlo.spatial_stability.models import SpatialCaseAssessment, SpatialSkippedRun
from montecarlo.spatial_stability.distributions import (
    case_features, population_ecdfs, load_saved_analysis, _csv,
)
from montecarlo.spatial_stability.clustering import build_case_profiles
from montecarlo.spatial_stability.characterization import build_inputs
from montecarlo.spatial_stability.technology_analysis import export_technology_analysis


def with_unmapped():
    analysis=sample(20)
    omitted=SpatialCaseAssessment("run_unmapped",
        assess_dynamic_stability(99,"Unstable","Damped","Stable"))
    return replace(analysis,cases=(*analysis.cases,omitted),
        skipped_runs=(SpatialSkippedRun("run_unmapped","parameters.csv unavailable"),))


class RemoteIntegrationTests(unittest.TestCase):
    def test_unmapped_case_is_audited_not_zero_filled_or_added_to_population_mass(self):
        analysis=with_unmapped()
        rows=case_features(analysis)
        self.assertEqual(len(rows),20)
        self.assertNotIn("run_unmapped",{r["case_id"] for r in rows})
        self.assertEqual(len(analysis.cases),21)
        curves=population_ecdfs(analysis)
        self.assertTrue(all(r["n_cases"]==10 for r in curves))
        profiles=build_case_profiles(analysis)
        self.assertEqual(len(profiles.case_ids),20)
        self.assertTrue(any(r["case_id"]=="run_unmapped" and "parameters.csv" in r["reason"]
                            for r in profiles.excluded))
        ids,y,matrices,excluded=build_inputs(analysis)
        self.assertEqual(len(ids),20)
        self.assertEqual(matrices["mva"].shape,(20,11))
        self.assertTrue(any(r["case_id"]=="run_unmapped" for r in excluded))

    def test_saved_skipped_rows_preserve_native_labels_and_reload_for_extra_studies(self):
        analysis=with_unmapped()
        with TemporaryDirectory() as directory:
            out=Path(directory)
            points=[asdict(p) for p in analysis.points]
            _csv(out/"device_case_parameters.csv",points,list(points[0]))
            rows=[dict(case_id=c.case_id,**asdict(c.assessment),diagnostics="",
                       spatial_status="Skipped" if c.case_id=="run_unmapped" else "Mapped",
                       spatial_reason="parameters.csv unavailable" if c.case_id=="run_unmapped" else "")
                  for c in analysis.cases]
            _csv(out/"case_assessments.csv",rows,list(rows[0]))
            loaded=load_saved_analysis(out)
            self.assertEqual([c.assessment.status for c in loaded.cases],
                             [c.assessment.status for c in analysis.cases])
            self.assertEqual(loaded.skipped_runs[0].case_id,"run_unmapped")
            self.assertEqual(len(case_features(loaded)),20)
            # A missing controller set is only accepted when the saved export marks it skipped.
            rows[-1]["spatial_status"]="Mapped"
            _csv(out/"case_assessments.csv",rows,list(rows[0]))
            with self.assertRaisesRegex(ValueError,"Skipped"):
                load_saved_analysis(out)

    def test_physical_grid_minimum_counts_only_cases_with_controller_values(self):
        analysis=with_unmapped()
        kept=analysis.cases[0].case_id
        cases=analysis.cases[:1]+tuple(replace(analysis.cases[-1],case_id=f"missing_{i}")
                                     for i in range(5))
        analysis=replace(analysis,cases=cases,
                         points=tuple(p for p in analysis.points if p.case_id==kept))
        with TemporaryDirectory() as directory:
            paths=export_technology_analysis(analysis,Path(directory),
                                             dict(included_case_ids=[kept]))
            self.assertEqual(paths,())
            metadata=json.loads((Path(directory)/"technology_analysis_metadata.json").read_text())
            self.assertEqual(metadata["evaluated_cases"],1)

    def test_playground_reuses_remote_spatial_result_and_skips_unavailable_extra_inputs(self):
        playground=Path(__file__).parent/"playground.py"
        for analysis in (None,SimpleNamespace(points=()),SimpleNamespace(points=(object(),))):
            spatial=SimpleNamespace(status="Skipped" if not getattr(analysis,"points",()) else "Completed",
                analysis=analysis,figure_paths=(),output_directory=Path("unused"))
            class FakeWorkbench:
                def __init__(self,*args,**kwargs):
                    self.spatial_stability_result=None
                def select_output_directory(self,*args):
                    return Path("unused")
                def analyze(self,*args,**kwargs):
                    self.spatial_stability_result=spatial
                def create_spatial_stability(self,*args,**kwargs):
                    raise AssertionError("Spatial analysis must not run twice")
            with patch("montecarlo.workbench.MonteCarloWorkbench",FakeWorkbench), \
                 patch("montecarlo.spatial_stability.characterization.export_characterization",return_value=()) as extra, \
                 redirect_stdout(io.StringIO()):
                runpy.run_path(str(playground),run_name="__main__")
            if analysis is not None and analysis.points:
                extra.assert_called_once_with(analysis,Path("unused"),repeats=5)
            else:
                extra.assert_not_called()


if __name__=="__main__":
    unittest.main()
