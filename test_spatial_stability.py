"""Native parameter, classification, selector and PNG-export regression checks."""
from dataclasses import replace
from math import pi, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import patch
import csv
import io
from contextlib import redirect_stdout
import json
import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import polars as pl
from openpyxl import Workbook

from data.assets import Assets
from data.run import MonteCarloRun
from montecarlo.spatial_stability.analysis import (
    SpatialAssetLookup, SpatialStabilityAnalyzer, assess_components, read_controller_points,
)
from montecarlo.spatial_stability.plotting import frequency_band, mean_points, weighted_value
from stability.scenario import DEFAULT_CONTINGENCY_SCENARIO, EvaluationProfile
from single_run.workbench import SingleRunWorkbench


def make_assets(path):
    book = Workbook(); sheet = book.active; sheet.title = "Unit Manager"
    sheet.append(["emtp_name", "App. Power [MVA]", "Inertia [s]", "posX", "posY"])
    for row in [("BESS_A",100,None,10,20),("PFV_LA_HUAYCA_II",200,None,30,40),
                ("HE_A",100,3,50,60),("HE_B",200,4,70,80)]: sheet.append(row)
    book.save(path); book.close()
    return Assets(path)


def write_parameters(folder, rows=None):
    folder.mkdir(parents=True,exist_ok=True)
    if rows is None:
        rows=[dict(GeneratorName="BESS_A",InService=1,Kp=10.,Ki=100.,UnitId=1),
              dict(GeneratorName="PFV_LA_HUAYCA_II",InService=1,Kp=30.,Ki=300.,UnitId=2)]
    pl.DataFrame(rows).write_csv(folder/"parameters.csv",separator=";")
    return MonteCarloRun.from_directory(folder)


def write_parquet(folder, *, angular_unstable=False):
    time=np.arange(0,6.001,.01)
    voltage=np.ones_like(time)
    voltage[(time>=1)&(time<1.12)]=.2
    recovery=(time>=1.12)&(time<1.2)
    voltage[recovery]=.9+1.25*(time[recovery]-1.12)
    power=np.full_like(time,100.)
    power[(time>=1)&(time<1.12)]=40.
    post=time>=1.12
    power[post]=90+8*np.exp(-.8*(time[post]-1.12))*np.cos(2*np.pi*(time[post]-1.12))
    angle_a=np.zeros_like(time)
    angle_a[post]=200. if angular_unstable else 10.
    pl.DataFrame({"time":time,"KIMAL220_MTR/u_rms":voltage,"PQ_NC_KIMAL_CHANGOS_C1/P":power,
                  "HE_A/Teta_1_":angle_a,"HE_B/Teta_1_":np.zeros_like(time)}).write_parquet(folder/"signals.parquet")


def components(angular_stable=True):
    voltage=NS(assessments=(NS(assessment=NS(is_stable=True)),))
    damping=NS(records=(NS(assessment=NS(status="damped",damping_ratio=.1)),))
    angular=NS(machines=(object(),),is_stable=angular_stable)
    return dict(voltage=voltage,damping=damping,angular=angular)


class SpatialStabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.assets=make_assets(self.root/"assets.xlsx")
        self.profile=EvaluationProfile.for_scenario(DEFAULT_CONTINGENCY_SCENARIO)

    def test_native_parameters_match_existing_controller_formulas(self):
        run=write_parameters(self.root/"run_000001")
        points,excluded=read_controller_points(run,SpatialAssetLookup(self.assets))
        native=run.read_active_controller_parameters(self.assets)
        self.assertFalse(excluded)
        for point,controller in zip(points,native):
            self.assertEqual(point.xi,controller.xi)
            self.assertEqual(point.fn_hz,controller.fn_hz)
            self.assertEqual(point.nominal_power_mva,controller.nominal_power_mva)
        self.assertAlmostEqual(weighted_value(points,"fn_hz"),(100*sqrt(100)+200*sqrt(300))/(300*2*pi))

    def test_confirmed_rename_and_controller_instances_share_plant_mva(self):
        rows=[dict(GeneratorName="PFV_POZOALMONTE_HUAYCA",InService=1,Kp=20.,Ki=100.,UnitId=i) for i in (1,2)]
        run=write_parameters(self.root/"run_000001",rows)
        points,excluded=read_controller_points(run,SpatialAssetLookup(self.assets))
        self.assertEqual(len(points),2); self.assertFalse(excluded)
        self.assertEqual(sum(p.nominal_power_mva for p in points),200)
        self.assertEqual({p.asset_name for p in points},{"PFV_LA_HUAYCA_II"})
        self.assertEqual(len({p.device_id for p in points}),2)

    def test_inactive_and_non_pll_rows_are_not_included(self):
        rows=[dict(GeneratorName="BESS_A",InService=1,Kp=10.,Ki=100.),
              dict(GeneratorName="OTHER",InService=0,Kp=10.,Ki=100.),
              dict(GeneratorName="HE_A",InService=1,Kp=None,Ki=None)]
        run=write_parameters(self.root/"run_000001",rows)
        points,excluded=read_controller_points(run,SpatialAssetLookup(self.assets))
        self.assertEqual(len(points),1);self.assertFalse(excluded)

    def test_ambiguous_instances_and_invalid_ki_are_rejected(self):
        rows=[dict(GeneratorName="BESS_A",InService=1,Kp=10.,Ki=100.)]*2
        run=write_parameters(self.root/"run_000001",rows)
        with self.assertRaisesRegex(ValueError,"duplicate controller"):
            read_controller_points(run,SpatialAssetLookup(self.assets))
        run=write_parameters(self.root/"run_000002",[dict(GeneratorName="BESS_A",InService=1,Kp=10.,Ki=0.)])
        with self.assertRaisesRegex(ValueError,"non-positive Ki"):
            read_controller_points(run,SpatialAssetLookup(self.assets))

    def test_unmatched_coordinates_are_reported_not_silently_substituted(self):
        rows=[dict(GeneratorName="BESS_A",InService=1,Kp=10.,Ki=100.),
              dict(GeneratorName="UNKNOWN",InService=1,Kp=10.,Ki=100.)]
        run=write_parameters(self.root/"run_000001",rows)
        points,excluded=read_controller_points(run,SpatialAssetLookup(self.assets))
        self.assertEqual(len(points),1);self.assertEqual(excluded[0]["generator_name"],"UNKNOWN")

    def test_stability_aggregation_matches_workbench_and_includes_angle(self):
        c=components()
        assessment=assess_components(1,**c)
        self.assertEqual(assessment.voltage_status,SingleRunWorkbench._voltage_status(c["voltage"]))
        self.assertEqual(assessment.modal_damping_status,SingleRunWorkbench._damping_status(c["damping"]))
        self.assertEqual(assessment.status,"Stable")
        self.assertEqual(assess_components(1,**components(False)).status,"Unstable")
        self.assertEqual(assess_components(1,c["voltage"],c["damping"],None).status,"Not evaluated")

    def test_missing_signals_do_not_create_stable_labels(self):
        run=write_parameters(self.root/"run_000001")
        result=SpatialStabilityAnalyzer(self.assets,self.profile).analyze([run])
        self.assertEqual(result.cases[0].assessment.status,"Not evaluated")
        self.assertEqual(len(result.cases[0].diagnostics),3)

    def test_mean_fn_is_mean_of_frequencies_not_frequency_of_mean_gain(self):
        run=write_parameters(self.root/"run_000001")
        points,_=read_controller_points(run,SpatialAssetLookup(self.assets))
        a=points[0];b=replace(a,case_id="run_000002",ki=400,fn_hz=20/(2*pi),xi=.25)
        averaged,rows=mean_points([a,b],{a.case_id,b.case_id})
        self.assertAlmostEqual(averaged[0].fn_hz,15/(2*pi))
        self.assertNotAlmostEqual(averaged[0].fn_hz,sqrt(250)/(2*pi))
        self.assertEqual(rows[0]["n_cases"],2)
        self.assertEqual(mean_points([a],set()),((),[]))

    def test_band_boundaries(self):
        self.assertEqual([frequency_band(v) for v in (0,1.99,2,5,10,20,100)],
                         ["0-2 Hz","0-2 Hz","2-5 Hz","5-10 Hz","10-20 Hz",">=20 Hz",">=20 Hz"])

    def test_native_parquet_pipeline_distinguishes_angular_failure(self):
        runs=[]
        for i in (1,2):
            folder=self.root/f"run_{i:06d}"
            write_parameters(folder);write_parquet(folder,angular_unstable=(i==2))
            runs.append(MonteCarloRun.from_directory(folder))
        analysis=SpatialStabilityAnalyzer(self.assets,self.profile).analyze(runs)
        self.assertEqual([c.assessment.status for c in analysis.cases],["Stable","Unstable"])
        self.assertFalse(any(c.diagnostics for c in analysis.cases))

    def test_single_run_workbench_reuses_analyses_and_exports_flat_png_only(self):
        run=write_parameters(self.root/"run_000001")
        workbench=SingleRunWorkbench(run.run_path,assets_path=self.assets.workbook_path)
        c=components()
        with patch("single_run.voltage_stability.SingleRunVoltageStability.analyze",side_effect=AssertionError("Unexpected reanalysis")):
            result=workbench.create_spatial_stability(output_directory=self.root/"out",**c)
        self.assertEqual(len(result.figure_paths),8)
        self.assertTrue(all(p.is_file() and p.suffix==".png" for p in result.figure_paths))
        self.assertFalse(list(result.output_directory.rglob("*.svg")))
        self.assertTrue((result.output_directory/"fn_maps/by_case/fn_map_run_000001.png").is_file())
        self.assertFalse(any(p.is_dir() for p in (result.output_directory/"xi_maps/by_case").iterdir()))
        with (result.output_directory/"case_assessments.csv").open(encoding="utf-8-sig",newline="") as f:
            row=next(csv.DictReader(f))
        self.assertEqual(row["status"],"Stable");self.assertEqual(row["angular_status"],"Stable")
        self.assertEqual(json.loads((result.output_directory/"spatial_metadata.json").read_text())["cases"],1)

    def test_study_selectors_and_maps_use_all_three_native_criteria(self):
        from montecarlo.workbench import MonteCarloWorkbench
        study_path=self.root/"study"
        study_path.mkdir();(study_path/"manifest.json").write_text("{}")
        for i in (1,2):
            folder=study_path/"runs"/f"run_{i:06d}"
            write_parameters(folder);write_parquet(folder,angular_unstable=(i==2))
        with patch.object(MonteCarloWorkbench,"_choose_context_path",return_value=study_path) as context_picker, patch.object(MonteCarloWorkbench,"_choose_assets_path",return_value=self.assets.workbook_path) as assets_picker:
            workbench=MonteCarloWorkbench()
        context_picker.assert_called_once_with();assets_picker.assert_called_once_with()
        result=workbench.create_spatial_stability()
        self.assertEqual(result.output_directory,(study_path/"analysis_outputs/spatial_stability").resolve())
        self.assertEqual(len(result.figure_paths),10)
        self.assertEqual([case.assessment.status for case in result.analysis.cases],["Stable","Unstable"])
        metadata=json.loads((result.output_directory/"spatial_metadata.json").read_text())
        self.assertEqual(metadata["counts"],{"Stable":1,"Unstable":1,"Not evaluated":0})
        self.assertTrue(all(p.is_file() for p in result.figure_paths))

    def test_spatial_analysis_does_not_print_per_run_progress(self):
        run=write_parameters(self.root/"run_000001")
        output=io.StringIO()
        with redirect_stdout(output):
            result=SpatialStabilityAnalyzer(self.assets,self.profile).analyze([run])
        self.assertEqual(output.getvalue(),"")
        self.assertEqual(len(result.cases),1)

    def test_saved_batch_figure_is_closed_after_png_export(self):
        import matplotlib.pyplot as plt
        from montecarlo.workbench import MonteCarloWorkbench
        workbench=object.__new__(MonteCarloWorkbench)
        workbench._show_figures=False
        figure=plt.figure()
        number=figure.number
        path=workbench.save_figure(figure,self.root,"saved.png")
        self.assertTrue(path.is_file())
        self.assertFalse(plt.fignum_exists(number))

    def test_output_picker_is_shared_by_all_analyses(self):
        from montecarlo.workbench import MonteCarloWorkbench
        run=write_parameters(self.root/"run_000001")
        single=SingleRunWorkbench(run.run_path,assets_path=self.assets.workbook_path)
        batch=object.__new__(MonteCarloWorkbench)
        batch._output_directory=None
        batch.study=NS(context=NS(context_path=self.root/"study"))
        destination=self.root/"chosen_output"
        with patch("tkinter.Tk") as window, patch("tkinter.filedialog.askdirectory",return_value=str(destination)) as picker:
            selected=single.select_output_directory()
        picker.assert_called_once_with(title="Select the output directory")
        window.return_value.destroy.assert_called_once_with()
        self.assertEqual(selected,destination.resolve())
        self.assertEqual(single.analysis_output_directory(),selected)
        with patch("tkinter.Tk",side_effect=AssertionError("Unexpected output dialog")):
            self.assertEqual(batch.select_output_directory(destination),selected)
            self.assertEqual(batch.analysis_output_directory(),selected)
            self.assertEqual(batch.controller_output_directory(),selected/"01_controllers")
        self.assertFalse((run.run_path/"analysis_outputs").exists())

    def test_cancelling_output_picker_does_not_generate_output(self):
        from utils.output_directory import select_output_directory
        with patch("tkinter.Tk") as window, patch("tkinter.filedialog.askdirectory",return_value=""):
            with self.assertRaisesRegex(RuntimeError,"cancelled"):
                select_output_directory()
        window.return_value.destroy.assert_called_once_with()

    def test_assets_dialog_uses_the_existing_picker(self):
        run=write_parameters(self.root/"run_000001")
        workbench=SingleRunWorkbench(run.run_path)
        with patch("montecarlo.workbench.MonteCarloWorkbench._choose_assets_path",return_value=self.assets.workbook_path) as picker:
            assets=workbench.open_assets()
        picker.assert_called_once_with()
        self.assertEqual(assets.workbook_path,self.assets.workbook_path)


if __name__=="__main__":unittest.main()
