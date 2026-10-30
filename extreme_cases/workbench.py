"""Orchestrate deterministic extreme-case source validation and analysis."""

from dataclasses import dataclass
from pathlib import Path

from data.assets import Assets
from data.run import MonteCarloRun
from extreme_cases.context import ExtremeCasesContext
from extreme_cases.npy_signals import NamedNpyDataset
from extreme_cases.reports import write_extreme_cases_report
from extreme_cases.signal_index import SignalIndex
from extreme_cases.study import ExtremeCasesStudy
from montecarlo.batch_analysis import (
    AnalysisPipeline,
    BusStudyAnalysis,
    ControllerPoint,
    ControllerStudyAnalysis,
    StudyAnalysis,
    WeightedControllerPoint,
)
from montecarlo.angular_stability import MonteCarloAngularStability
from stability.scenario import (
    ContingencyScenario,
    DEFAULT_CONTINGENCY_SCENARIO,
    EvaluationProfile,
)
from utils.analysis_logging import configure_analysis_logger


@dataclass(frozen=True)
class ExtremeCasesPreparation:
    """Validated source inputs and controller comparison output."""

    output_directory: Path
    signal_index_path: Path
    signal_count: int


@dataclass(frozen=True)
class ExtremeCaseAnalysisResult:
    """Published deterministic-study artefacts."""

    output_directory: Path
    controller_paths: tuple[Path, ...]
    kimal_paths: tuple[Path, ...]
    transfer_paths: tuple[Path, ...]
    angular_paths: tuple[Path, ...]
    report_path: Path


class ExtremeCasesWorkbench:
    """Run the special NPY/NET flow without changing normal study workflows."""

    def __init__(
        self,
        root_path: str | Path | None = None,
        scenario: ContingencyScenario | None = None,
        output_directory: str | Path | None = None,
        signal_index_path: str | Path | None = None,
        assets_path: str | Path | None = None,
    ) -> None:
        self.context = ExtremeCasesContext(root_path)
        self.scenario = scenario or DEFAULT_CONTINGENCY_SCENARIO
        self.profile = EvaluationProfile.for_scenario(self.scenario)
        self._output_directory = (
            Path(output_directory).expanduser().resolve()
            if output_directory is not None
            else self.context.root_path / "Extreme_Cases_Analysis"
        )
        self.signal_index_path = (
            Path(signal_index_path).expanduser().resolve()
            if signal_index_path is not None
            else Path(__file__).resolve().parent / "model" / "signal_index.json"
        )
        self._logger = None
        self.assets = Assets(assets_path) if assets_path is not None else None

    @property
    def output_directory(self) -> Path:
        self._output_directory.mkdir(parents=True, exist_ok=True)
        return self._output_directory

    def _analysis_logger(self):
        if self._logger is None:
            self._logger = configure_analysis_logger(
                self.output_directory,
                "extreme_cases",
            )
        return self._logger

    def open_assets(self, assets_path: str | Path | None = None) -> Assets:
        """Open explicit Assets or request the workbook needed for COI analysis."""
        if assets_path is None:
            from tkinter import Tk, filedialog

            dialog = Tk()
            dialog.withdraw()
            dialog.attributes("-topmost", True)
            try:
                selected = filedialog.askopenfilename(
                    title="Select the EMTP Assets workbook for angular stability",
                    filetypes=[("Excel workbooks", "*.xlsx")],
                )
            finally:
                dialog.destroy()
            if not selected:
                raise RuntimeError("Assets selection was cancelled.")
            assets_path = selected
        self.assets = Assets(assets_path)
        return self.assets

    def prepare(self) -> ExtremeCasesPreparation:
        """Validate the native signal merge before publishing analysis artefacts."""
        logger = self._analysis_logger()
        signal_index = SignalIndex.from_json(self.signal_index_path)
        layout = signal_index.bind(self.context.reference_case.signals)
        signal_index.validate_net_catalog(self.context.reference_case.net_catalog)
        logger.info("Validated %d extreme cases with a shared NET/NPY layout", len(self.context.cases))
        logger.info("Validated native signal index: %d signals", signal_index.signal_count)
        return ExtremeCasesPreparation(
            self.output_directory,
            signal_index.path,
            signal_index.signal_count,
        )

    def analyze(self) -> ExtremeCaseAnalysisResult:
        """Publish the non-spatial Monte Carlo flow for the three fixed cases."""
        self.prepare()
        signal_index = SignalIndex.from_json(self.signal_index_path)
        layout = signal_index.bind(self.context.reference_case.signals)
        runs = []
        for index, case in enumerate(self.context.cases, start=1):
            dataset = NamedNpyDataset(case.signals, layout)
            runs.append(MonteCarloRun(index, case.directory, dataset))
        study = ExtremeCasesStudy.from_runs(tuple(runs))
        study.assets = self.assets
        pipeline = AnalysisPipeline(study, profile=self.profile)
        logger = self._analysis_logger()
        logger.info("Starting deterministic extreme-case analysis: KIMAL_220")

        kimal_measurement = study.bus("KIMAL_220").voltage_rms()
        kimal_signal_analysis = pipeline.analyze_measurement(
            kimal_measurement,
            evaluate_despite_initialization=True,
        )
        study_analysis = StudyAnalysis((
            BusStudyAnalysis("KIMAL_220", kimal_signal_analysis),
        ))
        controller_analysis = self._controller_analysis()
        transfer_analysis = pipeline.analyze_power_transfers()
        high_frequency_analysis = pipeline.detect_high_frequency_transfer_components()
        angular_result = None
        if self.assets is not None:
            angular_result = MonteCarloAngularStability(
                tuple(runs), self.scenario, self.assets
            ).create(self.output_directory / "04_Angular_Stability")
        dynamic_stability_analysis = pipeline.analyze_contingency_dynamic_stability(
            study_analysis,
            transfer_analysis,
            angular_result.analysis if angular_result is not None else None,
        )

        controller_paths = self._publish_controller_figures(
            pipeline, controller_analysis, study_analysis, dynamic_stability_analysis
        )
        kimal_paths = self._publish_kimal_figures(
            pipeline, controller_analysis, kimal_signal_analysis
        )
        transfer_paths = self._publish_transfer_figures(
            pipeline, transfer_analysis, high_frequency_analysis
        )
        report_path = write_extreme_cases_report(
            self.output_directory,
            study_analysis=study_analysis,
            transfer_analysis=transfer_analysis,
            high_frequency_analysis=high_frequency_analysis,
            dynamic_stability_analysis=dynamic_stability_analysis,
            angular_stability_analysis=(
                angular_result.analysis if angular_result is not None else None
            ),
            scenario=self.scenario,
            profile=self.profile,
            cases=self.context.cases,
        )
        if angular_result is None:
            logger.warning("Angular stability not evaluated: Assets were not provided")
        else:
            logger.info("Angular stability completed")
        logger.info("Extreme-case analysis completed")
        return ExtremeCaseAnalysisResult(
            self.output_directory,
            controller_paths,
            kimal_paths,
            transfer_paths,
            angular_result.figure_paths if angular_result is not None else (),
            report_path,
        )

    def _controller_analysis(self) -> ControllerStudyAnalysis:
        """Represent each all-unit common setting as its coincident weighted point."""
        points = tuple(
            ControllerPoint(
                run_id=run_id,
                generator_name=case.definition.label,
                fn_hz=case.definition.fn_hz,
                xi=case.definition.xi,
                nominal_power_mva=1.0,
            )
            for run_id, case in enumerate(self.context.cases, start=1)
        )
        weighted = tuple(
            WeightedControllerPoint(
                run_id=point.run_id,
                fn_hz=point.fn_hz,
                xi=point.xi,
                total_nominal_power_mva=1.0,
                unit_count=1,
            )
            for point in points
        )
        return ControllerStudyAnalysis(points, weighted, ())

    def _save_figure(self, figure, directory: Path, filename: str) -> Path:
        from utils.figure_localization import save_figure_with_spanish_copy
        import matplotlib.pyplot as plt

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        try:
            save_figure_with_spanish_copy(figure, path, dpi=150)
        finally:
            plt.close(figure)
        return path

    def _publish_controller_figures(self, pipeline, controllers, study_analysis, dynamic_analysis) -> tuple[Path, ...]:
        directory = self.output_directory / "01_Controllers"
        figures = (
            ("01_population.png", pipeline.plot_controller_population(controllers, study_analysis, dynamic_analysis)),
            ("02_weighted.png", pipeline.plot_weighted_controllers(controllers, study_analysis, dynamic_analysis)),
            ("03_weighted_log_fn.png", pipeline.plot_weighted_controllers_log_fn(controllers, study_analysis, dynamic_analysis)),
        )
        return tuple(self._save_figure(figure, directory, filename) for filename, figure in figures)

    def _publish_kimal_figures(self, pipeline, controllers, analysis) -> tuple[Path, ...]:
        directory = self.output_directory / "02_KIMAL220"
        figures = [
            ("01_initialization.png", pipeline.plot_initialization(analysis)),
            ("02_voltage.png", pipeline.plot_voltage_stability(analysis)),
            ("03_voltage_contingency.png", pipeline.plot_voltage_contingency_closeup(analysis)),
            ("04_voltage_fault_closeup.png", pipeline.plot_voltage_fault_closeup(analysis)),
            ("05_frequency.png", pipeline.plot_frequency_by_voltage_stability(analysis)),
            ("06_fault_on.png", pipeline.plot_fault_on_area_vs_weighted_controllers(controllers, analysis)),
            ("07_recovery.png", pipeline.plot_recovery_area_vs_weighted_controllers(controllers, analysis)),
            ("08_unified_response.png", pipeline.plot_unified_response_area_vs_weighted_controllers(controllers, analysis)),
            ("09_below_070.png", pipeline.plot_crisis_decision_vs_weighted_controllers(controllers, analysis)),
            ("10_below_080.png", pipeline.plot_low_voltage_duration_vs_weighted_controllers(controllers, analysis)),
        ]
        return tuple(self._save_figure(figure, directory, filename) for filename, figure in figures)

    def _publish_transfer_figures(self, pipeline, transfer_analysis, high_frequency_analysis) -> tuple[Path, ...]:
        directory = self.output_directory / "03_Transfer_Stability"
        figures = [
            ("01_power_overview.png", pipeline.plot_power_transfer_overview(transfer_analysis=transfer_analysis)),
        ]
        if any(item.damping_ratio is not None for item in transfer_analysis.assessments):
            figures.extend((
                ("02_dominant_mode_damping.png", pipeline.plot_dominant_mode_damping_summary(transfer_analysis)),
                ("03_all_em_mode_damping.png", pipeline.plot_all_em_mode_damping_summary(transfer_analysis)),
                ("04_worst_damping_cases.png", pipeline.plot_worst_transfer_damping_cases(transfer_analysis)),
                ("05_worst_damping_em_components.png", pipeline.plot_worst_transfer_electromechanical_cases(transfer_analysis)),
                ("06_transfer_mode_decomposition.png", pipeline.plot_worst_transfer_mode_decomposition(transfer_analysis)),
                ("07_high_frequency_component_spectra.png", pipeline.plot_high_frequency_component_spectra(transfer_analysis, high_frequency_analysis)),
                ("08_high_frequency_component_decomposition.png", pipeline.plot_high_frequency_component_decomposition(transfer_analysis, high_frequency_analysis)),
            ))
        return tuple(self._save_figure(figure, directory, filename) for filename, figure in figures)
