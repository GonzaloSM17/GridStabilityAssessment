"""Manual workbench for opening studies and running selected analysis operations."""

from pathlib import Path

from montecarlo.batch_analysis import (
    AnalysisPipeline,
    BusStudyAnalysis,
    ContingencyDynamicStabilityAnalysis,
    StudyAnalysis,
)
from data.assets import Assets
from data.signals import VOLTAGE_RMS
from montecarlo.context import MonteCarloContext
from montecarlo.reports import AnalysisReport
from montecarlo.study import MonteCarloStudy
from stability.scenario import ContingencyScenario, EvaluationProfile
from utils.analysis_logging import configure_analysis_logger


class MonteCarloWorkbench:
    """Open a study and expose its analysis pipeline for manual exploration."""

    def __init__(
        self,
        context_path: str | Path | None = None,
        assets_path: str | Path | None = None,
        scenario: ContingencyScenario | None = None,
        profile: EvaluationProfile | None = None,
    ) -> None:
        self._output_directory: Path | None = None
        self._show_figures = False
        self._logger = None
        self.study: MonteCarloStudy | None = None
        self._pipeline: AnalysisPipeline | None = None
        self.dynamic_stability_analysis: ContingencyDynamicStabilityAnalysis | None = None
        self.spatial_stability_result = None
        self.open_study(context_path, assets_path, scenario, profile)

    @property
    def pipeline(self) -> AnalysisPipeline:
        """Return the analysis pipeline for the active, validated study."""
        if self._pipeline is None:
            raise RuntimeError("No study is open in this workbench.")
        return self._pipeline

    def open_study(
        self,
        context_path: str | Path | None = None,
        assets_path: str | Path | None = None,
        scenario: ContingencyScenario | None = None,
        profile: EvaluationProfile | None = None,
    ) -> AnalysisPipeline:
        """Open explicit inputs or request and validate missing inputs through Tk."""
        if context_path is None:
            context_path = self._choose_context_path()

        context = MonteCarloContext(context_path)

        if assets_path is None:
            assets_path = self._choose_assets_path()

        assets = Assets(assets_path)
        self.study = MonteCarloStudy(context, assets)
        self._pipeline = AnalysisPipeline(self.study, scenario, profile)
        return self._pipeline

    @staticmethod
    def _choose_context_path() -> Path:
        """Request a study directory only when the workbench has no path."""
        from tkinter import Tk, filedialog

        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        try:
            selected_path = filedialog.askdirectory(
                title="Select the Monte Carlo study directory"
            )
        finally:
            window.destroy()

        if not selected_path:
            raise RuntimeError("Monte Carlo study selection was cancelled.")
        return Path(selected_path)

    @staticmethod
    def _choose_assets_path() -> Path:
        """Request an Assets workbook only when the workbench has no path."""
        from tkinter import Tk, filedialog

        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        try:
            selected_path = filedialog.askopenfilename(
                title="Select the EMTP Assets workbook",
                filetypes=[("Excel workbooks", "*.xlsx")],
            )
        finally:
            window.destroy()

        if not selected_path:
            raise RuntimeError("Assets workbook selection was cancelled.")
        return Path(selected_path)

    def select_output_directory(self, output_directory: str | Path | None = None) -> Path:
        """Choose the common destination before generating any analysis output."""
        from utils.output_directory import select_output_directory
        self._output_directory = select_output_directory(output_directory)
        self._logger = None
        self._analysis_logger().info("Output directory selected: %s", self._output_directory)
        return self._output_directory

    def analysis_output_directory(self) -> Path:
        """Return the active study directory for generated analysis artefacts."""
        if self.study is None or self.study.context.context_path is None:
            raise RuntimeError("Open a study before requesting an output directory.")

        output_directory = self._output_directory if self._output_directory is not None else self.study.context.context_path / "analysis_outputs"
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def _analysis_logger(self):
        if self._logger is None:
            self._logger = configure_analysis_logger(
                self.analysis_output_directory(),
                "montecarlo",
            )
        return self._logger

    def analyze(
        self,
        *,
        buses: tuple[str, ...],
        show_figures: bool = False,
    ) -> StudyAnalysis:
        """Analyze selected buses, publish local artefacts, and return their result."""
        if self.study is None:
            raise RuntimeError("Open a study before starting an analysis.")
        if not buses:
            raise ValueError("Select at least one bus for analysis.")
        if len(buses) != len(set(buses)):
            raise ValueError("A bus may be selected only once.")

        logger = self._analysis_logger()
        logger.info(
            "Starting Monte Carlo analysis for %d run(s); selected buses: %s",
            len(self.study.context.runs),
            ", ".join(buses),
        )
        self._show_figures = show_figures
        logger.info("Starting selected-bus voltage analysis")
        bus_analyses = tuple(
            BusStudyAnalysis(
                bus_name=bus_name,
                signal_analysis=self.pipeline.analyze_measurement(
                    self.study.bus(bus_name).measurement(VOLTAGE_RMS)
                ),
            )
            for bus_name in buses
        )
        study_analysis = StudyAnalysis(bus_analyses)
        logger.info("Selected-bus voltage analysis completed")
        logger.info("Starting controller population analysis")
        controller_analysis = self.pipeline.analyze_controllers()
        for run_id, reason in controller_analysis.skipped_run_reasons:
            logger.warning("Controller population skipped run_%06d: %s", run_id, reason)
        logger.info(
            "Controller population completed: %d evaluated, %d skipped",
            len(controller_analysis.weighted_run_points),
            len(controller_analysis.skipped_run_ids),
        )
        logger.info("Starting active-power transfer analysis")
        transfer_analysis = self.pipeline.analyze_power_transfers()
        high_frequency_analysis = self.pipeline.detect_high_frequency_transfer_components()
        logger.info("Active-power transfer analysis completed")

        controller_output_directory = self.controller_output_directory()
        self.save_figure(
            self.pipeline.plot_controller_population(
                controller_analysis,
                study_analysis,
            ),
            controller_output_directory,
            "01_population.png",
        )
        self.save_figure(
            self.pipeline.plot_weighted_controllers(
                controller_analysis,
                study_analysis,
            ),
            controller_output_directory,
            "02_weighted.png",
        )

        for bus_analysis in study_analysis.bus_analyses:
            self._publish_bus_analysis(bus_analysis, controller_analysis)

        self._publish_transfer_analysis(transfer_analysis, high_frequency_analysis)

        self.dynamic_stability_analysis = self.pipeline.analyze_contingency_dynamic_stability(
            study_analysis,
            transfer_analysis,
        )
        logger.info("Starting spatial stability")
        self.spatial_stability_result = self.create_spatial_stability()

        AnalysisReport(
            study_analysis,
            transfer_analysis,
            high_frequency_analysis,
            self.dynamic_stability_analysis,
            scenario=self.pipeline.scenario,
            profile=self.pipeline.profile,
            spatial_stability=self.spatial_stability_result,
        ).write(
            self.analysis_output_directory()
        )
        logger.info("Analysis report created")

        if show_figures:
            import matplotlib.pyplot as plt

            plt.show()

        self._show_figures = False
        logger.info("Monte Carlo analysis completed")
        return study_analysis

    def create_spatial_stability(self, *, output_directory: str | Path | None = None):
        """Create population maps with voltage, damping AND angular assessment per run.

        This dedicated result does not relabel the existing batch reports, whose
        legacy dynamic decision currently omits the angular criterion.
        """
        from montecarlo.spatial_stability import SpatialStabilityAnalyzer
        if self.study is None:
            raise RuntimeError("Open a study before starting spatial analysis.")
        output = Path(output_directory) if output_directory is not None else self.analysis_output_directory() / "04_Spatial_Stability"
        result = SpatialStabilityAnalyzer.create_optional(
            self.study.assets,
            self.pipeline.profile,
            self.study.context.runs,
            output,
        )
        logger = self._analysis_logger()
        if result.analysis is not None:
            for item in result.analysis.skipped_runs:
                logger.warning("Spatial stability skipped %s: %s", item.case_id, item.reason)
        if result.status == "Skipped":
            logger.warning("Spatial stability skipped: %s", result.reason)
        else:
            skipped_count = len(result.analysis.skipped_runs) if result.analysis else 0
            logger.info(
                "Spatial stability %s: %d map(s), %d skipped run(s)",
                result.status.lower(),
                len(result.figure_paths),
                skipped_count,
            )
        return result

    def plot_power_transfers(
        self,
        *,
        start_time_s: float = 0.9,
        end_time_s: float | None = None,
        show_figure: bool = False,
    ) -> Path:
        """Calculate and publish active-power transfer modal-damping figures."""
        self._show_figures = show_figure
        transfer_analysis = self.pipeline.analyze_power_transfers()
        high_frequency_analysis = self.pipeline.detect_high_frequency_transfer_components()
        output_paths = self._publish_transfer_analysis(
            transfer_analysis,
            high_frequency_analysis,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        if show_figure:
            import matplotlib.pyplot as plt

            plt.show()
        self._show_figures = False
        return output_paths[0]

    def _publish_transfer_analysis(
        self,
        transfer_analysis,
        high_frequency_analysis,
        *,
        start_time_s: float = 0.9,
        end_time_s: float | None = None,
    ) -> tuple[Path, ...]:
        """Publish the overview, damping distribution, and worst-case evidence."""
        output_directory = self.analysis_output_directory() / "03_transfers"
        output_directory.mkdir(parents=True, exist_ok=True)
        figures = [
            (
                "01_power_overview.png",
                self.pipeline.plot_power_transfer_overview(
                    start_time_s=start_time_s,
                    end_time_s=end_time_s,
                    transfer_analysis=transfer_analysis,
                ),
            )
        ]
        if any(
            item.damping_ratio is not None for item in transfer_analysis.assessments
        ):
            figures.extend(
                (
                    (
                        "02_dominant_mode_damping.png",
                        self.pipeline.plot_dominant_mode_damping_summary(
                            transfer_analysis
                        ),
                    ),
                    (
                        "03_all_em_mode_damping.png",
                        self.pipeline.plot_all_em_mode_damping_summary(
                            transfer_analysis
                        ),
                    ),
                    (
                        "04_worst_damping_cases.png",
                        self.pipeline.plot_worst_transfer_damping_cases(
                            transfer_analysis
                        ),
                    ),
                    (
                        "05_worst_damping_em_components.png",
                        self.pipeline.plot_worst_transfer_electromechanical_cases(
                            transfer_analysis
                        ),
                    ),
                    (
                        "06_transfer_mode_decomposition.png",
                        self.pipeline.plot_worst_transfer_mode_decomposition(
                            transfer_analysis
                        ),
                    ),
                    (
                        "07_high_frequency_component_spectra.png",
                        self.pipeline.plot_high_frequency_component_spectra(
                            transfer_analysis,
                            high_frequency_analysis,
                        ),
                    ),
                    (
                        "08_high_frequency_component_decomposition.png",
                        self.pipeline.plot_high_frequency_component_decomposition(
                            transfer_analysis,
                            high_frequency_analysis,
                        ),
                    ),
                )
            )
        obsolete_filenames = (
            "02_damping_by_transfer.png",
            "03_worst_damping_cases.png",
            "04_worst_damping_em_components.png",
            "05_transfer_mode_decomposition.png",
            "06_high_frequency_component_spectra.png",
            "07_high_frequency_component_decomposition.png",
        )
        for filename in obsolete_filenames:
            obsolete_path = output_directory / filename
            if obsolete_path.is_file():
                obsolete_path.unlink()
        return tuple(
            self.save_figure(figure, output_directory, filename)
            for filename, figure in figures
        )

    def _publish_bus_analysis(
        self,
        bus_analysis: BusStudyAnalysis,
        controller_analysis,
    ) -> None:
        """Publish the local figures for one selected bus."""
        bus_output_directory = self.bus_output_directory(bus_analysis.bus_name)
        voltage_analysis = bus_analysis.signal_analysis
        self.save_figure(
            self.pipeline.plot_initialization(voltage_analysis),
            bus_output_directory,
            "01_initialization.png",
        )

        if not voltage_analysis.has_stability_assessments:
            return

        self.save_figure(
            self.pipeline.plot_voltage_stability(voltage_analysis),
            bus_output_directory,
            "02_voltage.png",
        )
        self.save_figure(
            self.pipeline.plot_voltage_contingency_closeup(voltage_analysis),
            bus_output_directory,
            "03_voltage_contingency.png",
        )
        self.save_figure(
            self.pipeline.plot_voltage_fault_closeup(voltage_analysis),
            bus_output_directory,
            "04_voltage_fault_closeup.png",
        )
        self.save_figure(
            self.pipeline.plot_frequency_by_voltage_stability(voltage_analysis),
            bus_output_directory,
            "05_frequency.png",
        )
        if controller_analysis.weighted_run_points:
            self._save_controller_indicator_figures(
                controller_analysis,
                voltage_analysis,
                bus_output_directory,
            )

    def controller_output_directory(self) -> Path:
        """Return the dedicated output directory for study-wide controller figures."""
        output_directory = self.analysis_output_directory() / "01_controllers"
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def bus_output_directory(self, bus_name: str) -> Path:
        """Return one direct, numbered output directory for a discovered bus."""
        if self.study is None:
            raise RuntimeError("Open a study before requesting a bus output directory.")

        bus = self.study.bus(bus_name)
        root_directory = self.analysis_output_directory()
        existing_directories = [
            path for path in root_directory.iterdir() if path.is_dir()
        ]
        bus_suffix = f"_{bus.name}"
        existing_directory = next(
            (path for path in existing_directories if path.name.endswith(bus_suffix)),
            None,
        )
        if existing_directory is not None:
            return existing_directory

        directory_numbers = [
            int(path.name.split("_", 1)[0])
            for path in existing_directories
            if path.name.split("_", 1)[0].isdigit()
        ]
        bus_number = max(directory_numbers, default=1) + 1
        output_directory = root_directory / f"{bus_number:02d}_{bus.name}"
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def save_figure(self, figure, output_directory: Path, filename: str) -> Path:
        """Save and release the figure unless interactive display was requested."""
        output_path = output_directory / filename
        try:
            figure.savefig(output_path, dpi=150)
        finally:
            if not self._show_figures:
                import matplotlib.pyplot as plt
                plt.close(figure)
        return output_path

    def _save_controller_indicator_figures(
        self,
        controller_analysis,
        voltage_analysis,
        output_directory: Path,
    ) -> None:
        """Publish controller indicators derived from the selected voltage analysis."""
        figures = (
            (
                "06_fault_on.png",
                self.pipeline.plot_fault_on_area_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
            (
                "07_recovery.png",
                self.pipeline.plot_recovery_area_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
            (
                "08_unified_response.png",
                self.pipeline.plot_unified_response_area_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
            (
                "09_below_070.png",
                self.pipeline.plot_crisis_decision_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
            (
                "10_below_080.png",
                self.pipeline.plot_low_voltage_duration_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
        )
        for filename, figure in figures:
            self.save_figure(figure, output_directory, filename)
