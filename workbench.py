"""Manual workbench for opening studies and running selected analysis operations."""

from pathlib import Path

from analyzer import AnalysisPipeline
from assets import Assets
from context import MonteCarloContext
from reports import AnalysisReport
from signals import VOLTAGE_RMS
from study import MonteCarloStudy


class MonteCarloWorkbench:
    """Open a study and expose its analysis pipeline for manual exploration."""

    def __init__(
        self,
        context_path: str | Path | None = None,
        assets_path: str | Path | None = None,
    ) -> None:
        self.study: MonteCarloStudy | None = None
        self._pipeline: AnalysisPipeline | None = None
        self.open_study(context_path, assets_path)

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
    ) -> AnalysisPipeline:
        """Open explicit inputs or request and validate missing inputs through Tk."""
        if context_path is None:
            context_path = self._choose_context_path()

        context = MonteCarloContext(context_path)

        if assets_path is None:
            assets_path = self._choose_assets_path()

        assets = Assets(assets_path)
        self.study = MonteCarloStudy(context, assets)
        self._pipeline = AnalysisPipeline(self.study)
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

    def analysis_output_directory(self) -> Path:
        """Return the active study directory for generated analysis artefacts."""
        if self.study is None or self.study.context.context_path is None:
            raise RuntimeError("Open a study before requesting an output directory.")

        output_directory = self.study.context.context_path / "analysis_outputs"
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def analyze_voltage_rms(self, bus_name: str, *, show_figures: bool = True):
        """Run and publish the current RMS-voltage analysis for one bus.

        Controller figures are study-wide and are generated once per execution.
        Frequency is published only as a complementary trace for the selected bus;
        it is not part of the RMS-voltage assessment.
        """
        if self.study is None:
            raise RuntimeError("Open a study before starting an analysis.")

        controller_analysis = self.pipeline.analyze_controllers()
        controller_output_directory = self.controller_output_directory()
        self.save_figure(
            self.pipeline.plot_controller_population(controller_analysis),
            controller_output_directory,
            "01_population.png",
        )
        self.save_figure(
            self.pipeline.plot_weighted_controllers(controller_analysis),
            controller_output_directory,
            "02_weighted.png",
        )

        bus = self.study.bus(bus_name)
        voltage_measurement = bus.measurement(VOLTAGE_RMS)
        voltage_analysis = self.pipeline.analyze_measurement(voltage_measurement)
        bus_output_directory = self.bus_output_directory(bus.name)
        self.save_figure(
            self.pipeline.plot_initialization(voltage_analysis),
            bus_output_directory,
            "01_initialization.png",
        )

        if voltage_analysis.has_stability_assessments:
            self.save_figure(
                self.pipeline.plot_voltage_stability(voltage_analysis),
                bus_output_directory,
                "02_voltage.png",
            )
            self.save_figure(
                self.pipeline.plot_frequency_by_voltage_stability(voltage_analysis),
                bus_output_directory,
                "03_frequency.png",
            )
            if controller_analysis.weighted_run_points:
                self._save_controller_indicator_figures(
                    controller_analysis,
                    voltage_analysis,
                    bus_output_directory,
                )

        AnalysisReport(bus.name, voltage_analysis).write(
            self.analysis_output_directory()
        )

        if show_figures:
            import matplotlib.pyplot as plt

            plt.show()

        return voltage_analysis

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
            (
                path
                for path in existing_directories
                if path.name.endswith(bus_suffix)
            ),
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

    @staticmethod
    def save_figure(figure, output_directory: Path, filename: str) -> Path:
        """Save one figure using the selected stable filename."""
        output_path = output_directory / filename
        figure.savefig(output_path, dpi=150)
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
                "04_post_clearance.png",
                self.pipeline.plot_post_clearance_area_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
            (
                "05_fault_on.png",
                self.pipeline.plot_fault_on_area_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
            (
                "06_recovery.png",
                self.pipeline.plot_recovery_area_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
            (
                "07_unified_response.png",
                self.pipeline.plot_unified_response_area_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
            (
                "08_persistence.png",
                self.pipeline.plot_post_fault_persistence_area_vs_weighted_controllers(
                    controller_analysis,
                    voltage_analysis,
                ),
            ),
        )
        for filename, figure in figures:
            self.save_figure(figure, output_directory, filename)

