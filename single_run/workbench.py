"""Focused workflow and individual artefacts for one published simulation run."""

from dataclasses import dataclass
from pathlib import Path

from data.assets import Assets
from data.run import MonteCarloRun

from single_run.angular_stability import (
    AngularStabilityResult,
    SingleRunAngularStability,
)
from single_run.damping_stability import (
    DampingStabilityResult,
    SingleRunDampingStability,
    TransferDampingAnalysis,
)
from single_run.general_overview import GeneralOverview, GeneralOverviewResult
from single_run.initialization import (
    InitializationAnalysis,
    InitializationResult,
    SingleRunInitialization,
)
from single_run.plant_recovery import (
    PlantRecoveryAnalysis,
    PlantRecoveryResult,
    SingleRunPlantRecovery,
)
from single_run.high_frequency import (
    HighFrequencyAnalysis,
    HighFrequencyResult,
    SingleRunHighFrequency,
)
from single_run.reports import SingleRunReport
from single_run.voltage_stability import (
    SingleRunVoltageStability,
    VoltageStabilityAnalysis,
    VoltageStabilityResult,
)
from stability.angular import AngularStabilityAnalysis
from stability.dynamic_stability import assess_dynamic_stability
from stability.scenario import (
    ContingencyScenario,
    DEFAULT_CONTINGENCY_SCENARIO,
    EvaluationProfile,
)
from utils.analysis_logging import configure_analysis_logger


@dataclass(frozen=True)
class SingleRunAnalysisResult:
    """All artefacts and the report produced by the official single-run flow."""

    general: GeneralOverviewResult
    initialization: InitializationResult
    voltage: VoltageStabilityResult
    damping: DampingStabilityResult
    high_frequency: HighFrequencyResult
    angular: AngularStabilityResult
    plant_recovery: PlantRecoveryResult
    spatial: object | None
    report_path: Path

    @property
    def artefact_results(self) -> tuple[object, ...]:
        """Return the created figure-producing artefacts in publication order."""
        return (
            self.general,
            self.initialization,
            self.voltage,
            self.damping,
            self.high_frequency,
            self.angular,
            self.plant_recovery,
        )


class SingleRunWorkbench:
    """Open one run and expose focused single-scenario analysis operations."""

    def __init__(
        self,
        run_path: str | Path | None = None,
        scenario: ContingencyScenario | None = None,
        assets_path: str | Path | None = None,
    ) -> None:
        self._output_directory: Path | None = None
        self._logger = None
        self.scenario = scenario or DEFAULT_CONTINGENCY_SCENARIO
        self.profile = EvaluationProfile.for_scenario(self.scenario)
        self.run = self.open_run(run_path)
        self.assets = Assets(assets_path) if assets_path is not None else None

    def open_run(self, run_path: str | Path | None = None) -> MonteCarloRun:
        """Open an explicit run directory or select one through Tk."""
        if run_path is None:
            run_path = self._choose_run_path()
        return MonteCarloRun.from_directory(
            run_path,
            allow_non_monte_carlo_name=True,
        )

    def select_output_directory(self, output_directory: str | Path | None = None) -> Path:
        """Choose the common destination before generating any analysis output."""
        from utils.output_directory import select_output_directory
        self._output_directory = select_output_directory(output_directory)
        self._logger = None
        self._analysis_logger().info("Output directory selected: %s", self._output_directory)
        return self._output_directory

    def analysis_output_directory(self) -> Path:
        """Return the run-local directory for all focused analysis artefacts."""
        output_directory = self._output_directory if self._output_directory is not None else self.run.run_path / "analysis_outputs"
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def _analysis_logger(self):
        if self._logger is None:
            self._logger = configure_analysis_logger(
                self.analysis_output_directory(),
                "single_run",
            )
        return self._logger

    def create_general_overview(self) -> GeneralOverviewResult:
        """Create the fixed system-wide overview under ``00_General``."""
        output_directory = self.analysis_output_directory() / "00_General"
        angular_calculation = (
            SingleRunAngularStability(self.run, self.scenario, self.assets).calculate()
            if self.assets is not None
            else None
        )
        return GeneralOverview(
            self.run,
            self.scenario,
            angular_calculation,
        ).create(output_directory)

    def create_initialization(self) -> InitializationResult:
        """Assess and plot 220/500 kV RMS-voltage initialization under ``01_Initialization``."""
        output_directory = self.analysis_output_directory() / "01_Initialization"
        return SingleRunInitialization(
            self.run,
            self.profile.initialization_criteria,
        ).create(output_directory)

    def create_voltage_stability(self) -> VoltageStabilityResult:
        """Assess all 220/500 kV voltages under ``02_Voltage_Stability``."""
        output_directory = self.analysis_output_directory() / "02_Voltage_Stability"
        return SingleRunVoltageStability(self.run, self.profile).create(
            output_directory
        )

    def create_damping_stability(self) -> DampingStabilityResult:
        """Assess transfer EM damping under ``03_Damping_Stability``."""
        output_directory = self.analysis_output_directory() / "03_Damping_Stability"
        return SingleRunDampingStability(self.run, self.profile).create(
            output_directory
        )

    def create_high_frequency(self) -> HighFrequencyResult:
        """Observe high-frequency transfer components under ``03_Damping_Stability``."""
        output_directory = self.analysis_output_directory() / "03_Damping_Stability"
        return SingleRunHighFrequency(self.run, self.profile).create(output_directory)

    def create_angular_stability(self) -> AngularStabilityResult:
        """Create the H·S weighted angular assessment under ``04_Angular_Stability``."""
        if self.assets is None:
            raise ValueError(
                "Angular stability requires Assets with 'App. Power [MVA]' and "
                "'Inertia [s]'. Provide assets_path when creating the workbench."
            )
        output_directory = self.analysis_output_directory() / "04_Angular_Stability"
        return SingleRunAngularStability(self.run, self.scenario, self.assets).create(
            output_directory
        )

    def open_assets(self, assets_path: str | Path | None = None) -> Assets:
        """Open explicit Assets or use the existing workbench Excel picker."""
        if assets_path is None:
            from montecarlo.workbench import MonteCarloWorkbench
            assets_path = MonteCarloWorkbench._choose_assets_path()
        self.assets = Assets(assets_path)
        return self.assets

    def create_spatial_stability(
        self, *, output_directory: str | Path | None = None,
        voltage: VoltageStabilityAnalysis | None = None,
        damping: TransferDampingAnalysis | None = None,
        angular: AngularStabilityAnalysis | None = None,
    ):
        """Create xi/fn maps using native dynamic stability, optionally reusing analyses."""
        from montecarlo.spatial_stability import SpatialStabilityAnalyzer
        if self.assets is None:
            self.open_assets()
        output = Path(output_directory) if output_directory is not None else self.analysis_output_directory() / "06_Spatial_Stability"
        components = {self.run.display_name: dict(voltage=voltage, damping=damping, angular=angular)}
        result = SpatialStabilityAnalyzer.create_optional(
            self.assets,
            self.profile,
            (self.run,),
            output,
            component_analyses=components,
        )
        logger = self._analysis_logger()
        if result.analysis is not None:
            for item in result.analysis.skipped_runs:
                logger.warning("Spatial stability skipped %s: %s", item.case_id, item.reason)
        if result.status == "Skipped":
            logger.warning("Spatial stability skipped: %s", result.reason)
        else:
            logger.info(
                "Spatial stability %s: %d map(s)",
                result.status.lower(),
                len(result.figure_paths),
            )
        return result

    def create_plant_recovery(self) -> PlantRecoveryResult:
        """Assess terminal PV, WF, and BESS P/Q recovery under ``05_Plant_Recovery``."""
        output_directory = self.analysis_output_directory() / "05_Plant_Recovery"
        return SingleRunPlantRecovery(self.run, self.scenario).create(output_directory)

    def analyze(self) -> SingleRunAnalysisResult:
        """Run the official focused-analysis sequence and write its report.

        Individual ``create_*`` methods remain public for targeted exploration.
        This method is the reproducible complete workflow used by the playground.
        """
        logger = self._analysis_logger()
        logger.info("Starting single-run analysis: %s", self.run.display_name)
        logger.info("Starting general overview")
        general = self.create_general_overview()
        logger.info("General overview completed")
        logger.info("Starting initialization assessment")
        initialization = self.create_initialization()
        logger.info("Initialization assessment completed")
        logger.info("Starting voltage stability")
        voltage = self.create_voltage_stability()
        logger.info("Voltage stability completed")
        logger.info("Starting damping stability")
        damping = self.create_damping_stability()
        logger.info("Damping stability completed")
        logger.info("Starting high-frequency observation")
        high_frequency = self.create_high_frequency()
        logger.info("High-frequency observation completed")
        logger.info("Starting angular stability")
        angular = self.create_angular_stability()
        logger.info("Angular stability completed")
        logger.info("Starting plant recovery")
        plant_recovery = self.create_plant_recovery()
        logger.info("Plant recovery completed")
        # Spatial aggregation is a Monte Carlo workflow. Keep the public
        # create_spatial_stability() method for deliberate manual use, but do
        # not execute it from the official focused single-run sequence.
        spatial = None
        logger.info("Spatial stability is disabled for the single-run workflow")
        report_path = self.create_report(
            initialization.analysis,
            voltage.analysis,
            damping.analysis,
            high_frequency.analysis,
            angular.analysis,
            plant_recovery.analysis,
            spatial,
        )
        logger.info("Analysis report created")
        logger.info("Single-run analysis completed")
        return SingleRunAnalysisResult(
            general=general,
            initialization=initialization,
            voltage=voltage,
            damping=damping,
            high_frequency=high_frequency,
            angular=angular,
            plant_recovery=plant_recovery,
            spatial=spatial,
            report_path=report_path,
        )

    def create_report(
        self,
        initialization: InitializationAnalysis,
        voltage: VoltageStabilityAnalysis,
        damping: TransferDampingAnalysis,
        high_frequency: HighFrequencyAnalysis,
        angular: AngularStabilityAnalysis,
        plant_recovery: PlantRecoveryAnalysis,
        spatial_stability=None,
    ) -> Path:
        """Write the database-style case report at the analysis-output root."""
        return SingleRunReport(
            self.run,
            self.profile,
            initialization,
            voltage,
            damping,
            high_frequency,
            angular,
            plant_recovery,
            spatial_stability,
            assess_dynamic_stability(
                self.run.run_id,
                self._voltage_status(voltage),
                self._damping_status(damping),
                "Stable" if angular.is_stable else "Unstable",
            ),
        ).write(self.analysis_output_directory())

    @staticmethod
    def _voltage_status(voltage: VoltageStabilityAnalysis) -> str:
        return "Stable" if all(item.assessment.is_stable for item in voltage.assessments) else "Unstable"

    @staticmethod
    def _damping_status(damping: TransferDampingAnalysis) -> str:
        assessments = [
            record.assessment
            for record in damping.records
            if record.assessment.status not in {"opened", "inactive_pre_fault"}
        ]
        if not assessments:
            return "Not applicable"
        if any(item.status == "growing" for item in assessments):
            return "Growing"
        if any(item.status == "damped_under" for item in assessments):
            return "Damped under"
        if any(item.damping_ratio is not None for item in assessments):
            return "Damped"
        return "Not measurable"

    @staticmethod
    def _choose_run_path() -> Path:
        """Request a single published run directory only when no path is supplied."""
        from tkinter import Tk, filedialog

        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        try:
            selected_path = filedialog.askdirectory(
                title="Select one published simulation run"
            )
        finally:
            window.destroy()

        if not selected_path:
            raise RuntimeError("Simulation run selection was cancelled.")
        return Path(selected_path)
