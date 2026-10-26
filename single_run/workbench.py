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

    def analysis_output_directory(self) -> Path:
        """Return the run-local directory for all focused analysis artefacts."""
        output_directory = self.run.run_path / "analysis_outputs"
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

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

    def create_plant_recovery(self) -> PlantRecoveryResult:
        """Assess terminal PV, WF, and BESS P/Q recovery under ``05_Plant_Recovery``."""
        output_directory = self.analysis_output_directory() / "05_Plant_Recovery"
        return SingleRunPlantRecovery(self.run, self.scenario).create(output_directory)

    def analyze(self) -> SingleRunAnalysisResult:
        """Run the official focused-analysis sequence and write its report.

        Individual ``create_*`` methods remain public for targeted exploration.
        This method is the reproducible complete workflow used by the playground.
        """
        general = self.create_general_overview()
        initialization = self.create_initialization()
        voltage = self.create_voltage_stability()
        damping = self.create_damping_stability()
        high_frequency = self.create_high_frequency()
        angular = self.create_angular_stability()
        plant_recovery = self.create_plant_recovery()
        report_path = self.create_report(
            initialization.analysis,
            voltage.analysis,
            damping.analysis,
            high_frequency.analysis,
            angular.analysis,
            plant_recovery.analysis,
        )
        return SingleRunAnalysisResult(
            general=general,
            initialization=initialization,
            voltage=voltage,
            damping=damping,
            high_frequency=high_frequency,
            angular=angular,
            plant_recovery=plant_recovery,
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
