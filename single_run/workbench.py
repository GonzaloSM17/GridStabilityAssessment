"""Manual workbench for generating artefacts from one published simulation run."""

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
)
from single_run.general_overview import GeneralOverview, GeneralOverviewResult
from single_run.reports import SingleRunReport
from single_run.voltage_stability import (
    SingleRunVoltageStability,
    VoltageStabilityAnalysis,
    VoltageStabilityResult,
)
from single_run.damping_stability import TransferDampingAnalysis
from single_run.angular_stability import AngularStabilityAnalysis
from stability.scenario import (
    ContingencyScenario,
    DEFAULT_CONTINGENCY_SCENARIO,
    EvaluationProfile,
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
        return MonteCarloRun.from_directory(run_path)

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

    def create_report(
        self,
        voltage: VoltageStabilityAnalysis,
        damping: TransferDampingAnalysis,
        angular: AngularStabilityAnalysis,
    ) -> Path:
        """Write the database-style case report at the analysis-output root."""
        return SingleRunReport(
            self.run,
            self.profile,
            voltage,
            damping,
            angular,
        ).write(self.analysis_output_directory())

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
