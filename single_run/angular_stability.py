"""Single-run presentation adapter for shared angular-stability calculations."""

from dataclasses import dataclass
from pathlib import Path
import re

from data.assets import Assets
from data.run import MonteCarloRun
from stability.angular import (
    ANGULAR_LIMIT_DEGREES,
    ANGULAR_REFERENCE_NOTE,
    AngularCalculation,
    AngularMachineInput,
    AngularStabilityAnalysis,
    angular_excursion_limit,
    assess_angular_stability,
)
from stability.scenario import ContingencyScenario


ROTOR_ANGLE_PATTERN = re.compile(r"^(?:HE|HP|GEO|TER)_[^/]+/Teta_1_")
ANGULAR_DISCLAIMER = ANGULAR_REFERENCE_NOTE


@dataclass(frozen=True)
class AngularStabilityResult:
    """Files created by the centre-of-inertia angular assessment."""

    output_directory: Path
    analysis: AngularStabilityAnalysis
    figure_paths: tuple[Path, ...]


class SingleRunAngularStability:
    """Read one run and render the shared H·S weighted angular calculation."""

    def __init__(self, run: MonteCarloRun, scenario: ContingencyScenario, assets: Assets) -> None:
        if run.dataset is None:
            raise ValueError(f"Run {run.run_id} does not contain a Parquet result.")
        self.run = run
        self.scenario = scenario
        self.assets = assets

    def analyze(self) -> AngularStabilityAnalysis:
        return self.calculate().analysis

    def calculate(self) -> AngularCalculation:
        """Adapt published signals and Assets properties to the shared calculator."""
        self.run.dataset.inspect()
        signal_names = tuple(
            series.name for series in self.run.dataset.series if ROTOR_ANGLE_PATTERN.search(series.name)
        )
        if not signal_names:
            raise ValueError("No synchronous-machine rotor-angle signals were found.")
        data = self.run.dataset.read_series_group(signal_names)
        machines = []
        angle_values = {}
        missing_assets = []
        for signal_name in signal_names:
            machine_name = signal_name.split("/", 1)[0]
            try:
                inertia_s = self.assets.get_inertia_s(machine_name)
                nominal_power_mva = self.assets.get_nominal_power_mva(machine_name)
            except (KeyError, ValueError):
                missing_assets.append(machine_name)
                continue
            machines.append(
                AngularMachineInput(machine_name, signal_name, inertia_s, nominal_power_mva)
            )
            angle_values[machine_name] = data[signal_name].to_list()
        if missing_assets:
            names = ", ".join(sorted(set(missing_assets)))
            raise KeyError(
                "A centre-of-inertia angle requires H and S for every published "
                f"machine angle. Missing or invalid Assets entries: {names}"
            )
        return assess_angular_stability(data["time"].to_list(), angle_values, machines)

    def create(self, output_directory: str | Path) -> AngularStabilityResult:
        """Save separate COI, machine-excursion and combined figures."""
        import matplotlib.pyplot as plt

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        calculation = self.calculate()
        figures = (
            ("01_centre_of_inertia_angle.png", self._plot_centre_of_inertia(calculation)),
            ("02_machine_angular_excursions.png", self._plot_machine_excursions(calculation)),
            ("03_angular_stability_overview.png", self._plot_overview(calculation)),
        )
        paths = []
        for filename, figure in figures:
            path = output_path / filename
            figure.savefig(path, dpi=150)
            plt.close(figure)
            paths.append(path)
        obsolete_path = output_path / "01_centre_of_inertia_angular_excursions.png"
        if obsolete_path.is_file():
            obsolete_path.unlink()
        return AngularStabilityResult(output_path, calculation.analysis, tuple(paths))

    def _plot_centre_of_inertia(self, calculation: AngularCalculation):
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(13.5, 5.5))
        axis.plot(calculation.time_values, calculation.centre_of_inertia, color="#1F3A5F", linewidth=1.1)
        axis.set_ylabel("COI angle (deg)")
        axis.set_xlabel("Time (s)")
        axis.set_title("H·S-Weighted Centre-of-Inertia Angle", fontsize=12, fontweight="normal", pad=7)
        self._style_axis(axis)
        figure.subplots_adjust(left=0.085, right=0.95, bottom=0.12, top=0.90)
        return figure

    def _plot_machine_excursions(self, calculation: AngularCalculation):
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(13.5, 6.5))
        self._draw_machine_excursions(axis, calculation)
        axis.set_title("Machine Angular Excursions Relative to the Centre of Inertia", fontsize=12, fontweight="normal", pad=7)
        figure.suptitle("Angular Stability — " + ("Stable" if calculation.analysis.is_stable else "Unstable"), fontsize=13, fontweight="normal", y=0.975)
        figure.text(0.5, 0.015, ANGULAR_DISCLAIMER, ha="center", va="bottom", fontsize=9, color="#404040")
        figure.subplots_adjust(left=0.085, right=0.95, bottom=0.12, top=0.89)
        return figure

    def _plot_overview(self, calculation: AngularCalculation):
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(13.5, 9.0), sharex=True)
        top_axis, bottom_axis = axes
        top_axis.plot(calculation.time_values, calculation.centre_of_inertia, color="#1F3A5F", linewidth=1.1)
        top_axis.set_ylabel("COI angle (deg)")
        top_axis.set_title("H·S-Weighted Centre-of-Inertia Angle", fontsize=11, fontweight="normal", pad=6)
        self._style_axis(top_axis)
        self._draw_machine_excursions(bottom_axis, calculation)
        bottom_axis.set_title("Machine Angular Excursions Relative to the Centre of Inertia", fontsize=11, fontweight="normal", pad=6)
        figure.suptitle("Angular Stability — " + ("Stable" if calculation.analysis.is_stable else "Unstable"), fontsize=13, fontweight="normal", y=0.975)
        figure.text(0.5, 0.015, ANGULAR_DISCLAIMER, ha="center", va="bottom", fontsize=9, color="#404040")
        figure.subplots_adjust(left=0.085, right=0.95, bottom=0.09, top=0.91, hspace=0.30)
        return figure

    def _draw_machine_excursions(self, axis, calculation: AngularCalculation) -> None:
        import matplotlib.pyplot as plt

        colors = plt.get_cmap("tab20")
        for index, excursion in enumerate(calculation.excursions):
            axis.plot(calculation.time_values, excursion, color=colors(index % colors.N), linewidth=0.75, alpha=0.72)
        axis.axhline(ANGULAR_LIMIT_DEGREES, color="#D73027", linestyle="--", linewidth=1.0)
        axis.axhline(-ANGULAR_LIMIT_DEGREES, color="#D73027", linestyle="--", linewidth=1.0)
        axis.axhline(0.0, color="#5E6472", linewidth=0.75)
        limit = angular_excursion_limit(calculation.excursions)
        axis.set(xlim=(0.0, self.run.dataset.maximum_time_s()), ylim=(-limit, limit), xlabel="Time (s)", ylabel="Angular excursion, Δδ (deg)")
        self._style_axis(axis)

    def _style_axis(self, axis) -> None:
        axis.axvspan(self.scenario.fault_time_s, self.scenario.clearance_time_s, color="#F0B323", alpha=0.16, zorder=0)
        axis.set_xlim(0.0, self.run.dataset.maximum_time_s())
        axis.set_facecolor("white")
        axis.grid(visible=True, color="#D6D9DC", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
