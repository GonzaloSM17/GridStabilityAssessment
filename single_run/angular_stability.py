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
    angular_excursion_ticks,
    assess_angular_stability,
)
from stability.scenario import ContingencyScenario
from utils.figure_localization import (
    remove_figure_with_spanish_copy,
    save_figure_with_spanish_copy,
)


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
        """Save COI, time-series, overview and machine-excursion evidence."""
        import matplotlib.pyplot as plt

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        calculation = self.calculate()
        figures = (
            ("01_centre_of_inertia_angle.png", self._plot_centre_of_inertia(calculation)),
            ("02_machine_angular_excursions.png", self._plot_machine_excursions(calculation)),
            ("03_angular_stability_overview.png", self._plot_overview(calculation)),
            (
                "04_machine_excursion_envelope.png",
                self._plot_machine_excursion_envelope(calculation),
            ),
            (
                "05_angular_stability_tmax.png",
                self._plot_angular_stability_at_tmax(calculation),
            ),
        )
        paths = []
        for filename, figure in figures:
            path = output_path / filename
            save_figure_with_spanish_copy(figure, path, dpi=150)
            plt.close(figure)
            paths.append(path)
        obsolete_path = output_path / "01_centre_of_inertia_angular_excursions.png"
        if obsolete_path.is_file():
            remove_figure_with_spanish_copy(obsolete_path)
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

    def _plot_machine_excursion_envelope(self, calculation: AngularCalculation):
        """Show the signed excursion envelope and critical extreme per machine."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        machines = sorted(
            calculation.analysis.machines,
            key=lambda machine: machine.maximum_absolute_excursion_degrees,
            reverse=True,
        )
        limit = angular_excursion_limit(
            (
                (machine.minimum_excursion_degrees, machine.maximum_excursion_degrees)
                for machine in machines
            )
        )
        figure, axis = plt.subplots(
            figsize=(13.5, max(5.5, 0.40 * len(machines) + 2.7))
        )
        for position, machine in enumerate(machines):
            color = "#D55E00" if not machine.is_within_limit else "#0072B2"
            representative = (
                machine.maximum_excursion_degrees
                if abs(machine.maximum_excursion_degrees)
                >= abs(machine.minimum_excursion_degrees)
                else machine.minimum_excursion_degrees
            )
            axis.hlines(
                position,
                machine.minimum_excursion_degrees,
                machine.maximum_excursion_degrees,
                color=color,
                linewidth=3.0,
                alpha=0.78,
                zorder=2,
            )
            axis.scatter(
                representative,
                position,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                s=42,
                zorder=3,
            )
        axis.axvline(0.0, color="#5E6472", linewidth=0.8, zorder=0)
        for value in (-ANGULAR_LIMIT_DEGREES, ANGULAR_LIMIT_DEGREES):
            axis.axvline(value, color="#D73027", linestyle="--", linewidth=1.0, zorder=1)
        axis.set(
            xlim=(-limit, limit),
            xticks=angular_excursion_ticks(limit),
            yticks=range(len(machines)),
            yticklabels=[machine.machine_name for machine in machines],
            xlabel="Angular excursion relative to COI, Δδ (deg)",
            ylabel="Synchronous machine",
            title="Machine Excursion Envelope Relative to the Centre of Inertia",
        )
        axis.invert_yaxis()
        axis.grid(axis="x", color="#D6D9DC", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            handles=(
                Line2D([], [], color="#0072B2", linewidth=3, label="Within ±120°"),
                Line2D([], [], color="#D55E00", linewidth=3, label="At or beyond ±120°"),
                Line2D(
                    [], [], color="#4A4A4A", marker="o", linestyle="None",
                    label="Extreme with maximum |Δδ|",
                ),
            ),
            loc="lower right",
            framealpha=0.92,
        )
        figure.suptitle(
            "Maximum Angular Excursion by Synchronous Machine",
            fontsize=13,
            fontweight="normal",
            y=0.975,
        )
        figure.text(
            0.5,
            0.015,
            "Each horizontal segment spans the machine minimum and maximum COI-referenced excursion; "
            "the marker identifies the signed extreme with the greatest absolute value.",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#404040",
        )
        figure.subplots_adjust(left=0.25, right=0.96, bottom=0.12, top=0.90)
        return figure

    def _plot_angular_stability_at_tmax(self, calculation: AngularCalculation):
        """Render the COI and machine evidence through the available tmax."""
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(13.5, 9.0), sharex=True)
        coi_axis, machine_axis = axes
        end_time_s = self.run.dataset.maximum_time_s()
        coi_axis.plot(
            calculation.time_values,
            calculation.centre_of_inertia,
            color="#1F3A5F",
            linewidth=1.1,
        )
        coi_axis.set_ylabel("COI angle (deg)")
        coi_axis.set_title(
            "H·S-Weighted Centre-of-Inertia Angle",
            fontsize=11,
            fontweight="normal",
            pad=6,
        )
        self._style_axis(coi_axis, end_time_s=end_time_s)
        self._draw_machine_excursions(
            machine_axis,
            calculation,
            end_time_s=end_time_s,
        )
        machine_axis.set_title(
            "Machine Angular Excursions Relative to the Centre of Inertia — Full Simulation",
            fontsize=11,
            fontweight="normal",
            pad=6,
        )
        figure.suptitle(
            "Angular Stability — "
            + ("Stable" if calculation.analysis.is_stable else "Unstable"),
            fontsize=13,
            fontweight="normal",
            y=0.975,
        )
        figure.text(
            0.5,
            0.015,
            "Displayed through the available simulation tmax; the ±120° lines remain the angular-separation limits.",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#404040",
        )
        figure.subplots_adjust(left=0.085, right=0.95, bottom=0.09, top=0.91, hspace=0.30)
        return figure

    def _draw_machine_excursions(
        self,
        axis,
        calculation: AngularCalculation,
        *,
        end_time_s: float | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        colors = plt.get_cmap("tab20")
        for index, excursion in enumerate(calculation.excursions):
            axis.plot(calculation.time_values, excursion, color=colors(index % colors.N), linewidth=0.75, alpha=0.72)
        axis.axhline(ANGULAR_LIMIT_DEGREES, color="#D73027", linestyle="--", linewidth=1.0)
        axis.axhline(-ANGULAR_LIMIT_DEGREES, color="#D73027", linestyle="--", linewidth=1.0)
        axis.axhline(0.0, color="#5E6472", linewidth=0.75)
        limit = angular_excursion_limit(calculation.excursions)
        axis.set_yticks(angular_excursion_ticks(limit))
        axis.set(
            xlim=(0.0, end_time_s or self._plot_end_time_s()),
            ylim=(-limit, limit),
            xlabel="Time (s)",
            ylabel="Angular excursion, Δδ (deg)",
        )
        self._style_axis(axis, end_time_s=end_time_s)

    def _style_axis(self, axis, *, end_time_s: float | None = None) -> None:
        axis.axvspan(self.scenario.fault_time_s, self.scenario.clearance_time_s, color="#F0B323", alpha=0.16, zorder=0)
        axis.set_xlim(0.0, end_time_s or self._plot_end_time_s())
        axis.set_facecolor("white")
        axis.grid(visible=True, color="#D6D9DC", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    def _plot_end_time_s(self) -> float:
        """Return the available simulation time, capped at ten seconds."""
        return min(self.run.dataset.maximum_time_s(), 10.0)
