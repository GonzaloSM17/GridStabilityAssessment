"""Monte Carlo presentation of the shared centre-of-inertia angular criterion."""

from dataclasses import dataclass
from pathlib import Path

from data.assets import Assets
from data.run import MonteCarloRun
from single_run.angular_stability import SingleRunAngularStability
from stability.angular import (
    ANGULAR_LIMIT_DEGREES,
    ANGULAR_REFERENCE_NOTE,
    AngularCalculation,
)
from stability.scenario import ContingencyScenario
from utils.figure_localization import save_figure_with_spanish_copy


@dataclass(frozen=True)
class AngularRunAssessment:
    """Angular result available for one Monte Carlo run."""

    run_id: int
    status: str
    calculation: AngularCalculation | None
    reason: str | None = None

    @property
    def critical_machine(self):
        if self.calculation is None:
            return None
        return max(
            self.calculation.analysis.machines,
            key=lambda machine: machine.maximum_absolute_excursion_degrees,
            default=None,
        )


@dataclass(frozen=True)
class AngularStudyAnalysis:
    """Angular outcomes for every selected Monte Carlo run."""

    run_assessments: tuple[AngularRunAssessment, ...]

    def assessment_for(self, run_id: int) -> AngularRunAssessment:
        for item in self.run_assessments:
            if item.run_id == run_id:
                return item
        raise KeyError(f"No angular assessment is available for run_{run_id:06d}.")

    def status_for_run(self, run_id: int) -> str:
        return self.assessment_for(run_id).status


@dataclass(frozen=True)
class AngularStabilityResult:
    """Published Monte Carlo angular figures and their source analysis."""

    output_directory: Path
    analysis: AngularStudyAnalysis
    figure_paths: tuple[Path, ...]


class MonteCarloAngularStability:
    """Evaluate and summarize COI-referenced angular stability across runs."""

    def __init__(
        self,
        runs: tuple[MonteCarloRun, ...],
        scenario: ContingencyScenario,
        assets: Assets,
    ) -> None:
        self.runs = tuple(runs)
        self.scenario = scenario
        self.assets = assets

    def analyze(self) -> AngularStudyAnalysis:
        """Evaluate every run independently, retaining unavailable evidence."""
        assessments = []
        for run in self.runs:
            if run.dataset is None:
                assessments.append(
                    AngularRunAssessment(
                        run.run_id,
                        "Not evaluated",
                        None,
                        "Run has no Parquet dataset.",
                    )
                )
                continue
            try:
                calculation = SingleRunAngularStability(
                    run,
                    self.scenario,
                    self.assets,
                ).calculate()
            except (FileNotFoundError, KeyError, ValueError) as error:
                assessments.append(
                    AngularRunAssessment(run.run_id, "Not evaluated", None, str(error))
                )
                continue
            assessments.append(
                AngularRunAssessment(
                    run.run_id,
                    "Stable" if calculation.analysis.is_stable else "Unstable",
                    calculation,
                )
            )
        return AngularStudyAnalysis(tuple(assessments))

    def create(self, output_directory: str | Path) -> AngularStabilityResult:
        """Save compact evidence for the run-level angular conclusion."""
        import matplotlib.pyplot as plt

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        analysis = self.analyze()
        figures = (
            ("01_coi_angle_envelope.png", self._plot_coi_angle_envelope(analysis)),
            (
                "02_critical_angular_excursions.png",
                self._plot_critical_excursions(analysis),
            ),
            (
                "03_angular_instability_by_machine.png",
                self._plot_instability_by_machine(analysis),
            ),
        )
        paths = []
        for filename, figure in figures:
            path = output_path / filename
            save_figure_with_spanish_copy(figure, path, dpi=150)
            plt.close(figure)
            paths.append(path)
        return AngularStabilityResult(output_path, analysis, tuple(paths))

    def _plot_coi_angle_envelope(self, analysis: AngularStudyAnalysis):
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(13.5, 5.6))
        available = [item for item in analysis.run_assessments if item.calculation]
        colors = {"Stable": "#0072B2", "Unstable": "#D55E00"}
        for item in available:
            calculation = item.calculation
            time, values = self._plot_window(
                calculation.time_values,
                calculation.centre_of_inertia,
            )
            axis.plot(time, values, color=colors[item.status], alpha=0.16, linewidth=0.7)
        self._draw_event_context(axis)
        axis.set(
            xlim=(0.0, self._maximum_plot_time(available)),
            xlabel="Time (s)",
            ylabel="COI angle (deg)",
            title="Centre-of-Inertia Angle Across Monte Carlo Runs",
        )
        axis.grid(alpha=0.28)
        self._add_status_legend(axis, available)
        figure.text(0.5, 0.015, ANGULAR_REFERENCE_NOTE, ha="center", va="bottom", fontsize=8)
        figure.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.90)
        return figure

    def _plot_critical_excursions(self, analysis: AngularStudyAnalysis):
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(13.5, 6.0))
        colors = {"Stable": "#0072B2", "Unstable": "#D55E00", "Not evaluated": "#777777"}
        for status in colors:
            selected = [
                item
                for item in analysis.run_assessments
                if item.status == status and item.critical_machine is not None
            ]
            if selected:
                axis.scatter(
                    [item.run_id for item in selected],
                    [item.critical_machine.maximum_absolute_excursion_degrees for item in selected],
                    color=colors[status],
                    s=28,
                    alpha=0.80,
                    label=status,
                )
        axis.axhline(
            ANGULAR_LIMIT_DEGREES,
            color="#D73027",
            linestyle="--",
            linewidth=1.2,
            label="Angular limit (120 deg)",
        )
        axis.set(
            xlabel="Monte Carlo run",
            ylabel="Maximum absolute angular excursion (deg)",
            title="Critical COI-Referenced Angular Excursion by Run",
        )
        axis.grid(alpha=0.28)
        axis.legend(loc="upper left", framealpha=0.92)
        figure.text(
            0.5,
            0.015,
            "Each point is the largest absolute machine excursion in that run. "
            "A value at or above 120 deg is angularly unstable.",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        figure.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.90)
        return figure

    def _plot_instability_by_machine(self, analysis: AngularStudyAnalysis):
        import matplotlib.pyplot as plt
        import numpy as np

        figure, axis = plt.subplots(figsize=(13.5, 6.0))
        unstable = [
            item
            for item in analysis.run_assessments
            if item.status == "Unstable" and item.critical_machine is not None
        ]
        if not unstable:
            axis.text(0.5, 0.5, "No angularly unstable runs.", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            return figure
        machine_names = sorted({item.critical_machine.machine_name for item in unstable})
        positions = {name: index for index, name in enumerate(machine_names)}
        offsets = np.linspace(-0.16, 0.16, max(len(unstable), 2))
        for index, item in enumerate(unstable):
            machine = item.critical_machine
            axis.scatter(
                positions[machine.machine_name] + offsets[index % len(offsets)],
                machine.maximum_absolute_excursion_degrees,
                color="#D55E00",
                s=30,
                alpha=0.80,
            )
        axis.axhline(ANGULAR_LIMIT_DEGREES, color="#D73027", linestyle="--", linewidth=1.2)
        axis.set(
            xticks=range(len(machine_names)),
            xticklabels=machine_names,
            ylabel="Maximum absolute angular excursion (deg)",
            title="Critical Machine for Angularly Unstable Runs",
        )
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        axis.grid(axis="y", alpha=0.28)
        figure.text(0.5, 0.015, "Each point is one unstable run, grouped by its critical machine.", ha="center", va="bottom", fontsize=8)
        figure.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.90)
        return figure

    def _draw_event_context(self, axis) -> None:
        axis.axvspan(
            self.scenario.fault_time_s,
            self.scenario.clearance_time_s,
            color="#F0B323",
            alpha=0.16,
            zorder=0,
        )

    @staticmethod
    def _plot_window(time_values, values):
        return (
            [time for time in time_values if time <= 10.0],
            [value for time, value in zip(time_values, values) if time <= 10.0],
        )

    @staticmethod
    def _maximum_plot_time(assessments) -> float:
        return min(
            10.0,
            max(item.calculation.time_values[-1] for item in assessments),
        ) if assessments else 10.0

    @staticmethod
    def _add_status_legend(axis, assessments) -> None:
        from matplotlib.lines import Line2D

        present = {item.status for item in assessments}
        handles = [
            Line2D([], [], color=color, label=status)
            for status, color in (("Stable", "#0072B2"), ("Unstable", "#D55E00"))
            if status in present
        ]
        if handles:
            axis.legend(handles=handles, loc="upper left", framealpha=0.92)
