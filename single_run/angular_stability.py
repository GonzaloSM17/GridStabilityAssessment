"""Centre-of-inertia angular-excursion assessment for one simulation run."""

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import re

from data.assets import Assets
from data.run import MonteCarloRun
from stability.scenario import ContingencyScenario


ROTOR_ANGLE_PATTERN = re.compile(r"^(?:HE|HP|GEO|TER)_[^/]+/Teta_1_")
ANGULAR_LIMIT_DEGREES = 120.0
ANGULAR_DISCLAIMER = (
    "Angular excursions are referred to the centre of inertia calculated at every "
    "time step with the H·S weight from Assets. The ±120° lines are the active "
    "angular-separation reference limits."
)


def angular_excursion_limit(excursions) -> float:
    """Return a symmetric angular axis of at least ±130°, extended in 10° steps."""
    maximum_excursion = max(
        (abs(value) for excursion in excursions for value in excursion),
        default=0.0,
    )
    return float(max(130, ceil(maximum_excursion / 10.0) * 10))


@dataclass(frozen=True)
class AngularMachineAssessment:
    """One machine's H·S weight and centre-of-inertia angular excursion."""

    machine_name: str
    signal_name: str
    inertia_s: float
    nominal_power_mva: float
    inertia_weight_mva_s: float
    minimum_excursion_degrees: float
    maximum_excursion_degrees: float
    maximum_absolute_excursion_degrees: float
    first_limit_violation_time_s: float | None

    @property
    def is_within_limit(self) -> bool:
        """Return whether the full excursion remains strictly inside ±120 degrees."""
        return self.first_limit_violation_time_s is None


@dataclass(frozen=True)
class AngularStabilityAnalysis:
    """Machine excursions relative to the H·S weighted centre-of-inertia angle."""

    machines: tuple[AngularMachineAssessment, ...]
    centre_of_inertia_minimum_degrees: float
    centre_of_inertia_maximum_degrees: float
    disclaimer: str = ANGULAR_DISCLAIMER

    @property
    def is_stable(self) -> bool:
        """Return whether every published synchronous machine remains inside ±120°."""
        return all(machine.is_within_limit for machine in self.machines)


@dataclass(frozen=True)
class AngularStabilityResult:
    """Files created by the centre-of-inertia angular assessment."""

    output_directory: Path
    analysis: AngularStabilityAnalysis
    figure_paths: tuple[Path, ...]


@dataclass(frozen=True)
class AngularCalculation:
    """Full sample-by-sample COI calculation reused by single-run artefacts."""

    analysis: AngularStabilityAnalysis
    time_values: tuple[float, ...]
    centre_of_inertia: tuple[float, ...]
    excursions: tuple[tuple[float, ...], ...]


class SingleRunAngularStability:
    """Calculate H·S weighted angular excursions for every published machine angle."""

    def __init__(
        self,
        run: MonteCarloRun,
        scenario: ContingencyScenario,
        assets: Assets,
    ) -> None:
        if run.dataset is None:
            raise ValueError(f"Run {run.run_id} does not contain a Parquet result.")
        self.run = run
        self.scenario = scenario
        self.assets = assets

    def analyze(self) -> AngularStabilityAnalysis:
        """Calculate the centre of inertia and each machine excursion at every sample."""
        return self.calculate().analysis

    def calculate(self) -> AngularCalculation:
        """Return the sample-by-sample COI and machine-excursion calculation."""
        return self._calculate()

    def create(self, output_directory: str | Path) -> AngularStabilityResult:
        """Save the centre-of-inertia and individual angular-excursion figure."""
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

    def _calculate(self):
        import numpy as np

        self.run.dataset.inspect()
        signal_names = tuple(
            series.name
            for series in self.run.dataset.series
            if ROTOR_ANGLE_PATTERN.search(series.name)
        )
        if not signal_names:
            raise ValueError("No synchronous-machine rotor-angle signals were found.")
        machine_names = tuple(signal_name.split("/", 1)[0] for signal_name in signal_names)
        missing_assets = []
        weights = []
        inertias = []
        powers = []
        for machine_name in machine_names:
            try:
                inertia_s = self.assets.get_inertia_s(machine_name)
                nominal_power_mva = self.assets.get_nominal_power_mva(machine_name)
            except (KeyError, ValueError):
                missing_assets.append(machine_name)
                continue
            inertias.append(inertia_s)
            powers.append(nominal_power_mva)
            weights.append(inertia_s * nominal_power_mva)
        if missing_assets:
            names = ", ".join(sorted(set(missing_assets)))
            raise KeyError(
                "A centre-of-inertia angle requires H and S for every published "
                f"machine angle. Missing or invalid Assets entries: {names}"
            )
        data = self.run.dataset.read_series_group(signal_names)
        time_values = np.asarray(data["time"].to_list(), dtype=float)
        angles = np.asarray([data[name].to_list() for name in signal_names], dtype=float)
        weight_values = np.asarray(weights, dtype=float)
        centre_of_inertia = np.average(angles, axis=0, weights=weight_values)
        excursions = angles - centre_of_inertia
        machines = tuple(
            AngularMachineAssessment(
                machine_name=machine_name,
                signal_name=signal_name,
                inertia_s=inertia_s,
                nominal_power_mva=nominal_power_mva,
                inertia_weight_mva_s=weight,
                minimum_excursion_degrees=float(np.min(excursion)),
                maximum_excursion_degrees=float(np.max(excursion)),
                maximum_absolute_excursion_degrees=float(np.max(np.abs(excursion))),
                first_limit_violation_time_s=self._first_limit_violation_time(
                    time_values,
                    excursion,
                ),
            )
            for machine_name, signal_name, inertia_s, nominal_power_mva, weight, excursion in zip(
                machine_names,
                signal_names,
                inertias,
                powers,
                weights,
                excursions,
            )
        )
        analysis = AngularStabilityAnalysis(
            machines=machines,
            centre_of_inertia_minimum_degrees=float(np.min(centre_of_inertia)),
            centre_of_inertia_maximum_degrees=float(np.max(centre_of_inertia)),
        )
        return AngularCalculation(
            analysis,
            tuple(time_values.tolist()),
            tuple(centre_of_inertia.tolist()),
            tuple(tuple(excursion.tolist()) for excursion in excursions),
        )

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
        figure.text(0.5, 0.015, calculation.analysis.disclaimer, ha="center", va="bottom", fontsize=9, color="#404040")
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
        figure.text(0.5, 0.015, calculation.analysis.disclaimer, ha="center", va="bottom", fontsize=9, color="#404040")
        figure.subplots_adjust(left=0.085, right=0.95, bottom=0.09, top=0.91, hspace=0.30)
        return figure

    def _draw_machine_excursions(self, axis, calculation: AngularCalculation) -> None:
        import matplotlib.pyplot as plt

        colours = plt.get_cmap("tab20")
        for index, excursion in enumerate(calculation.excursions):
            axis.plot(calculation.time_values, excursion, color=colours(index % colours.N), linewidth=0.75, alpha=0.72)
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

    @staticmethod
    def _first_limit_violation_time(time_values, excursion) -> float | None:
        """Return the first instant at which the strict ±120° criterion is violated."""
        import numpy as np

        violation_indices = np.flatnonzero(
            (excursion >= ANGULAR_LIMIT_DEGREES)
            | (excursion <= -ANGULAR_LIMIT_DEGREES)
        )
        return (
            float(time_values[violation_indices[0]])
            if len(violation_indices)
            else None
        )

