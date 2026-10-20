"""Study-level signal analysis built on a Monte Carlo context."""

from dataclasses import dataclass
from pathlib import Path

from assets import Assets
from context import MonteCarloContext
from signal_analysis import (
    InitializationAssessment,
    InitializationCriteria,
    StabilityAssessment,
    StabilityCriteria,
    assess_initialization,
    assess_stability,
)


FN_PLOT_LIMITS_HZ = (0.0, 60.0)
XI_PLOT_LIMITS = (0.0, 1.1)
DESIGN_FN_RANGE_HZ = (2.0, 50.0)
DESIGN_XI_RANGE = (0.6, 1.0)


@dataclass(frozen=True)
class ControllerPoint:
    """One active machine controller represented in fn-xi space."""

    run_id: int
    generator_name: str
    fn_hz: float
    xi: float
    nominal_power_mva: float


@dataclass(frozen=True)
class WeightedControllerPoint:
    """One run represented by its nominal-power-weighted controller point."""

    run_id: int
    fn_hz: float
    xi: float
    total_nominal_power_mva: float
    unit_count: int


@dataclass(frozen=True)
class ControllerStudyAnalysis:
    """Individual and weighted controller points across a Monte Carlo study."""

    controller_points: tuple[ControllerPoint, ...]
    weighted_run_points: tuple[WeightedControllerPoint, ...]
    skipped_run_ids: tuple[int, ...]


@dataclass(frozen=True)
class MonteCarloStudyAnalysis:
    """Results produced by one complete analyzer execution."""

    controller_analysis: ControllerStudyAnalysis | None
    signal_analysis: "SignalStudyAnalysis | None"


@dataclass(frozen=True)
class RunSignalAnalysis:
    """Analysis results for one signal in one Monte Carlo run."""

    run_id: int
    initialization: InitializationAssessment
    stability: StabilityAssessment | None


@dataclass(frozen=True)
class SignalStudyAnalysis:
    """Signal analysis results across all readable runs in one study."""

    signal_name: str
    run_results: tuple[RunSignalAnalysis, ...]
    skipped_run_ids: tuple[int, ...]

    @property
    def well_initialized_run_ids(self) -> tuple[int, ...]:
        """Return runs that passed the initialization assessment."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.initialization.is_well_initialized
        )

    @property
    def unsettled_initialization_run_ids(self) -> tuple[int, ...]:
        """Return runs rejected before the stability assessment."""
        return tuple(
            result.run_id
            for result in self.run_results
            if not result.initialization.is_well_initialized
        )

    @property
    def stable_run_ids(self) -> tuple[int, ...]:
        """Return initialized runs that settled by the end of simulation."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None and result.stability.is_stable
        )

    @property
    def unsettled_post_contingency_run_ids(self) -> tuple[int, ...]:
        """Return initialized runs that did not settle by simulation end."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None and not result.stability.is_stable
        )


class MonteCarloAnalyzer:
    """Analyze a selected signal across one Monte Carlo study."""

    def __init__(
        self,
        context: MonteCarloContext,
        assets: Assets | None = None,
    ) -> None:
        self.context = context
        self.assets = assets

    @classmethod
    def from_paths(
        cls,
        context_path: str | Path | None,
        assets_path: str | Path | None = None,
    ) -> "MonteCarloAnalyzer":
        """Create an analyzer from the study and optional Assets workbook paths."""
        context = MonteCarloContext(context_path)
        assets = Assets(assets_path) if assets_path is not None else None
        return cls(context, assets)

    @classmethod
    def select_inputs(cls) -> "MonteCarloAnalyzer | None":
        """Request both input paths before loading either study input."""
        context_path = MonteCarloContext.choose_directory()
        if context_path is None:
            return None

        assets_path = Assets.choose_workbook()
        if assets_path is None:
            return None

        return cls.from_paths(context_path, assets_path)

    def analyze(
        self,
        signal_name: str | None = None,
        initialization_criteria: InitializationCriteria | None = None,
        stability_criteria: StabilityCriteria | None = None,
        include_controllers: bool = True,
    ) -> MonteCarloStudyAnalysis:
        """Run the selected study analyses and return their consolidated results.

        Controller analysis is independent of signal analysis. When a signal is
        supplied, its initialization is assessed first and stability is assessed
        only for runs that pass that initialization assessment.
        """
        controller_analysis = (
            self.analyze_controllers() if include_controllers else None
        )
        signal_analysis = (
            self.analyze_signal(
                signal_name,
                initialization_criteria=initialization_criteria,
                stability_criteria=stability_criteria,
            )
            if signal_name is not None
            else None
        )
        return MonteCarloStudyAnalysis(
            controller_analysis=controller_analysis,
            signal_analysis=signal_analysis,
        )

    def analyze_controllers(self) -> ControllerStudyAnalysis:
        """Collect individual and nominal-power-weighted controller points."""
        if self.assets is None:
            raise ValueError("Controller analysis requires an Assets workbook.")

        controller_points: list[ControllerPoint] = []
        weighted_run_points: list[WeightedControllerPoint] = []
        skipped_run_ids: list[int] = []

        for run in self.context.runs:
            try:
                controllers = run.read_active_controller_parameters(self.assets)
            except (FileNotFoundError, KeyError, ValueError):
                skipped_run_ids.append(run.run_id)
                continue

            total_nominal_power_mva = sum(
                controller.nominal_power_mva for controller in controllers
            )
            weighted_fn_hz = sum(
                controller.nominal_power_mva * controller.fn_hz
                for controller in controllers
            ) / total_nominal_power_mva
            weighted_xi = sum(
                controller.nominal_power_mva * controller.xi
                for controller in controllers
            ) / total_nominal_power_mva

            controller_points.extend(
                ControllerPoint(
                    run_id=run.run_id,
                    generator_name=controller.generator_name,
                    fn_hz=controller.fn_hz,
                    xi=controller.xi,
                    nominal_power_mva=controller.nominal_power_mva,
                )
                for controller in controllers
            )
            weighted_run_points.append(
                WeightedControllerPoint(
                    run_id=run.run_id,
                    fn_hz=weighted_fn_hz,
                    xi=weighted_xi,
                    total_nominal_power_mva=total_nominal_power_mva,
                    unit_count=len(controllers),
                )
            )

        return ControllerStudyAnalysis(
            controller_points=tuple(controller_points),
            weighted_run_points=tuple(weighted_run_points),
            skipped_run_ids=tuple(skipped_run_ids),
        )

    def plot_controller_population(self, analysis: ControllerStudyAnalysis):
        """Plot one fn-xi point per active machine across every run."""
        figure, axis = self._create_controller_figure(
            "Controller population across all active machines"
        )
        inside_points, outside_points = self._split_controller_points(
            analysis.controller_points
        )
        axis.scatter(
            [point.fn_hz for point in inside_points],
            [point.xi for point in inside_points],
            s=8,
            color="tab:blue",
            alpha=0.18,
            label="Inside design range",
        )
        axis.scatter(
            [point.fn_hz for point in outside_points],
            [point.xi for point in outside_points],
            s=12,
            color="tab:red",
            alpha=0.6,
            label="Outside design range",
        )
        axis.legend(loc="upper right")
        figure.tight_layout()
        return figure

    def plot_weighted_controllers(self, analysis: ControllerStudyAnalysis):
        """Plot one nominal-power-weighted fn-xi point per Monte Carlo run."""
        figure, axis = self._create_controller_figure(
            "Nominal-power-weighted controller point by run"
        )
        inside_points, outside_points = self._split_controller_points(
            analysis.weighted_run_points
        )
        axis.scatter(
            [point.fn_hz for point in inside_points],
            [point.xi for point in inside_points],
            s=22,
            color="tab:blue",
            alpha=0.6,
            label="Inside design range",
        )
        axis.scatter(
            [point.fn_hz for point in outside_points],
            [point.xi for point in outside_points],
            s=28,
            color="tab:red",
            alpha=0.8,
            label="Outside design range",
        )
        axis.legend(loc="upper right")
        figure.tight_layout()
        return figure

    @staticmethod
    def _split_controller_points(points):
        """Split controller points according to the requested design range."""
        inside_points = []
        outside_points = []
        for point in points:
            if (
                DESIGN_FN_RANGE_HZ[0] <= point.fn_hz <= DESIGN_FN_RANGE_HZ[1]
                and DESIGN_XI_RANGE[0] <= point.xi <= DESIGN_XI_RANGE[1]
            ):
                inside_points.append(point)
            else:
                outside_points.append(point)
        return inside_points, outside_points

    @staticmethod
    def _create_controller_figure(title: str):
        """Create the shared fn-xi figure with its design-range frame."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        figure, axis = plt.subplots(figsize=(10, 7))
        axis.add_patch(
            Rectangle(
                (DESIGN_FN_RANGE_HZ[0], DESIGN_XI_RANGE[0]),
                DESIGN_FN_RANGE_HZ[1] - DESIGN_FN_RANGE_HZ[0],
                DESIGN_XI_RANGE[1] - DESIGN_XI_RANGE[0],
                facecolor="tab:green",
                edgecolor="tab:green",
                alpha=0.12,
                linewidth=1.5,
                label="Design range: 2–50 Hz, 0.6–1.0",
            )
        )
        axis.set_xlim(*FN_PLOT_LIMITS_HZ)
        axis.set_ylim(*XI_PLOT_LIMITS)
        axis.set_xlabel("Natural frequency, fn (Hz)")
        axis.set_ylabel("Damping ratio, xi")
        axis.set_title(title)
        axis.grid(visible=True, alpha=0.3)
        return figure, axis

    def analyze_signal(
        self,
        signal_name: str,
        initialization_criteria: InitializationCriteria | None = None,
        stability_criteria: StabilityCriteria | None = None,
        evaluate_stability: bool = True,
    ) -> SignalStudyAnalysis:
        """Run initialization and post-clearance analysis for one signal.

        Stability is evaluated only when the run passes initialization. Each run's
        Parquet signal is read once for this analysis.
        """
        initialization_criteria = initialization_criteria or InitializationCriteria(
            evaluation_start_time_s=0.9,
            evaluation_end_time_s=1.0,
            reference_start_time_s=0.9,
            reference_end_time_s=1.0,
            relative_error_limit=0.01,
            maximum_out_of_tolerance_fraction=0.01,
            minimum_settled_duration_s=0.05,
        )
        stability_criteria = stability_criteria or StabilityCriteria()

        run_results: list[RunSignalAnalysis] = []
        skipped_run_ids: list[int] = []
        for run in self.context.runs:
            if run.dataset is None:
                skipped_run_ids.append(run.run_id)
                continue

            try:
                data = run.dataset.read_series(signal_name)
                time_values = data["time"].to_list()
                signal_values = data[signal_name].to_list()
                initialization = assess_initialization(
                    time_values,
                    signal_values,
                    initialization_criteria,
                )
                stability = (
                    assess_stability(time_values, signal_values, stability_criteria)
                    if evaluate_stability and initialization.is_well_initialized
                    else None
                )
            except (KeyError, ValueError):
                skipped_run_ids.append(run.run_id)
                continue

            run_results.append(
                RunSignalAnalysis(
                    run_id=run.run_id,
                    initialization=initialization,
                    stability=stability,
                )
            )

        return SignalStudyAnalysis(
            signal_name=signal_name,
            run_results=tuple(run_results),
            skipped_run_ids=tuple(skipped_run_ids),
        )

    def analyze_initialization(
        self,
        signal_name: str,
        criteria: InitializationCriteria | None = None,
    ) -> SignalStudyAnalysis:
        """Run only the initialization assessment for one signal."""
        return self.analyze_signal(
            signal_name,
            initialization_criteria=criteria,
            evaluate_stability=False,
        )

    def plot_initialization(self, analysis: SignalStudyAnalysis):
        """Plot the signal from zero to one second by initialization result."""
        import matplotlib.pyplot as plt
        import polars as pl

        groups = (
            ("Well-initialized runs", set(analysis.well_initialized_run_ids), "tab:blue"),
            (
                "Unsettled initialization runs",
                set(analysis.unsettled_initialization_run_ids),
                "tab:red",
            ),
        )
        figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        for axis, (title, run_ids, color) in zip(axes, groups):
            for run in self.context.runs:
                if run.run_id not in run_ids or run.dataset is None:
                    continue

                data = run.dataset.read_series(analysis.signal_name).filter(
                    (pl.col("time") >= 0.0) & (pl.col("time") < 1.0)
                )
                axis.plot(
                    data["time"],
                    data[analysis.signal_name],
                    color=color,
                    alpha=0.25,
                    linewidth=0.65,
                )
            axis.set_title(f"{title} ({len(run_ids)} runs)")
            axis.set_ylabel(analysis.signal_name)
            axis.grid(visible=True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        figure.suptitle(f"{analysis.signal_name}: initialization assessment")
        figure.tight_layout()
        return figure

    def plot_stability(self, analysis: SignalStudyAnalysis):
        """Plot post-clearance signals by their final-tail stability result."""
        import matplotlib.pyplot as plt
        import polars as pl

        groups = (
            ("Stable post-contingency runs", set(analysis.stable_run_ids), "tab:blue"),
            (
                "Unsettled post-contingency runs",
                set(analysis.unsettled_post_contingency_run_ids),
                "tab:red",
            ),
        )
        figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        for axis, (title, run_ids, color) in zip(axes, groups):
            for run in self.context.runs:
                if run.run_id not in run_ids or run.dataset is None:
                    continue

                data = run.dataset.read_series(analysis.signal_name).filter(
                    pl.col("time") >= 1.12
                )
                axis.plot(
                    data["time"],
                    data[analysis.signal_name],
                    color=color,
                    alpha=0.25,
                    linewidth=0.65,
                )
            axis.set_title(f"{title} ({len(run_ids)} runs)")
            axis.set_ylabel(analysis.signal_name)
            axis.grid(visible=True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        figure.suptitle(f"{analysis.signal_name}: post-contingency assessment")
        figure.tight_layout()
        return figure
