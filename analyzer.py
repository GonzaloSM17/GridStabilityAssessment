"""Study-level signal analysis built on a Monte Carlo context."""

from dataclasses import dataclass
from math import ceil, floor, log10
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
XI_PLOT_LIMITS = (0.4, 1.1)
DESIGN_FN_RANGE_HZ = (2.0, 50.0)
DESIGN_XI_RANGE = (0.6, 1.0)
POST_CONTINGENCY_PLOT_END_TIME_S = 5.0


def _round_up_to_nice_number(value: float) -> float:
    """Round a positive value up to a 1-2-5 value at its decimal scale."""
    if value <= 0:
        raise ValueError("The value to round must be positive.")

    magnitude = 10.0 ** floor(log10(value))
    scaled_value = value / magnitude
    for rounded_value in (1.0, 2.0, 5.0, 10.0):
        if scaled_value <= rounded_value:
            return rounded_value * magnitude

    raise AssertionError("A 1-2-5 rounded value should always be found.")


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
        """Return initialized runs settled within the final-timing tolerance."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None and result.stability.is_settled
        )

    @property
    def unsettled_post_contingency_run_ids(self) -> tuple[int, ...]:
        """Return initialized runs outside the final-timing tolerance."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None and not result.stability.is_settled
        )

    @property
    def stability_not_evaluated_run_ids(self) -> tuple[int, ...]:
        """Return initialized runs that do not reach the minimum duration."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.initialization.is_well_initialized and result.stability is None
        )

    @property
    def has_stability_assessments(self) -> bool:
        """Return whether at least one run has a post-fault assessment."""
        return any(result.stability is not None for result in self.run_results)

    @property
    def above_voltage_threshold_after_evaluation_run_ids(self) -> tuple[int, ...]:
        """Return runs that remain above 0.8 pu after the area window."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None
            and result.stability.remains_above_voltage_threshold_after_evaluation
        )

    @property
    def below_voltage_threshold_after_evaluation_run_ids(self) -> tuple[int, ...]:
        """Return runs that fall below 0.8 pu after the area window."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None
            and not result.stability.remains_above_voltage_threshold_after_evaluation
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

    def plot_post_clearance_area_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Plot the first post-clearance-second area against weighted fn and xi."""
        import matplotlib.pyplot as plt
        import numpy as np

        weighted_points_by_run_id = {
            point.run_id: point for point in controller_analysis.weighted_run_points
        }
        rows = [
            (
                weighted_points_by_run_id[result.run_id],
                result.stability.relative_error_area_s * 100,
                result.stability.is_settled,
            )
            for result in signal_analysis.run_results
            if result.stability is not None
            and result.run_id in weighted_points_by_run_id
        ]
        if not rows:
            raise ValueError("No runs contain both post-clearance area and controller data.")

        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
        groups = (
            (True, "Settled", "tab:green"),
            (False, "Unsettled", "tab:red"),
        )
        dimensions = (
            ("fn_hz", "Power-weighted natural frequency, fn (Hz)", (2.0, 20.0)),
            ("xi", "Power-weighted damping ratio, xi", (0.6, 1.0)),
        )
        for axis, (attribute, label, x_limits) in zip(axes, dimensions):
            for is_settled, group_label, color in groups:
                points = [row for row in rows if row[2] == is_settled]
                axis.scatter(
                    [getattr(point, attribute) for point, _, _ in points],
                    [area for _, area, _ in points],
                    color=color,
                    s=20,
                    alpha=0.7,
                    label=group_label,
                )
            x_values = np.asarray(
                [getattr(point, attribute) for point, _, _ in rows], dtype=float
            )
            area_values = np.asarray([area for _, area, _ in rows], dtype=float)
            linear_coefficients = np.polyfit(x_values, area_values, deg=1)
            fitted_areas = np.polyval(linear_coefficients, x_values)
            residual_sum = float(np.sum((area_values - fitted_areas) ** 2))
            total_sum = float(np.sum((area_values - area_values.mean()) ** 2))
            r_squared = 1.0 - residual_sum / total_sum if total_sum else 0.0
            curve_x = np.linspace(x_values.min(), x_values.max(), 200)
            axis.plot(
                curve_x,
                np.polyval(linear_coefficients, curve_x),
                color="tab:blue",
                linewidth=1.5,
                label=f"Linear fit (R^2={r_squared:.3f})",
            )
            axis.axhline(
                10.0,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label="10 %*s reference",
            )
            axis.axhline(
                5.0,
                color="tab:orange",
                linestyle="--",
                linewidth=1.0,
                label="5 %*s reference",
            )
            axis.set_xlabel(label)
            axis.set_xlim(*x_limits)
            axis.set_ylim(bottom=0.0)
            axis.grid(visible=True, alpha=0.3)
            axis.legend(loc="upper right")

        axes[0].set_ylabel("Post-clearance relative-error area (%*s)")
        figure.suptitle(
            f"{signal_analysis.signal_name}: post-clearance area vs. weighted parameters"
        )
        figure.tight_layout()
        return figure

    def plot_fault_on_area_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Plot the area from fault inception to clearance."""
        criteria = self._stability_criteria_from(signal_analysis)
        return self._plot_pre_fault_reference_area_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
            metric_attribute="dynamic_performance_index_pu_s",
            y_label="Fault-on absolute-voltage area (pu*s)",
            title=(
                f"{signal_analysis.signal_name}: fault-on area "
                f"({criteria.fault_time_s:.2f}-{criteria.dynamic_performance_index_end_time_s:.2f} s)"
            ),
        )

    def plot_recovery_area_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Plot the area from clearance through the remainder of the first second."""
        criteria = self._stability_criteria_from(signal_analysis)
        return self._plot_pre_fault_reference_area_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
            metric_attribute="recovery_area_pu_s",
            y_label="Recovery absolute-voltage area (pu*s)",
            title=(
                f"{signal_analysis.signal_name}: recovery area "
                f"({criteria.dynamic_performance_index_end_time_s:.2f}-"
                f"{criteria.response_area_end_time_s:.2f} s)"
            ),
        )

    def plot_unified_response_area_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Plot the unified one-second area from fault inception."""
        criteria = self._stability_criteria_from(signal_analysis)
        return self._plot_pre_fault_reference_area_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
            metric_attribute="unified_response_area_pu_s",
            y_label="Unified response absolute-voltage area (pu*s)",
            title=(
                f"{signal_analysis.signal_name}: unified response area "
                f"({criteria.fault_time_s:.2f}-{criteria.response_area_end_time_s:.2f} s)"
            ),
        )

    def plot_post_fault_persistence_area_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Plot the separate area from 2 s to each run end, capped at 10 s."""
        criteria = self._stability_criteria_from(signal_analysis)
        return self._plot_pre_fault_reference_area_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
            metric_attribute="post_fault_persistence_area_pu_s",
            y_label="Post-fault persistence area (pu*s)",
            title=(
                f"{signal_analysis.signal_name}: post-fault persistence area "
                f"({criteria.post_fault_persistence_start_time_s:.2f} s to "
                f"min(simulation end, "
                f"{criteria.post_fault_persistence_max_end_time_s:.2f} s))"
            ),
        )

    def plot_dynamic_performance_index_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Retain the former IDD plotting API for the fault-on area."""
        return self.plot_fault_on_area_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
        )

    @staticmethod
    def _stability_criteria_from(signal_analysis: SignalStudyAnalysis) -> StabilityCriteria:
        """Return the common criteria from the first complete assessment."""
        for result in signal_analysis.run_results:
            if result.stability is not None:
                return result.stability.criteria
        raise ValueError(
            "No post-fault assessments are available. "
            "At least one run must pass initialization and reach the required duration."
        )

    def _plot_pre_fault_reference_area_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
        *,
        metric_attribute: str,
        y_label: str,
        title: str,
    ):
        """Plot one pre-fault-reference area against the weighted controls."""
        import matplotlib.pyplot as plt
        import numpy as np

        weighted_points_by_run_id = {
            point.run_id: point for point in controller_analysis.weighted_run_points
        }
        rows = [
            (
                weighted_points_by_run_id[result.run_id],
                getattr(result.stability, metric_attribute),
                result.stability.is_settled,
            )
            for result in signal_analysis.run_results
            if result.stability is not None
            and result.run_id in weighted_points_by_run_id
        ]
        if not rows:
            raise ValueError("No runs contain both area and controller data.")

        area_values = np.asarray([area for _, area, _ in rows], dtype=float)
        maximum_area = float(area_values.max())
        area_y_upper_limit = ceil(maximum_area * 10.0) / 10.0
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
        groups = (
            (True, "Settled", "tab:green"),
            (False, "Unsettled", "tab:red"),
        )
        dimensions = (
            ("fn_hz", "Power-weighted natural frequency, fn (Hz)", (2.0, 20.0)),
            ("xi", "Power-weighted damping ratio, xi", (0.6, 1.0)),
        )
        for axis, (attribute, label, x_limits) in zip(axes, dimensions):
            for is_settled, group_label, color in groups:
                points = [row for row in rows if row[2] == is_settled]
                axis.scatter(
                    [getattr(point, attribute) for point, _, _ in points],
                    [area for _, area, _ in points],
                    color=color,
                    s=20,
                    alpha=0.7,
                    label=group_label,
                )
            x_values = np.asarray(
                [getattr(point, attribute) for point, _, _ in rows], dtype=float
            )
            linear_coefficients = np.polyfit(x_values, area_values, deg=1)
            fitted_areas = np.polyval(linear_coefficients, x_values)
            residual_sum = float(np.sum((area_values - fitted_areas) ** 2))
            total_sum = float(np.sum((area_values - area_values.mean()) ** 2))
            r_squared = 1.0 - residual_sum / total_sum if total_sum else 0.0
            curve_x = np.linspace(x_values.min(), x_values.max(), 200)
            axis.plot(
                curve_x,
                np.polyval(linear_coefficients, curve_x),
                color="tab:blue",
                linewidth=1.5,
                label=f"Linear fit (R^2={r_squared:.3f})",
            )
            axis.set_xlabel(label)
            axis.set_xlim(*x_limits)
            axis.set_ylim(0.0, area_y_upper_limit)
            axis.grid(visible=True, alpha=0.3)
            axis.legend(loc="upper right")

        axes[0].set_ylabel(y_label)
        figure.suptitle(f"{title} vs. weighted parameters")
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
                label="Design range: 2-50 Hz, 0.6-1.0",
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
        initialization_criteria = initialization_criteria or InitializationCriteria()
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
            except (KeyError, ValueError):
                skipped_run_ids.append(run.run_id)
                continue

            stability = None
            if evaluate_stability and initialization.is_well_initialized:
                if time_values[-1] >= stability_criteria.minimum_simulation_end_time_s:
                    try:
                        stability = assess_stability(
                            time_values,
                            signal_values,
                            stability_criteria,
                            pre_fault_reference_value=initialization.reference_value,
                        )
                    except ValueError:
                        skipped_run_ids.append(run.run_id)
                        continue
                # An initialized but short simulation is preserved and shown as
                # not evaluated instead of being counted as a bad signal.

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
        """Plot pre-contingency signals colored by fast-oscillation severity."""
        import matplotlib.pyplot as plt
        import polars as pl
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import TwoSlopeNorm

        groups = (
            ("Well-initialized runs", set(analysis.well_initialized_run_ids)),
            (
                "Poorly initialized runs",
                set(analysis.unsettled_initialization_run_ids),
            ),
        )
        figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        criteria = (
            analysis.run_results[0].initialization.criteria
            if analysis.run_results
            else InitializationCriteria()
        )
        severity_by_run_id = {
            result.run_id: result.initialization.fast_relative_error_percentile_99
            for result in analysis.run_results
        }
        maximum_severity = max(
            severity_by_run_id.values(),
            default=criteria.maximum_fast_relative_error_percentile_99,
        )
        color_bar_maximum = _round_up_to_nice_number(
            max(
                maximum_severity,
                criteria.maximum_fast_relative_error_percentile_99 * 1.01,
            )
            * 100
        ) / 100
        color_normalization = TwoSlopeNorm(
            vmin=0.0,
            vcenter=criteria.maximum_fast_relative_error_percentile_99,
            vmax=color_bar_maximum,
        )
        color_map = plt.get_cmap("RdYlGn_r")
        for axis, (title, run_ids) in zip(axes, groups):
            axis.axvspan(
                criteria.evaluation_start_time_s,
                criteria.evaluation_end_time_s,
                color="gold",
                alpha=0.12,
                label="Fast-oscillation window",
            )
            for run in self.context.runs:
                if run.run_id not in run_ids or run.dataset is None:
                    continue

                data = run.dataset.read_series(analysis.signal_name).filter(
                    (pl.col("time") >= 0.0)
                    & (pl.col("time") < criteria.evaluation_end_time_s)
                )
                axis.plot(
                    data["time"],
                    data[analysis.signal_name],
                    color=color_map(color_normalization(severity_by_run_id[run.run_id])),
                    alpha=0.55,
                    linewidth=0.65,
                )
            axis.set_title(f"{title} ({len(run_ids)} runs)")
            axis.set_ylabel("Voltage (pu)")
            axis.set_ylim(0.9, 1.1)
            axis.grid(visible=True, alpha=0.3)
            axis.legend(loc="upper left")

        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(0.0, 1.0)
        color_bar = figure.colorbar(
            ScalarMappable(norm=color_normalization, cmap=color_map),
            ax=axes,
            pad=0.015,
        )
        color_bar.set_label(
            "Fast-oscillation P99 (% of reference voltage); "
            f"acceptance limit: "
            f"{criteria.maximum_fast_relative_error_percentile_99 * 100:.2f} %"
        )
        color_bar.set_ticks(
            (
                0.0,
                criteria.maximum_fast_relative_error_percentile_99,
                color_bar_maximum,
            )
        )
        color_bar.set_ticklabels(
            (
                "0.00 % " + chr(10) + "(best)",
                f"{criteria.maximum_fast_relative_error_percentile_99 * 100:.2f} % "
                + chr(10) + "(limit)",
                f"{color_bar_maximum * 100:.1f} % " + chr(10) + "(worst)",
            )
        )
        figure.subplots_adjust(
            left=0.09,
            right=0.78,
            bottom=0.08,
            top=0.88,
            hspace=0.14,
        )
        color_bar.ax.set_position((0.83, 0.14, 0.025, 0.70))
        rejected_run_count = len(analysis.unsettled_initialization_run_ids)
        figure.suptitle(
            f"{analysis.signal_name}: initialization assessment" + chr(10)
            + f"Fast-oscillation rejections: {rejected_run_count}"
        )
        return figure

    def plot_stability(self, analysis: SignalStudyAnalysis):
        """Plot runs according to their post-fault recovery and final settlement."""
        import matplotlib.pyplot as plt
        import polars as pl
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        passing_run_ids = {
            result.run_id
            for result in analysis.run_results
            if result.stability is not None
            and result.stability.remains_above_voltage_threshold_after_evaluation
            and result.stability.is_settled
        }
        evaluated_run_ids = {
            result.run_id
            for result in analysis.run_results
            if result.stability is not None
        }
        groups = (
            (
                "Pass 0.8 pu and final-settlement criteria",
                passing_run_ids,
            ),
            (
                "Fail at least one recovery or settlement criterion",
                evaluated_run_ids - passing_run_ids,
            ),
        )
        figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        completed_assessments = {
            result.run_id: result.stability
            for result in analysis.run_results
            if result.stability is not None
        }
        criteria = next(
            (assessment.criteria for assessment in completed_assessments.values()),
            StabilityCriteria(),
        )
        severity_by_run_id = {
            run_id: assessment.recovery_area_pu_s
            for run_id, assessment in completed_assessments.items()
        }
        maximum_severity = max(
            severity_by_run_id.values(),
            default=0.0,
        )
        color_bar_maximum = ceil(max(maximum_severity, 0.01) * 10.0) / 10.0
        color_normalization = Normalize(
            vmin=0.0,
            vmax=color_bar_maximum,
        )
        color_map = plt.get_cmap("RdYlGn_r")
        recovery_end_time_s = criteria.response_area_end_time_s

        for axis, (title, run_ids) in zip(axes, groups):
            axis.axvspan(
                criteria.clearance_time_s,
                recovery_end_time_s,
                color="gold",
                alpha=0.10,
                label="Recovery-area window",
            )
            axis.axhline(
                criteria.post_evaluation_voltage_threshold_pu,
                color="tab:orange",
                linestyle="--",
                linewidth=1.0,
                label="0.8 pu",
            )
            axis.axhline(
                0.7,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label="0.7 pu",
            )
            for run in self.context.runs:
                if run.run_id not in run_ids or run.dataset is None:
                    continue

                data = run.dataset.read_series(analysis.signal_name).filter(
                    (pl.col("time") >= 0.0)
                    & (pl.col("time") <= POST_CONTINGENCY_PLOT_END_TIME_S)
                )
                axis.plot(
                    data["time"],
                    data[analysis.signal_name],
                    color=color_map(
                        color_normalization(severity_by_run_id[run.run_id])
                    ),
                    alpha=0.55,
                    linewidth=0.65,
                )
            axis.set_title(f"{title} ({len(run_ids)} runs)")
            axis.set_ylabel("Voltage (pu)")
            axis.set_ylim(0.1, 1.3)
            axis.grid(visible=True, alpha=0.3)
            axis.legend(loc="lower right")

        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(0.0, POST_CONTINGENCY_PLOT_END_TIME_S)
        color_bar = figure.colorbar(
            ScalarMappable(norm=color_normalization, cmap=color_map),
            ax=axes,
            pad=0.015,
        )
        color_bar.set_label(
            "Recovery absolute-voltage area (pu*s)"
        )
        color_bar.set_ticks((0.0, color_bar_maximum))
        color_bar.set_ticklabels(
            (
                "0.0 pu*s " + chr(10) + "(best)",
                f"{color_bar_maximum:.1f} pu*s " + chr(10) + "(worst)",
            )
        )
        figure.subplots_adjust(
            left=0.09,
            right=0.78,
            bottom=0.08,
            top=0.88,
            hspace=0.28,
        )
        color_bar.ax.set_position((0.83, 0.14, 0.025, 0.70))
        figure.suptitle(
            f"{analysis.signal_name}: recovery and final settlement" + chr(10)
            + f"Color: recovery area from {criteria.clearance_time_s:.2f} to "
            + f"{recovery_end_time_s:.2f} s"
        )
        return figure
