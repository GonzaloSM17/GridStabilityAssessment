"""Monte Carlo population analysis and figure generation."""

from dataclasses import dataclass
from math import sqrt
from math import ceil, floor, log10
from stability.voltage import (
    StabilityAssessment,
    StabilityCriteria,
    assess_stability,
)
from stability.initialization import (
    InitializationAssessment,
    InitializationCriteria,
    assess_initialization,
)
from data.signals import (
    Measurement,
    VOLTAGE_RMS,
    order_power_transfers_north_to_south,
    transfer_north_to_south_order_key,
)
from stability.scenario import (
    DEFAULT_CONTINGENCY_SCENARIO,
    ContingencyScenario,
    EvaluationProfile,
)
from montecarlo.study import MonteCarloStudy
from montecarlo.angular_stability import AngularStudyAnalysis
from stability.dynamic_stability import (
    DynamicStabilityAssessment,
    assess_monte_carlo_stability,
)
from stability.power_transfer import (
    DampingCriteria,
    HighFrequencyCriteria,
    HighFrequencyStudyAnalysis,
    TransferStudyAnalysis,
    assess_high_frequency_components,
    assess_transfer_damping,
    isolate_mode,
    spectral_power_spectrum,
)
from stability.plotting import close_y_axis, label_damping_reference_limits


FN_PLOT_LIMITS_HZ = (0.0, 60.0)
XI_PLOT_LIMITS = (0.5, 1.1)
DESIGN_FN_RANGE_HZ = (2.0, 50.0)
DESIGN_XI_RANGE = (0.6, 1.0)
BANDWIDTH_RANGE_HZ = (5.0, 10.0)
POST_CONTINGENCY_PLOT_END_TIME_S = 5.0


def natural_frequency_from_bandwidth(bandwidth_hz: float, xi: float) -> float:
    """Return fn from bandwidth and xi using the selected bandwidth relation."""
    if bandwidth_hz <= 0.0:
        raise ValueError("Bandwidth must be positive.")
    if xi < 0.0:
        raise ValueError("Damping ratio xi cannot be negative.")

    bandwidth_ratio = sqrt(
        1.0
        + 2.0 * xi**2
        + sqrt((1.0 + 2.0 * xi**2) ** 2 + 1.0)
    )
    return bandwidth_hz / bandwidth_ratio


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


def _severity_color_map():
    """Return the shared low-to-high voltage-response severity color scale."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "voltage_response_severity",
        ("#009E73", "#F0B323", "#D73027"),
    )


def _response_area_scale(area_by_run_id: dict[int, float]) -> tuple[float, float, float]:
    """Return rounded response-area bounds and a legible tick step."""
    areas = tuple(area_by_run_id.values())
    if not areas:
        return 0.0, 1.0, 0.2

    minimum_area = min(areas)
    maximum_area = max(areas)
    if minimum_area == maximum_area:
        step = _round_up_to_nice_number(maximum_area / 5.0) if maximum_area else 0.2
        return minimum_area - step, maximum_area + step, step

    step = _round_up_to_nice_number((maximum_area - minimum_area) / 5.0)
    lower_limit = floor(minimum_area / step) * step
    upper_limit = ceil(maximum_area / step) * step
    return lower_limit, upper_limit, step


def _response_area_normalization(area_by_run_id: dict[int, float]):
    """Normalize response areas across rounded study limits based on their span."""
    from matplotlib.colors import Normalize

    lower_limit, upper_limit, _ = _response_area_scale(area_by_run_id)
    return Normalize(vmin=lower_limit, vmax=upper_limit, clip=True)


def _response_area_color_scale(area_by_run_id: dict[int, float]):
    """Return rounded display limits with saturated observed severity extremes."""
    from matplotlib.colors import LinearSegmentedColormap

    normalization = _response_area_normalization(area_by_run_id)
    areas = tuple(area_by_run_id.values())
    if not areas or min(areas) == max(areas):
        return normalization, _severity_color_map()
    # Keep rounded limits and ticks, while assigning full green/red to the
    # actual best/worst response inside those limits.
    minimum_position = normalization(min(areas))
    maximum_position = normalization(max(areas))
    midpoint = (minimum_position + maximum_position) / 2.0
    color_map = LinearSegmentedColormap.from_list(
        "voltage_response_severity",
        (
            (0.0, "#009E73"),
            (minimum_position, "#009E73"),
            (midpoint, "#F0B323"),
            (maximum_position, "#D73027"),
            (1.0, "#D73027"),
        ),
    )
    return normalization, color_map


def _response_area_colorbar_ticks(area_by_run_id: dict[int, float]) -> tuple[float, ...]:
    """Return every explicit, rounded tick shown on a response-area colorbar."""
    lower_limit, upper_limit, step = _response_area_scale(area_by_run_id)
    tick_count = round((upper_limit - lower_limit) / step)
    return tuple(lower_limit + step * index for index in range(tick_count + 1))


def _response_area_colorbar_label(area_by_run_id: dict[int, float]) -> str:
    """Describe both display limits and the observed values behind one colorbar."""
    lower_limit, upper_limit, _ = _response_area_scale(area_by_run_id)
    areas = tuple(area_by_run_id.values())
    observed_range = "no assessed runs"
    if areas:
        observed_range = f"{min(areas):.3f}–{max(areas):.3f}"
    return (
        "Unified voltage-response area, 1.00-2.00 s (pu*s)" + chr(10)
        + f"Scale: {lower_limit:.3f}–{upper_limit:.3f}; observed: {observed_range}"
    )


def _damping_color_scale(damping_ratios: tuple[float, ...]):
    """Return a semantic damping scale with the 5 percent limit visible."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    maximum_ratio = max((*damping_ratios, 0.05))
    upper_ratio = _round_up_to_nice_number(maximum_ratio * 100.0) / 100.0
    normalization = Normalize(vmin=0.0, vmax=upper_ratio, clip=True)
    limit_position = min(0.05 / upper_ratio, 1.0)
    color_map = LinearSegmentedColormap.from_list(
        "transfer_damping",
        (
            (0.0, "#D73027"),
            (limit_position * 0.8, "#E67E22"),
            (limit_position, "#E6C229"),
            (1.0, "#009E73"),
        ),
    )
    return normalization, color_map


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
    skipped_run_reasons: tuple[tuple[int, str], ...] = ()


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
    frequency_signal_name: str | None = None

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
        """Return runs that meet every active voltage-stability criterion."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None and result.stability.is_stable
        )

    @property
    def unsettled_post_contingency_run_ids(self) -> tuple[int, ...]:
        """Return evaluated runs that fail one or more voltage criteria."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None and not result.stability.is_stable
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
    def crisis_voltage_compliant_run_ids(self) -> tuple[int, ...]:
        """Return runs that meet the 0.70 pu post-clearance criterion."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None
            and result.stability.is_crisis_voltage_compliant
        )

    @property
    def low_voltage_duration_compliant_run_ids(self) -> tuple[int, ...]:
        """Return runs within the cumulative 0.80 pu duration limit."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None
            and result.stability.is_low_voltage_duration_compliant
        )


@dataclass(frozen=True)
class BusStudyAnalysis:
    """One selected bus and its local voltage-analysis result."""

    bus_name: str
    signal_analysis: SignalStudyAnalysis


@dataclass(frozen=True)
class StudyAnalysis:
    """The selected local analyses and their consolidated run status."""

    bus_analyses: tuple[BusStudyAnalysis, ...]

    def __post_init__(self) -> None:
        if not self.bus_analyses:
            raise ValueError("A study analysis requires at least one selected bus.")
        names = [analysis.bus_name for analysis in self.bus_analyses]
        if len(names) != len(set(names)):
            raise ValueError("A bus may be selected only once per study analysis.")

    @property
    def run_ids(self) -> tuple[int, ...]:
        """Return every run represented by at least one local analysis."""
        return tuple(
            sorted(
                {
                    result.run_id
                    for bus_analysis in self.bus_analyses
                    for result in bus_analysis.signal_analysis.run_results
                }
            )
        )

    @property
    def stable_run_ids(self) -> tuple[int, ...]:
        """Return runs that pass every selected local analysis."""
        return tuple(
            run_id for run_id in self.run_ids if self.status_for_run(run_id) == "Stable"
        )

    def status_for_run(self, run_id: int) -> str:
        """Return the consolidated status across all selected buses for one run."""
        results = self._results_for_run(run_id)
        if any(result is None or result.stability is None for _, result in results):
            return "Not evaluated"
        if any(not result.stability.is_stable for _, result in results):
            return "Unstable"
        return "Stable"

    def reasons_for_run(self, run_id: int) -> tuple[str, ...]:
        """Return the selected local reasons behind a non-stable run status."""
        reasons = []
        for bus_name, result in self._results_for_run(run_id):
            if result is None:
                reasons.append(f"{bus_name}: no readable signal")
            elif result.stability is None:
                reason = (
                    "initialization"
                    if not result.initialization.is_well_initialized
                    else "insufficient simulation duration"
                )
                reasons.append(f"{bus_name}: {reason}")
            elif not result.stability.is_stable:
                reasons.append(
                    f"{bus_name}: {', '.join(result.stability.failed_criteria)}"
                )
        return tuple(reasons)

    def _results_for_run(self, run_id: int):
        """Pair each selected bus with its local result for one run."""
        return tuple(
            (
                bus_analysis.bus_name,
                next(
                    (
                        result
                        for result in bus_analysis.signal_analysis.run_results
                        if result.run_id == run_id
                    ),
                    None,
                ),
            )
            for bus_analysis in self.bus_analyses
        )


@dataclass(frozen=True)
class ContingencyDynamicStabilityAnalysis:
    """Combined dynamic-stability assessments for one defined contingency."""

    run_assessments: tuple[DynamicStabilityAssessment, ...]

    def assessment_for(self, run_id: int) -> DynamicStabilityAssessment:
        return next(item for item in self.run_assessments if item.run_id == run_id)


class AnalysisPipeline:
    """Execute reusable analyses for one fully configured Monte Carlo study."""

    def __init__(
        self,
        study: MonteCarloStudy,
        scenario: ContingencyScenario | None = None,
        profile: EvaluationProfile | None = None,
    ) -> None:
        if scenario is not None and profile is not None:
            raise ValueError("Provide either a scenario or an evaluation profile, not both.")
        self.study = study
        self.profile = profile or EvaluationProfile.for_scenario(
            scenario or DEFAULT_CONTINGENCY_SCENARIO
        )

    @property
    def scenario(self) -> ContingencyScenario:
        """Return the contingency scenario used by every pipeline operation."""
        return self.profile.scenario

    @property
    def context(self):
        """Expose the study context to existing analysis and plot operations."""
        return self.study.context

    @property
    def assets(self):
        """Expose the study Assets metadata to controller analysis."""
        return self.study.assets

    def analyze_controllers(self) -> ControllerStudyAnalysis:
        """Collect individual and nominal-power-weighted controller points."""
        if self.assets is None:
            raise ValueError("Controller analysis requires an Assets workbook.")

        controller_points: list[ControllerPoint] = []
        weighted_run_points: list[WeightedControllerPoint] = []
        skipped_run_ids: list[int] = []
        skipped_run_reasons: list[tuple[int, str]] = []

        for run in self.context.runs:
            try:
                controllers = run.read_active_controller_parameters(self.assets)
            except (FileNotFoundError, KeyError, ValueError) as error:
                skipped_run_ids.append(run.run_id)
                skipped_run_reasons.append((run.run_id, str(error)))
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
            skipped_run_reasons=tuple(skipped_run_reasons),
        )

    def analyze_power_transfers(
        self,
        criteria: DampingCriteria | None = None,
    ) -> TransferStudyAnalysis:
        """Assess every explicitly published active-power transfer in every run."""
        criteria = criteria or self.profile.transfer_damping_criteria
        transfers = order_power_transfers_north_to_south(self.study.power_transfers)
        columns = tuple(transfer.column_name for transfer in transfers)
        assessments = []
        for run in self.context.runs:
            if run.dataset is None:
                continue
            try:
                data = run.dataset.read_series_group(columns)
            except (KeyError, ValueError):
                continue
            time_values = data["time"].to_list()
            for transfer in transfers:
                try:
                    assessments.append(
                        assess_transfer_damping(
                            time_values,
                            data[transfer.column_name].to_list(),
                            run_id=run.run_id,
                            transfer_name=transfer.name,
                            criteria=criteria,
                        )
                    )
                except ValueError:
                    continue
        return TransferStudyAnalysis(tuple(assessments))

    @staticmethod
    def analyze_contingency_dynamic_stability(
        study_analysis: StudyAnalysis,
        transfer_analysis: TransferStudyAnalysis,
        angular_analysis: AngularStudyAnalysis | None = None,
    ) -> ContingencyDynamicStabilityAnalysis:
        """Combine voltage, active-power damping, and angular stability."""
        assessments = []
        for run_id in study_analysis.run_ids:
            voltage_status = study_analysis.status_for_run(run_id)
            modal_damping_status = transfer_analysis.modal_damping_status_for_run(run_id)
            angular_status = (
                angular_analysis.status_for_run(run_id)
                if angular_analysis is not None
                else None
            )
            assessments.append(
                assess_monte_carlo_stability(
                    run_id,
                    voltage_status,
                    modal_damping_status,
                    angular_status,
                )
            )
        return ContingencyDynamicStabilityAnalysis(tuple(assessments))

    def detect_high_frequency_transfer_components(
        self,
        criteria: HighFrequencyCriteria | None = None,
    ) -> HighFrequencyStudyAnalysis:
        """Observe spectral transfer content above the EM damping band.

        This is deliberately separate from ``analyze_power_transfers``: no
        damping value or stability classification is calculated here.
        """
        criteria = criteria or self.profile.high_frequency_criteria
        transfers = order_power_transfers_north_to_south(self.study.power_transfers)
        columns = tuple(transfer.column_name for transfer in transfers)
        assessments = []
        for run in self.context.runs:
            if run.dataset is None:
                continue
            try:
                data = run.dataset.read_series_group(columns)
            except (KeyError, ValueError):
                continue
            time_values = data["time"].to_list()
            for transfer in transfers:
                try:
                    assessments.append(
                        assess_high_frequency_components(
                            time_values,
                            data[transfer.column_name].to_list(),
                            run_id=run.run_id,
                            transfer_name=transfer.name,
                            criteria=criteria,
                        )
                    )
                except ValueError:
                    continue
        return HighFrequencyStudyAnalysis(tuple(assessments))

    def plot_power_transfer_overview(
        self,
        *,
        start_time_s: float = 0.9,
        end_time_s: float | None = None,
        transfer_analysis: TransferStudyAnalysis | None = None,
    ):
        """Plot every published active-power transfer over a common time window.

        This is intentionally an observational view: individual Monte Carlo
        runs are shown lightly and their pointwise median is emphasized. No
        transfer stability or damping decision is made here.
        """
        end_time_s = self._transfer_plot_end_time(start_time_s, end_time_s)
        transfers = order_power_transfers_north_to_south(self.study.power_transfers)
        if not transfers:
            raise ValueError("No PQ_* active-power transfer monitors were found.")

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D

        columns = tuple(transfer.column_name for transfer in transfers)
        column_values = {column: [] for column in columns}
        time_values = None
        for run in self.context.runs:
            if run.dataset is None:
                continue
            try:
                data = run.dataset.read_series_window(
                    columns,
                    start_time_s,
                    end_time_s,
                )
            except (KeyError, ValueError):
                continue
            current_time_values = data["time"].to_numpy()
            if time_values is None:
                time_values = current_time_values
            if (
                time_values is None
                or len(current_time_values) != len(time_values)
                or not np.array_equal(current_time_values, time_values)
            ):
                continue
            for column in columns:
                column_values[column].append((run.run_id, data[column].to_numpy()))

        if time_values is None:
            raise ValueError("No transfer data covers the selected time window.")

        column_count = 3
        row_count = ceil(len(transfers) / column_count)
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(18, 3.7 * row_count),
            sharex=True,
        )
        axes = np.asarray(axes).reshape(-1)
        criteria = self.profile.voltage_criteria
        assessment_by_key = {
            (assessment.run_id, assessment.transfer_name): assessment
            for assessment in (transfer_analysis.assessments if transfer_analysis else ())
        }
        damping_ratios = tuple(
            assessment.damping_ratio
            for assessment in assessment_by_key.values()
            if assessment.damping_ratio is not None
        )
        damping_normalization, damping_color_map = _damping_color_scale(damping_ratios)
        for axis, transfer in zip(axes, transfers):
            axis.axvspan(
                criteria.fault_time_s,
                criteria.clearance_time_s,
                color="gold",
                alpha=0.20,
            )
            axis.axvspan(
                criteria.clearance_time_s,
                criteria.response_end_time_s,
                color="gold",
                alpha=0.10,
            )
            values = column_values[transfer.column_name]
            for run_id, series_values in values:
                assessment = assessment_by_key.get((run_id, transfer.name))
                color = "#4C78A8"
                if assessment is not None:
                    if assessment.damping_ratio is not None:
                        color = damping_color_map(
                            damping_normalization(assessment.damping_ratio)
                        )
                    elif assessment.status in {
                        "opened",
                        "inactive_pre_fault",
                        "not_measurable",
                    }:
                        color = "#9E9E9E"
                axis.plot(
                    time_values,
                    series_values,
                    color=color,
                    alpha=0.055,
                    linewidth=0.55,
                )
            if values:
                axis.plot(
                    time_values,
                    np.median(np.asarray([series for _, series in values]), axis=0),
                    color="#1F3A5F",
                    linewidth=1.4,
                )
            axis.set_title(self._display_transfer_name(transfer.name), fontsize=9)
            axis.grid(visible=True, alpha=0.3)
            axis.set_xlim(start_time_s, end_time_s)
            close_y_axis(axis)

        for axis in axes[len(transfers):]:
            axis.set_visible(False)
        for axis in axes[::column_count]:
            axis.set_ylabel("Active power")
        for axis in axes[-column_count:]:
            if axis.get_visible():
                axis.set_xlabel("Time (s)")
        legend_handles = [
            Line2D((0,), (0,), color="#1F3A5F", linewidth=1.4, label="Pointwise median"),
            Line2D((0,), (0,), color="gold", linewidth=7, alpha=0.35, label="Contingency / recovery"),
        ]
        if transfer_analysis is None:
            legend_handles.insert(
                0,
                Line2D((0,), (0,), color="#4C78A8", alpha=0.45, label="Monte Carlo run"),
            )
        else:
            legend_handles[0:0] = [
                Line2D((0,), (0,), color="#D73027", alpha=0.65, label="Growing or damped under"),
                Line2D((0,), (0,), color="#009E73", alpha=0.65, label="Damped (xi at least 5%)"),
                Line2D((0,), (0,), color="#9E9E9E", alpha=0.65, label="Not measurable / not applicable"),
            ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
        )
        figure.suptitle(
            f"500 kV transmission-system active-power transfers "
            f"({start_time_s:.1f}-{end_time_s:.1f} s)"
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.98,
            bottom=0.12,
            top=0.94,
            hspace=0.36,
            wspace=0.22,
        )
        return figure

    def plot_dominant_mode_damping_summary(self, analysis: TransferStudyAnalysis):
        """Plot the selected damping of the greatest-spectral-power EM mode."""
        records = [
            (assessment.transfer_name, assessment.dominant_mode)
            for assessment in analysis.assessments
            if assessment.dominant_damping_ratio is not None
        ]
        return self._plot_em_mode_damping(
            analysis,
            records,
            "Dominant Electromechanical-Mode Damping",
        )

    def plot_all_em_mode_damping_summary(self, analysis: TransferStudyAnalysis):
        """Plot the selected damping of every measurable EM mode."""
        records = [
            (assessment.transfer_name, mode)
            for assessment in analysis.assessments
            for mode in assessment.modes
            if mode.damping_ratio is not None
        ]
        return self._plot_em_mode_damping(
            analysis,
            records,
            "All Electromechanical-Mode Damping",
        )

    def _plot_em_mode_damping(
        self,
        analysis: TransferStudyAnalysis,
        records,
        title: str,
    ):
        """Plot selected modal damping with frequency encoded by one color scale."""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        transfer_names = self._ordered_transfer_names(analysis)
        if not transfer_names:
            raise ValueError("No active-power transfer modal-damping assessments are available.")
        damping_ratios = tuple(mode.damping_ratio for _, mode in records)
        if not damping_ratios:
            raise ValueError("No transfer contains identifiable electromechanical modes.")
        evaluated_transfer_names = tuple(
            transfer_name
            for transfer_name in transfer_names
            if any(name == transfer_name for name, _ in records)
        )
        # Keep the two damping-distribution figures compact: transfer labels are
        # multiline, but the previous height left excessive space between rows.
        figure, axis = plt.subplots(
            figsize=(12, max(4.2, 0.60 * len(evaluated_transfer_names) + 2.0))
        )
        reference_criteria = next(
            assessment.criteria
            for assessment in analysis.assessments
            if assessment.modes
        )
        frequency_normalization = Normalize(
            vmin=reference_criteria.mode_minimum_frequency_hz,
            vmax=reference_criteria.mode_maximum_frequency_hz,
        )
        frequency_color_map = plt.get_cmap("viridis")
        labels = []
        for index, transfer_name in enumerate(evaluated_transfer_names):
            evaluated = [mode for name, mode in records if name == transfer_name]
            jitter = np.linspace(-0.22, 0.22, len(evaluated)) if evaluated else ()
            axis.scatter(
                [mode.damping_ratio * 100.0 for mode in evaluated],
                [index + offset for offset in jitter],
                c=[frequency_color_map(frequency_normalization(mode.frequency_hz)) for mode in evaluated],
                s=20,
                alpha=0.78,
                edgecolors="none",
            )
            labels.append(
                f"{self._display_transfer_name(transfer_name)}\n{len(evaluated)} modes"
            )
        axis.axvline(0.0, color="#555555", linestyle="-", linewidth=1.0)
        axis.axvline(5.0, color="#D73027", linestyle="--", linewidth=1.2)
        axis.set_yticks(range(len(evaluated_transfer_names)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Damping ratio, xi (%)", fontsize=10, labelpad=7)
        axis.set_ylabel("500 kV Transmission System", fontsize=10, labelpad=10)
        axis.tick_params(axis="x", labelsize=9)
        axis.tick_params(axis="y", labelsize=9)
        lower_bound = min(0.0, min(damping_ratios) * 100.0)
        largest_positive_ratio = max(0.0, max(damping_ratios) * 100.0)
        upper_bound = (
            max(5.0, _round_up_to_nice_number(largest_positive_ratio))
            if largest_positive_ratio > 0.0
            else 5.0
        )
        axis.set_xlim(lower_bound, upper_bound)
        axis.grid(axis="x", visible=True, alpha=0.3)
        label_damping_reference_limits(axis)
        color_bar = figure.colorbar(
            ScalarMappable(norm=frequency_normalization, cmap=frequency_color_map),
            ax=axis,
            pad=0.02,
        )
        color_bar.set_label("EM mode frequency (Hz)")
        excluded_transfer_count = len(transfer_names) - len(evaluated_transfer_names)
        excluded_run_count = sum(
            item.status in {
                "opened",
                "inactive_pre_fault",
                "not_measurable",
            }
            for item in analysis.assessments
        )
        if excluded_transfer_count:
            figure.text(
                0.5,
                0.01,
                f"{excluded_transfer_count} transfer(s), {excluded_run_count} run assessment(s) "
                "excluded from xi distribution: opened, inactive, or not measurable. "
                "See Transfer Assessments in the report.",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        figure.suptitle(title, y=0.975)
        # Explicit bounds use the canvas more efficiently than tight_layout for
        # this long categorical axis and its adjacent frequency colour bar.
        figure.subplots_adjust(
            left=0.23,
            right=0.89,
            bottom=0.12 if excluded_transfer_count else 0.10,
            top=0.91,
        )
        return figure

    def plot_worst_transfer_damping_cases(
        self,
        analysis: TransferStudyAnalysis,
        *,
        start_time_s: float = 1.12,
        end_time_s: float | None = None,
    ):
        """Plot the limiting measured EM mode per transmission transfer."""
        import matplotlib.pyplot as plt
        import numpy as np

        end_time_s = self._transfer_plot_end_time(start_time_s, end_time_s)
        lowest_assessments = []
        for transfer_name in self._ordered_transfer_names(analysis):
            assessed = [
                item
                for item in analysis.assessments_for_transfer(transfer_name)
                if item.critical_mode is not None
            ]
            if assessed:
                lowest_assessments.append(min(assessed, key=lambda item: item.damping_ratio))
        if not lowest_assessments:
            raise ValueError("No transfer contains identifiable electromechanical modes.")

        transfer_by_name = {transfer.name: transfer for transfer in self.study.power_transfers}
        run_by_id = {run.run_id: run for run in self.context.runs}
        columns = 3
        rows = ceil(len(lowest_assessments) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(18, 3.9 * rows), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        for axis, assessment in zip(axes, lowest_assessments):
            transfer = transfer_by_name[assessment.transfer_name]
            run = run_by_id[assessment.run_id]
            data = run.dataset.read_series_window(
                (transfer.column_name,), start_time_s, end_time_s
            )
            mode = assessment.critical_mode
            pair = mode.selected_pair
            axis.plot(data["time"], data[transfer.column_name], color="#4C78A8", linewidth=0.8)
            axis.axhline(assessment.final_power, color="#1F3A5F", linestyle="--", linewidth=1.0)
            peak_values = [
                assessment.final_power + pair.first_value,
                assessment.final_power + pair.second_value,
            ]
            axis.scatter(
                (pair.first_time_s, pair.second_time_s),
                peak_values,
                color="#D73027",
                s=28,
                zorder=3,
            )
            axis.set_title(
                f"{self._display_transfer_name(assessment.transfer_name)} | run_{assessment.run_id:05d} | "
                f"f_mode={mode.frequency_hz:.2f} Hz | "
                f"xi={assessment.damping_ratio * 100:.2f}%",
                fontsize=8.5,
                pad=8,
            )
            axis.grid(visible=True, alpha=0.3)
            axis.set_xlim(start_time_s, end_time_s)
            close_y_axis(axis)
        for axis in axes[len(lowest_assessments):]:
            axis.set_visible(False)
        for axis in axes[::columns]:
            axis.set_ylabel("Active power")
        for axis in axes[-columns:]:
            if axis.get_visible():
                axis.set_xlabel("Time (s)")
        figure.suptitle(
            "Lowest Measured Electromechanical-Mode Damping by 500 kV Transmission System",
            y=0.985,
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.985,
            bottom=0.07,
            top=0.92,
            hspace=0.48,
            wspace=0.22,
        )
        return figure

    def plot_worst_transfer_electromechanical_cases(
        self,
        analysis: TransferStudyAnalysis,
    ):
        """Plot isolated EM components for the same transfer cases shown in figure 03."""
        import matplotlib.pyplot as plt
        import numpy as np

        lowest_assessments = []
        for transfer_name in self._ordered_transfer_names(analysis):
            assessed = [
                item
                for item in analysis.assessments_for_transfer(transfer_name)
                if item.critical_mode is not None
            ]
            if assessed:
                lowest_assessments.append(min(assessed, key=lambda item: item.damping_ratio))
        if not lowest_assessments:
            raise ValueError("No transfer contains identifiable electromechanical modes.")

        transfer_by_name = {transfer.name: transfer for transfer in self.study.power_transfers}
        run_by_id = {run.run_id: run for run in self.context.runs}
        columns = 3
        rows = ceil(len(lowest_assessments) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(18, 3.9 * rows), sharex=False)
        axes = np.asarray(axes).reshape(-1)
        for axis, assessment in zip(axes, lowest_assessments):
            transfer = transfer_by_name[assessment.transfer_name]
            run = run_by_id[assessment.run_id]
            mode = assessment.critical_mode
            pair = mode.selected_pair
            start_time_s = assessment.criteria.analysis_start_time_s
            end_time_s = run.dataset.maximum_time_s()
            data = run.dataset.read_series_window(
                (transfer.column_name,), start_time_s, end_time_s
            )
            samples = list(zip(
                data["time"].to_list(),
                (data[transfer.column_name] - assessment.final_power).to_list(),
            ))
            isolated = isolate_mode(
                samples,
                mode.lower_band_hz,
                mode.upper_band_hz,
            )
            if isolated is None:
                axis.text(
                    0.5,
                    0.5,
                    "Mode could not be isolated.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            else:
                time, response = zip(*isolated)
                axis.plot(time, response, color="#5B3F8C", linewidth=1.1)
                axis.scatter(
                    (pair.first_time_s, pair.second_time_s),
                    (pair.first_value, pair.second_value),
                    color="#D73027",
                    s=28,
                    zorder=3,
                )
            axis.axhline(0.0, color="#555555", linewidth=0.75)
            axis.set_title(
                f"{self._display_transfer_name(assessment.transfer_name)} | run_{assessment.run_id:05d} | "
                f"f_mode={mode.frequency_hz:.2f} Hz | "
                f"xi={assessment.damping_ratio * 100:.2f}%",
                fontsize=8.5,
                pad=8,
            )
            axis.grid(visible=True, alpha=0.3)
            axis.set_xlim(start_time_s, end_time_s)
            close_y_axis(axis)
        for axis in axes[len(lowest_assessments):]:
            axis.set_visible(False)
        for axis in axes[::columns]:
            if axis.get_visible():
                axis.set_ylabel("P(t) - P_final")
        for axis in axes:
            if axis.get_visible():
                axis.set_xlabel("Time (s)")
        figure.suptitle(
            "Isolated EM Components for Worst Active-Power Transfer Modal-Damping Cases",
            y=0.985,
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.985,
            bottom=0.07,
            top=0.92,
            hspace=0.48,
            wspace=0.22,
        )
        return figure

    def plot_worst_transfer_mode_decomposition(
        self,
        analysis: TransferStudyAnalysis,
    ):
        """Overlay every detected mode for the same limiting cases shown in figure 04."""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        lowest_assessments = []
        for transfer_name in self._ordered_transfer_names(analysis):
            assessed = [
                item
                for item in analysis.assessments_for_transfer(transfer_name)
                if item.critical_mode is not None
            ]
            if assessed:
                lowest_assessments.append(min(assessed, key=lambda item: item.damping_ratio))
        if not lowest_assessments:
            raise ValueError("No transfer contains identifiable electromechanical modes.")

        transfer_by_name = {transfer.name: transfer for transfer in self.study.power_transfers}
        run_by_id = {run.run_id: run for run in self.context.runs}
        columns = 3
        rows = ceil(len(lowest_assessments) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(18, 3.9 * rows), sharex=False)
        axes = np.asarray(axes).reshape(-1)
        reference_criteria = lowest_assessments[0].criteria
        frequency_normalization = Normalize(
            vmin=reference_criteria.mode_minimum_frequency_hz,
            vmax=reference_criteria.mode_maximum_frequency_hz,
        )
        frequency_color_map = plt.get_cmap("viridis")

        for axis, assessment in zip(axes, lowest_assessments):
            transfer = transfer_by_name[assessment.transfer_name]
            run = run_by_id[assessment.run_id]
            start_time_s = assessment.criteria.analysis_start_time_s
            end_time_s = run.dataset.maximum_time_s()
            data = run.dataset.read_series_window(
                (transfer.column_name,), start_time_s, end_time_s
            )
            samples = list(zip(
                data["time"].to_list(),
                (data[transfer.column_name] - assessment.final_power).to_list(),
            ))
            complete_em = isolate_mode(
                samples,
                assessment.criteria.mode_minimum_frequency_hz,
                assessment.criteria.mode_maximum_frequency_hz,
            )
            if complete_em is not None:
                time, response = zip(*complete_em)
                axis.plot(
                    time,
                    response,
                    color="#2F2F2F",
                    linewidth=1.25,
                    linestyle="--",
                    alpha=0.8,
                    zorder=1,
                )

            for mode in assessment.modes:
                isolated = isolate_mode(samples, mode.lower_band_hz, mode.upper_band_hz)
                if isolated is None:
                    continue
                time, response = zip(*isolated)
                color = frequency_color_map(frequency_normalization(mode.frequency_hz))
                is_limiting_mode = mode is assessment.critical_mode
                axis.plot(
                    time,
                    response,
                    color=color,
                    linewidth=1.45 if is_limiting_mode else 0.9,
                    alpha=0.95 if is_limiting_mode else 0.7,
                    zorder=2 if is_limiting_mode else 1,
                )

            axis.axhline(0.0, color="#555555", linewidth=0.75, zorder=0)
            axis.set_title(
                f"{self._display_transfer_name(assessment.transfer_name)} | run_{assessment.run_id:05d} | "
                f"limiting xi={assessment.damping_ratio * 100:.2f}%",
                fontsize=8.5,
                pad=8,
            )
            axis.grid(visible=True, alpha=0.3)
            axis.set_xlim(start_time_s, end_time_s)
            close_y_axis(axis)

        for axis in axes[len(lowest_assessments):]:
            axis.set_visible(False)
        for axis in axes[::columns]:
            if axis.get_visible():
                axis.set_ylabel("P(t) - P_final")
        for axis in axes:
            if axis.get_visible():
                axis.set_xlabel("Time (s)")
        figure.suptitle(
            "Electromechanical-Mode Decomposition for Worst Active-Power Transfer Cases",
            y=0.985,
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.86,
            bottom=0.07,
            top=0.92,
            hspace=0.48,
            wspace=0.22,
        )
        color_bar = figure.colorbar(
            ScalarMappable(norm=frequency_normalization, cmap=frequency_color_map),
            cax=figure.add_axes((0.90, 0.14, 0.018, 0.70)),
        )
        color_bar.set_label("EM mode frequency (Hz)")
        return figure

    def plot_high_frequency_component_spectra(
        self,
        transfer_analysis: TransferStudyAnalysis,
        high_frequency_analysis: HighFrequencyStudyAnalysis,
    ):
        """Plot high-frequency spectra for the same limiting transfer cases as figure 04."""
        import matplotlib.pyplot as plt
        import numpy as np

        lowest_assessments = []
        for transfer_name in self._ordered_transfer_names(transfer_analysis):
            assessed = [
                item
                for item in transfer_analysis.assessments_for_transfer(transfer_name)
                if item.critical_mode is not None
            ]
            if assessed:
                lowest_assessments.append(min(assessed, key=lambda item: item.damping_ratio))
        if not lowest_assessments:
            raise ValueError("No transfer contains identifiable electromechanical modes.")

        transfer_by_name = {transfer.name: transfer for transfer in self.study.power_transfers}
        run_by_id = {run.run_id: run for run in self.context.runs}
        columns = 3
        rows = ceil(len(lowest_assessments) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(18, 3.9 * rows), sharex=True)
        axes = np.asarray(axes).reshape(-1)

        for axis, damping_assessment in zip(axes, lowest_assessments):
            observation = high_frequency_analysis.assessment_for(
                damping_assessment.run_id,
                damping_assessment.transfer_name,
            )
            if observation is None:
                axis.text(0.5, 0.5, "No high-frequency observation is available.", ha="center", va="center", transform=axis.transAxes)
                axis.set_visible(True)
                continue
            transfer = transfer_by_name[damping_assessment.transfer_name]
            run = run_by_id[damping_assessment.run_id]
            start_time_s = observation.criteria.analysis_start_time_s
            data = run.dataset.read_series_window(
                (transfer.column_name,),
                start_time_s,
                run.dataset.maximum_time_s(),
            )
            samples = list(zip(
                data["time"].to_list(),
                (data[transfer.column_name] - observation.final_power).to_list(),
            ))
            spectrum = spectral_power_spectrum(
                samples,
                observation.criteria.minimum_frequency_hz,
                observation.criteria.maximum_frequency_hz,
            )
            if spectrum is None:
                axis.text(0.5, 0.5, "No observable high-frequency spectrum.", ha="center", va="center", transform=axis.transAxes)
            else:
                frequencies, powers = spectrum
                axis.plot(frequencies, powers, color="#4C78A8", linewidth=0.9)
                if observation.components:
                    axis.scatter(
                        [component.frequency_hz for component in observation.components],
                        [component.spectral_power for component in observation.components],
                        color="#D73027",
                        s=14,
                        zorder=3,
                    )
                close_y_axis(axis)
            component_count = len(observation.components)
            axis.set_title(
                f"{self._display_transfer_name(damping_assessment.transfer_name)} | run_{damping_assessment.run_id:05d} | "
                f"{component_count} components",
                fontsize=8.5,
                pad=8,
            )
            axis.grid(visible=True, alpha=0.3)
            axis.set_xlim(
                observation.criteria.minimum_frequency_hz,
                observation.criteria.maximum_frequency_hz,
            )
        for axis in axes[len(lowest_assessments):]:
            axis.set_visible(False)
        for axis in axes[::columns]:
            if axis.get_visible():
                axis.set_ylabel("Spectral power")
        for axis in axes:
            if axis.get_visible():
                axis.set_xlabel("Frequency (Hz)")
        figure.suptitle(
            "High-Frequency Components for Worst Active-Power Transfer Cases",
            y=0.985,
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.985,
            bottom=0.07,
            top=0.92,
            hspace=0.48,
            wspace=0.22,
        )
        return figure

    def plot_high_frequency_component_decomposition(
        self,
        transfer_analysis: TransferStudyAnalysis,
        high_frequency_analysis: HighFrequencyStudyAnalysis,
    ):
        """Overlay detected high-frequency components without evaluating damping."""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        lowest_assessments = []
        for transfer_name in self._ordered_transfer_names(transfer_analysis):
            assessed = [
                item
                for item in transfer_analysis.assessments_for_transfer(transfer_name)
                if item.critical_mode is not None
            ]
            if assessed:
                lowest_assessments.append(min(assessed, key=lambda item: item.damping_ratio))
        if not lowest_assessments:
            raise ValueError("No transfer contains identifiable electromechanical modes.")

        transfer_by_name = {transfer.name: transfer for transfer in self.study.power_transfers}
        run_by_id = {run.run_id: run for run in self.context.runs}
        columns = 3
        rows = ceil(len(lowest_assessments) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(18, 3.9 * rows), sharex=False)
        axes = np.asarray(axes).reshape(-1)
        reference_criteria = next(
            item.criteria
            for item in high_frequency_analysis.assessments
        )
        frequency_normalization = Normalize(
            vmin=reference_criteria.minimum_frequency_hz,
            vmax=reference_criteria.maximum_frequency_hz,
        )
        frequency_color_map = plt.get_cmap("plasma")

        for axis, damping_assessment in zip(axes, lowest_assessments):
            observation = high_frequency_analysis.assessment_for(
                damping_assessment.run_id,
                damping_assessment.transfer_name,
            )
            if observation is None:
                axis.text(0.5, 0.5, "No high-frequency observation is available.", ha="center", va="center", transform=axis.transAxes)
                axis.set_visible(True)
                continue
            transfer = transfer_by_name[damping_assessment.transfer_name]
            run = run_by_id[damping_assessment.run_id]
            start_time_s = observation.criteria.analysis_start_time_s
            end_time_s = run.dataset.maximum_time_s()
            data = run.dataset.read_series_window(
                (transfer.column_name,), start_time_s, end_time_s
            )
            samples = list(zip(
                data["time"].to_list(),
                (data[transfer.column_name] - observation.final_power).to_list(),
            ))
            full_band = isolate_mode(
                samples,
                observation.criteria.minimum_frequency_hz,
                observation.criteria.maximum_frequency_hz,
            )
            if full_band is not None:
                time, response = zip(*full_band)
                axis.plot(
                    time,
                    response,
                    color="#2F2F2F",
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.75,
                )
            for component in observation.components:
                isolated = isolate_mode(
                    samples,
                    component.lower_band_hz,
                    component.upper_band_hz,
                )
                if isolated is None:
                    continue
                time, response = zip(*isolated)
                axis.plot(
                    time,
                    response,
                    color=frequency_color_map(
                        frequency_normalization(component.frequency_hz)
                    ),
                    linewidth=0.85,
                    alpha=0.8,
                )
            if not observation.components:
                axis.text(0.5, 0.5, "No detected high-frequency components.", ha="center", va="center", transform=axis.transAxes)
            axis.axhline(0.0, color="#555555", linewidth=0.75, zorder=0)
            axis.set_title(
                f"{self._display_transfer_name(damping_assessment.transfer_name)} | run_{damping_assessment.run_id:05d} | "
                f"{len(observation.components)} components",
                fontsize=8.5,
                pad=8,
            )
            axis.grid(visible=True, alpha=0.3)
            axis.set_xlim(start_time_s, end_time_s)
            close_y_axis(axis)
        for axis in axes[len(lowest_assessments):]:
            axis.set_visible(False)
        for axis in axes[::columns]:
            if axis.get_visible():
                axis.set_ylabel("P(t) - P_final")
        for axis in axes:
            if axis.get_visible():
                axis.set_xlabel("Time (s)")
        figure.suptitle(
            "High-Frequency Component Decomposition for Worst Active-Power Transfer Cases",
            y=0.985,
        )
        figure.subplots_adjust(
            left=0.06,
            right=0.86,
            bottom=0.07,
            top=0.92,
            hspace=0.48,
            wspace=0.22,
        )
        color_bar = figure.colorbar(
            ScalarMappable(norm=frequency_normalization, cmap=frequency_color_map),
            cax=figure.add_axes((0.90, 0.14, 0.018, 0.70)),
        )
        color_bar.set_label("High-frequency component (Hz)")
        return figure

    def _transfer_plot_end_time(
        self,
        start_time_s: float,
        requested_end_time_s: float | None,
    ) -> float:
        """Return a common transfer horizon, capped at five simulation seconds."""
        available_end_times = [
            run.dataset.maximum_time_s()
            for run in self.context.runs
            if run.dataset is not None
        ]
        if not available_end_times:
            raise ValueError("No transfer dataset is available for plotting.")
        end_time_s = min(5.0, min(available_end_times))
        if requested_end_time_s is not None:
            end_time_s = min(end_time_s, requested_end_time_s)
        if start_time_s >= end_time_s:
            raise ValueError("The transfer-view start must precede its end.")
        return end_time_s

    @staticmethod
    def _ordered_transfer_names(
        analysis: TransferStudyAnalysis,
    ) -> tuple[str, ...]:
        """Return transfer names in the shared north-to-south presentation order."""
        return tuple(
            sorted(
                analysis.transfer_names,
                key=transfer_north_to_south_order_key,
            )
        )

    @staticmethod
    def _display_transfer_name(name: str) -> str:
        """Remove the measurement-area prefix from transfer figure labels."""
        return name.removeprefix("NC_").removeprefix("SC_")

    def plot_controller_population(
        self,
        analysis: ControllerStudyAnalysis,
        study_analysis: StudyAnalysis | None = None,
        dynamic_stability_analysis: ContingencyDynamicStabilityAnalysis | None = None,
    ):
        """Plot all machine controllers using the global run stability status."""
        figure, axis = self._create_controller_figure(
            self._controller_figure_title(
                "Controller population across all active machines",
                study_analysis,
                dynamic_stability_analysis,
            )
        )
        stable_run_ids = self._stable_run_ids(
            study_analysis,
            dynamic_stability_analysis,
        )
        stable_points = [
            point for point in analysis.controller_points if point.run_id in stable_run_ids
        ]
        unstable_points = [
            point for point in analysis.controller_points if point.run_id not in stable_run_ids
        ]
        axis.scatter(
            [point.fn_hz for point in stable_points],
            [point.xi for point in stable_points],
            s=8,
            color="tab:blue",
            alpha=0.28,
            label="Stable",
        )
        if study_analysis is not None:
            axis.scatter(
                [point.fn_hz for point in unstable_points],
                [point.xi for point in unstable_points],
                s=8,
                color="tab:red",
                alpha=0.38,
            label="Unstable",
            )
        elif unstable_points:
            axis.scatter(
                [point.fn_hz for point in unstable_points],
                [point.xi for point in unstable_points],
                s=8,
                color="tab:blue",
                alpha=0.28,
                label="Controller point",
            )
        axis.legend(
            loc="lower right",
            bbox_to_anchor=(0.99, 0.01),
            ncol=4,
            fontsize=7,
            framealpha=0.92,
        )
        figure.tight_layout()
        return figure

    def plot_weighted_controllers(
        self,
        analysis: ControllerStudyAnalysis,
        study_analysis: StudyAnalysis | None = None,
        dynamic_stability_analysis: ContingencyDynamicStabilityAnalysis | None = None,
    ):
        """Plot weighted controllers using the global run stability status."""
        figure, axis = self._create_controller_figure(
            self._controller_figure_title(
                "Nominal-power-weighted controller point by run",
                study_analysis,
                dynamic_stability_analysis,
            )
        )
        stable_run_ids = self._stable_run_ids(
            study_analysis,
            dynamic_stability_analysis,
        )
        stable_points = [
            point for point in analysis.weighted_run_points if point.run_id in stable_run_ids
        ]
        unstable_points = [
            point for point in analysis.weighted_run_points if point.run_id not in stable_run_ids
        ]
        axis.scatter(
            [point.fn_hz for point in stable_points],
            [point.xi for point in stable_points],
            s=22,
            color="tab:blue",
            alpha=0.55,
            label="Stable",
        )
        if study_analysis is not None:
            axis.scatter(
                [point.fn_hz for point in unstable_points],
                [point.xi for point in unstable_points],
                s=22,
                color="tab:red",
                alpha=0.65,
            label="Unstable",
            )
        elif unstable_points:
            axis.scatter(
                [point.fn_hz for point in unstable_points],
                [point.xi for point in unstable_points],
                s=22,
                color="tab:blue",
                alpha=0.55,
                label="Controller point",
            )
        axis.legend(
            loc="lower right",
            bbox_to_anchor=(0.99, 0.01),
            ncol=4,
            fontsize=7,
            framealpha=0.92,
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
        return self._plot_voltage_indicator_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
            metric_attribute="fault_on_area_pu_s",
            y_label="Fault-on absolute-voltage area (pu*s)",
            title=(
                f"{signal_analysis.signal_name}: fault-on area "
                f"({criteria.fault_time_s:.2f}-{criteria.clearance_time_s:.2f} s)"
            ),
        )

    def plot_recovery_area_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Plot the area from clearance through the remainder of the first second."""
        criteria = self._stability_criteria_from(signal_analysis)
        return self._plot_voltage_indicator_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
            metric_attribute="recovery_area_pu_s",
            y_label="Recovery absolute-voltage area (pu*s)",
            title=(
                f"{signal_analysis.signal_name}: recovery area "
                f"({criteria.clearance_time_s:.2f}-"
                f"{criteria.response_end_time_s:.2f} s)"
            ),
        )

    def plot_unified_response_area_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Plot the unified one-second area from fault inception."""
        criteria = self._stability_criteria_from(signal_analysis)
        return self._plot_voltage_indicator_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
            metric_attribute="unified_response_area_pu_s",
            y_label="Unified response absolute-voltage area (pu*s)",
            title=(
                f"{signal_analysis.signal_name}: unified response area "
                f"({criteria.fault_time_s:.2f}-{criteria.response_end_time_s:.2f} s)"
            ),
        )

    def plot_crisis_decision_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Plot the threshold event that decides the 0.70 pu criterion."""
        criteria = self._stability_criteria_from(signal_analysis)
        return self._plot_voltage_indicator_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
            metric_attribute="crisis_decision_duration_s",
            y_label="0.70 pu decision time after clearance (s)",
            title=(
                f"{signal_analysis.signal_name}: 0.70 pu stability decision"
            ),
            criterion_limit=criteria.post_clearance_grace_period_s,
            criterion_label=(
                "0.70 pu deadline: "
                f"{criteria.post_clearance_grace_period_s:.3f} s"
            ),
            status_attribute="is_crisis_voltage_compliant",
            passed_label="Pass 0.70 pu recovery",
            failed_label="Fail 0.70 pu recovery",
        )

    def plot_low_voltage_duration_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
    ):
        """Plot the active cumulative 0.80 pu duration criterion."""
        criteria = self._stability_criteria_from(signal_analysis)
        return self._plot_voltage_indicator_vs_weighted_controllers(
            controller_analysis,
            signal_analysis,
            metric_attribute="cumulative_low_voltage_duration_s",
            y_label="Cumulative time below 0.80 pu (s)",
            title=(
                f"{signal_analysis.signal_name}: cumulative 0.80 pu duration "
                f"(from {criteria.fault_time_s:.2f} s)"
            ),
            criterion_limit=criteria.maximum_cumulative_low_voltage_duration_s,
            criterion_label=(
                "0.80 pu limit: "
                f"{criteria.maximum_cumulative_low_voltage_duration_s:.3f} s"
            ),
        )

    @staticmethod
    def _stability_criteria_from(signal_analysis: SignalStudyAnalysis) -> StabilityCriteria:
        """Return the common criteria from the first complete assessment."""
        for result in signal_analysis.run_results:
            if result.stability is not None:
                return result.stability.criteria
        raise ValueError(
            "No voltage-stability assessments are available. "
            "At least one run must pass initialization and reach the required duration."
        )

    def _plot_voltage_indicator_vs_weighted_controllers(
        self,
        controller_analysis: ControllerStudyAnalysis,
        signal_analysis: SignalStudyAnalysis,
        *,
        metric_attribute: str,
        y_label: str,
        title: str,
        criterion_limit: float | None = None,
        criterion_label: str | None = None,
        status_attribute: str | None = None,
        passed_label: str = "Stable",
        failed_label: str = "Unstable",
    ):
        """Plot one local voltage indicator against the weighted controls."""
        import matplotlib.pyplot as plt
        import numpy as np

        weighted_points_by_run_id = {
            point.run_id: point for point in controller_analysis.weighted_run_points
        }
        rows = []
        missing_metric_count = 0
        for result in signal_analysis.run_results:
            if result.stability is None or result.run_id not in weighted_points_by_run_id:
                continue
            metric = getattr(result.stability, metric_attribute)
            if metric is None:
                missing_metric_count += 1
                continue
            passed = (
                getattr(result.stability, status_attribute)
                if status_attribute is not None
                else result.stability.is_stable
            )
            rows.append((weighted_points_by_run_id[result.run_id], metric, passed))
        if not rows:
            raise ValueError("No runs contain both the selected indicator and controller data.")

        metric_values = np.asarray([metric for _, metric, _ in rows], dtype=float)
        maximum_metric = max(
            float(metric_values.max()),
            criterion_limit if criterion_limit is not None else 0.0,
        )
        metric_y_upper_limit = (
            ceil(maximum_metric * 1.1 * 100.0) / 100.0
            if maximum_metric <= 0.1
            else ceil(maximum_metric * 1.1 * 10.0) / 10.0
        )
        metric_y_upper_limit = max(metric_y_upper_limit, 0.01)
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
        groups = (
            (True, passed_label, "tab:blue"),
            (False, failed_label, "tab:red"),
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
                    [metric for _, metric, _ in points],
                    color=color,
                    s=20,
                    alpha=0.7,
                    label=group_label,
                )
            x_values = np.asarray(
                [getattr(point, attribute) for point, _, _ in rows], dtype=float
            )
            linear_coefficients = np.polyfit(x_values, metric_values, deg=1)
            fitted_metrics = np.polyval(linear_coefficients, x_values)
            residual_sum = float(np.sum((metric_values - fitted_metrics) ** 2))
            total_sum = float(np.sum((metric_values - metric_values.mean()) ** 2))
            r_squared = 1.0 - residual_sum / total_sum if total_sum else 0.0
            curve_x = np.linspace(x_values.min(), x_values.max(), 200)
            axis.plot(
                curve_x,
                np.polyval(linear_coefficients, curve_x),
                color="#5E3C99",
                linewidth=1.5,
                label=f"Linear fit (R^2={r_squared:.3f})",
            )
            axis.set_xlabel(label)
            axis.set_xlim(*x_limits)
            if criterion_limit is not None:
                axis.axhline(
                    criterion_limit,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.0,
                    label=criterion_label,
                )
            axis.set_ylim(0.0, metric_y_upper_limit)
            close_y_axis(axis)
            axis.grid(visible=True, alpha=0.3)
            axis.legend(loc="upper right")

        axes[0].set_ylabel(y_label)
        note = (
            f"; {missing_metric_count} run(s) did not complete final recovery"
            if missing_metric_count
            else ""
        )
        figure.suptitle(f"{title} vs. weighted parameters{note}")
        figure.tight_layout()
        return figure

    @staticmethod
    def _create_controller_figure(title: str):
        """Create the shared fn-xi figure with design and bandwidth regions."""
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
        xi_values = [
            DESIGN_XI_RANGE[0]
            + (DESIGN_XI_RANGE[1] - DESIGN_XI_RANGE[0]) * index / 100.0
            for index in range(101)
        ]
        lower_bandwidth_fn = [
            natural_frequency_from_bandwidth(BANDWIDTH_RANGE_HZ[0], xi)
            for xi in xi_values
        ]
        upper_bandwidth_fn = [
            natural_frequency_from_bandwidth(BANDWIDTH_RANGE_HZ[1], xi)
            for xi in xi_values
        ]
        axis.fill_betweenx(
            xi_values,
            lower_bandwidth_fn,
            upper_bandwidth_fn,
            color="#5E3C99",
            alpha=0.12,
            label=(
                f"Bandwidth guide: {BANDWIDTH_RANGE_HZ[0]:.2f}-"
                f"{BANDWIDTH_RANGE_HZ[1]:.2f} Hz"
            ),
        )
        axis.plot(
            [*lower_bandwidth_fn, *reversed(upper_bandwidth_fn), lower_bandwidth_fn[0]],
            [*xi_values, *reversed(xi_values), xi_values[0]],
            color="#5E3C99",
            linewidth=0.9,
        )
        axis.set_xlim(*FN_PLOT_LIMITS_HZ)
        axis.set_ylim(*XI_PLOT_LIMITS)
        axis.set_xlabel("Natural frequency, fn (Hz)")
        axis.set_ylabel("Damping ratio, xi")
        axis.set_title(title)
        axis.grid(visible=True, alpha=0.3)
        return figure, axis

    @staticmethod
    def _stable_run_ids(
        study_analysis: StudyAnalysis | None,
        dynamic_stability_analysis: ContingencyDynamicStabilityAnalysis | None = None,
    ) -> set[int]:
        """Return consolidated stable run identifiers, when available."""
        if dynamic_stability_analysis is not None:
            return {
                assessment.run_id
                for assessment in dynamic_stability_analysis.run_assessments
                if assessment.status == "Stable"
            }
        if study_analysis is None:
            return set()
        return set(study_analysis.stable_run_ids)

    @staticmethod
    def _controller_figure_title(
        title: str,
        study_analysis: StudyAnalysis | None,
        dynamic_stability_analysis: ContingencyDynamicStabilityAnalysis | None = None,
    ) -> str:
        """Identify the selected-study status used to color controller points."""
        if dynamic_stability_analysis is not None:
            return f"{title}: global dynamic-stability status"
        if study_analysis is None:
            return title
        return f"{title}: consolidated selected-analysis status"


    def analyze_measurement(
        self,
        measurement: Measurement,
        initialization_criteria: InitializationCriteria | None = None,
        stability_criteria: StabilityCriteria | None = None,
    ) -> SignalStudyAnalysis:
        """Analyze the currently supported RMS-voltage measurement.

        Stability is evaluated only when the run passes initialization. Each run's
        Parquet signal is read once for this analysis.
        """
        if measurement.kind != VOLTAGE_RMS:
            raise ValueError(
                "Only RMS-voltage measurements support the current assessment."
            )

        signal_name = measurement.column_name
        initialization_criteria = initialization_criteria or self.profile.initialization_criteria
        stability_criteria = stability_criteria or self.profile.voltage_criteria

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
            if initialization.is_well_initialized:
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
            frequency_signal_name=f"{measurement.source_name}/Frequency_Meter/f",
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
        figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex="col")
        criteria = (
            analysis.run_results[0].initialization.criteria
            if analysis.run_results
            else self.profile.initialization_criteria
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
        color_map = _severity_color_map()
        for row_index, (title, run_ids) in enumerate(groups):
            voltage_axis = axes[row_index, 0]
            frequency_axis = axes[row_index, 1]
            voltage_axis.axvspan(
                criteria.evaluation_start_time_s,
                criteria.evaluation_end_time_s,
                color="gold",
                alpha=0.12,
                label="Fast-oscillation window",
            )
            frequency_axis.axvspan(
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
                color = color_map(color_normalization(severity_by_run_id[run.run_id]))
                voltage_axis.plot(
                    data["time"],
                    data[analysis.signal_name],
                    color=color,
                    alpha=0.55,
                    linewidth=0.65,
                )
                if analysis.frequency_signal_name is None:
                    continue
                try:
                    frequency_data = run.dataset.read_series(
                        analysis.frequency_signal_name
                    ).filter(
                        (pl.col("time") >= 0.0)
                        & (pl.col("time") < criteria.evaluation_end_time_s)
                    )
                except (KeyError, ValueError):
                    continue
                frequency_axis.plot(
                    frequency_data["time"],
                    frequency_data[analysis.frequency_signal_name],
                    color=color,
                    alpha=0.55,
                    linewidth=0.65,
                )

            voltage_axis.set_title(f"{title} — voltage ({len(run_ids)} runs)")
            voltage_axis.set_ylabel("Voltage (pu)")
            voltage_axis.set_ylim(0.9, 1.5)
            voltage_axis.grid(visible=True, alpha=0.3)
            voltage_axis.legend(loc="upper left")

            frequency_axis.set_title(f"{title} — frequency ({len(run_ids)} runs)")
            frequency_axis.set_ylabel("Frequency (Hz)")
            frequency_axis.grid(visible=True, alpha=0.3)
            frequency_axis.legend(loc="upper left")

        for axis in axes[-1]:
            axis.set_xlabel("Time (s)")
        for axis in axes.flat:
            axis.set_xlim(0.0, 1.0)
        color_bar = figure.colorbar(
            ScalarMappable(norm=color_normalization, cmap=color_map),
            ax=axes.ravel().tolist(),
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
            right=0.88,
            bottom=0.08,
            top=0.88,
            hspace=0.14,
            wspace=0.22,
        )
        color_bar.ax.set_position((0.91, 0.14, 0.018, 0.70))
        rejected_run_count = len(analysis.unsettled_initialization_run_ids)
        figure.suptitle(
            f"{analysis.signal_name}: initialization assessment" + chr(10)
            + f"Fast-oscillation rejections: {rejected_run_count}"
        )
        return figure

    def plot_voltage_stability(
        self,
        analysis: SignalStudyAnalysis,
        *,
        start_time_s: float = 0.0,
        end_time_s: float = POST_CONTINGENCY_PLOT_END_TIME_S,
        title: str = "recovery and final settlement",
        legend_location: str = "lower right",
    ):
        """Plot every active criterion separately; failure groups overlap by design."""
        import matplotlib.pyplot as plt
        import polars as pl

        passing_run_ids = {
            result.run_id
            for result in analysis.run_results
            if result.stability is not None
            and result.stability.is_stable
        }
        crisis_failure_run_ids = {
            result.run_id
            for result in analysis.run_results
            if result.stability is not None
            and not result.stability.is_crisis_voltage_compliant
        }
        low_voltage_duration_failure_run_ids = {
            result.run_id
            for result in analysis.run_results
            if result.stability is not None
            and not result.stability.is_low_voltage_duration_compliant
        }
        settlement_failure_run_ids = {
            result.run_id
            for result in analysis.run_results
            if result.stability is not None and not result.stability.is_settled
        }
        groups = (
            ("Pass all active voltage criteria", passing_run_ids),
            ("Fail final recovery above 0.70 pu", crisis_failure_run_ids),
            ("Fail cumulative time below 0.80 pu", low_voltage_duration_failure_run_ids),
            ("Fail final-settlement criterion", settlement_failure_run_ids),
        )
        figure, axes = plt.subplots(2, 2, figsize=(17, 11), sharex=True, sharey=True)
        completed_assessments = {
            result.run_id: result.stability
            for result in analysis.run_results
            if result.stability is not None
        }
        criteria = next(
            (assessment.criteria for assessment in completed_assessments.values()),
            self.profile.voltage_criteria,
        )
        crisis_deadline_time_s = (
            criteria.clearance_time_s + criteria.post_clearance_grace_period_s
        )
        import matplotlib.cm as cm

        area_by_run_id = {
            run_id: assessment.unified_response_area_pu_s
            for run_id, assessment in completed_assessments.items()
        }
        normalization, color_map = _response_area_color_scale(area_by_run_id)

        for axis, (group_title, run_ids) in zip(axes.flat, groups):
            axis.axvspan(
                criteria.fault_time_s,
                criteria.clearance_time_s,
                color="gold",
                alpha=0.20,
                label="Contingency",
            )
            axis.axvspan(
                criteria.clearance_time_s,
                criteria.response_end_time_s,
                color="gold",
                alpha=0.10,
                label="Recovery window",
            )
            axis.axhline(
                criteria.low_voltage_threshold_pu,
                color="tab:orange",
                linestyle="--",
                linewidth=1.0,
                label="0.80 pu",
            )
            axis.axhline(
                criteria.crisis_voltage_threshold_pu,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label="0.70 pu",
            )
            axis.axhline(
                1.10,
                color="#4C78A8",
                linestyle="--",
                linewidth=1.0,
                label="1.10 pu",
            )
            axis.axhline(
                1.20,
                color="#6A3D9A",
                linestyle="--",
                linewidth=1.0,
                label="1.20 pu",
            )
            axis.axvline(
                crisis_deadline_time_s,
                color="tab:red",
                linestyle=":",
                linewidth=1.0,
                label="50 ms after clearance",
            )
            for run in self.context.runs:
                if run.run_id not in run_ids or run.dataset is None:
                    continue

                data = run.dataset.read_series(analysis.signal_name).filter(
                    (pl.col("time") >= start_time_s)
                    & (pl.col("time") <= end_time_s)
                )
                axis.plot(
                    data["time"],
                    data[analysis.signal_name],
                    color=color_map(normalization(area_by_run_id[run.run_id])),
                    alpha=0.55,
                    linewidth=0.65,
                )
            axis.set_title(f"{group_title} ({len(run_ids)} runs)")
            axis.set_ylabel("Voltage (pu)")
            axis.set_ylim(0.0, 1.5)
            axis.grid(visible=True, alpha=0.3)
            axis.legend(loc=legend_location, fontsize="small")

        for axis in axes[-1]:
            axis.set_xlabel("Time (s)")
        for axis in axes.flat:
            axis.set_xlim(start_time_s, end_time_s)
        figure.suptitle(
            f"{analysis.signal_name}: {title}" + chr(10)
            + "Failure panels are non-exclusive: a run can appear in more than one."
        )
        figure.subplots_adjust(
            left=0.07,
            right=0.85,
            bottom=0.08,
            top=0.87,
            hspace=0.28,
            wspace=0.18,
        )
        colorbar_axis = figure.add_axes((0.90, 0.15, 0.018, 0.66))
        colorbar = figure.colorbar(
            cm.ScalarMappable(norm=normalization, cmap=color_map),
            cax=colorbar_axis,
        )
        colorbar.set_ticks(_response_area_colorbar_ticks(area_by_run_id))
        colorbar.set_label(_response_area_colorbar_label(area_by_run_id))
        return figure

    def plot_voltage_contingency_closeup(self, analysis: SignalStudyAnalysis):
        """Plot the voltage-stability groups around the contingency interval."""
        criteria = self._stability_criteria_from(analysis)
        return self.plot_voltage_stability(
            analysis,
            start_time_s=0.9,
            end_time_s=2.2,
            title="contingency close-up",
        )

    def plot_voltage_fault_closeup(self, analysis: SignalStudyAnalysis):
        """Plot the immediate fault and clearance interval in detail."""
        return self.plot_voltage_stability(
            analysis,
            start_time_s=0.9,
            end_time_s=1.2,
            title="fault and clearance close-up",
            legend_location="lower left",
        )

    def plot_frequency_by_voltage_stability(self, analysis: SignalStudyAnalysis):
        """Plot frequency using the voltage groups and their shared severity color."""
        if analysis.frequency_signal_name is None:
            raise ValueError("The voltage analysis has no associated frequency signal.")

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import polars as pl

        passing_run_ids = {
            result.run_id
            for result in analysis.run_results
            if result.stability is not None
            and result.stability.is_stable
        }
        evaluated_run_ids = {
            result.run_id
            for result in analysis.run_results
            if result.stability is not None
        }
        groups = (
            ("Pass all voltage criteria", passing_run_ids),
            ("Fail one or more voltage criteria", evaluated_run_ids - passing_run_ids),
        )
        criteria = next(
            (
                result.stability.criteria
                for result in analysis.run_results
                if result.stability is not None
            ),
            self.profile.voltage_criteria,
        )
        area_by_run_id = {
            result.run_id: result.stability.unified_response_area_pu_s
            for result in analysis.run_results
            if result.stability is not None
        }
        normalization, color_map = _response_area_color_scale(area_by_run_id)

        figure, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        for axis, (title, run_ids) in zip(axes, groups):
            axis.axvspan(
                criteria.fault_time_s,
                criteria.clearance_time_s,
                color="gold",
                alpha=0.20,
                label="Contingency",
            )
            axis.axvspan(
                criteria.clearance_time_s,
                criteria.response_end_time_s,
                color="gold",
                alpha=0.10,
                label="Voltage recovery window",
            )
            for run in self.context.runs:
                if run.run_id not in run_ids or run.dataset is None:
                    continue
                try:
                    data = run.dataset.read_series(
                        analysis.frequency_signal_name
                    ).filter(
                        (pl.col("time") >= 0.0)
                        & (pl.col("time") <= POST_CONTINGENCY_PLOT_END_TIME_S)
                    )
                except (KeyError, ValueError):
                    continue
                axis.plot(
                    data["time"],
                    data[analysis.frequency_signal_name],
                    color=color_map(normalization(area_by_run_id[run.run_id])),
                    alpha=0.55,
                    linewidth=0.65,
                )

            axis.set_title(f"{title} ({len(run_ids)} runs)")
            axis.set_ylabel("Frequency (Hz)")
            axis.set_ylim(48.5, 51.5)
            axis.grid(visible=True, alpha=0.3)
            axis.legend(loc="lower right")

        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(0.0, POST_CONTINGENCY_PLOT_END_TIME_S)
        figure.suptitle(
            f"{analysis.frequency_signal_name}: grouped by voltage stability"
        )
        figure.subplots_adjust(
            left=0.08,
            right=0.85,
            bottom=0.08,
            top=0.91,
            hspace=0.22,
        )
        colorbar_axis = figure.add_axes((0.90, 0.18, 0.018, 0.64))
        colorbar = figure.colorbar(
            cm.ScalarMappable(norm=normalization, cmap=color_map),
            cax=colorbar_axis,
        )
        colorbar.set_ticks(_response_area_colorbar_ticks(area_by_run_id))
        colorbar.set_label(_response_area_colorbar_label(area_by_run_id))
        return figure
