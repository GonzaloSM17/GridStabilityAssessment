"""Reusable models and calculations for individual time-series signals."""

from dataclasses import dataclass
from math import isclose, isfinite, sqrt
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class InitializationCriteria:
    """RMS-voltage initialization limits.

    The signal may follow a slow pre-contingency trajectory. Initialization is
    therefore assessed from the fast residual around a locally smoothed trend.
    Relative quantities use the median across the analysis window.
    """

    evaluation_start_time_s: float = 0.5
    evaluation_end_time_s: float = 1.0
    smoothing_duration_s: float = 0.05
    maximum_fast_relative_error_percentile_99: float = 0.001


@dataclass(frozen=True)
class InitializationAssessment:
    """Pre-contingency settling metrics for one signal."""

    reference_value: float
    fast_relative_error_percentile_99: float
    fast_rms_relative_error: float
    is_well_initialized: bool
    rejection_reasons: tuple[str, ...]
    criteria: InitializationCriteria


@dataclass(frozen=True)
class StabilityCriteria:
    """Voltage recovery limits and response indicators for one contingency.

    Response indicators cover fault-to-clearance, clearance-to-recovery, and
    the complete fault-to-recovery interval. Settlement uses the final
    one-second tail after a sufficiently long simulation.
    The IDD follows NTSyCS Anexo Técnico art. 3-10: it integrates the absolute
    deviation from the pre-fault voltage during the fault transient.
    """

    fault_time_s: float = 1.0
    clearance_time_s: float = 1.12
    response_end_time_s: float = 2.0
    final_reference_duration_s: float = 1.0
    final_settlement_duration_s: float = 1.0
    minimum_final_reference_start_time_s: float = 4.0
    minimum_simulation_end_time_s: float = 5.0
    maximum_final_absolute_relative_error: float = 0.10
    maximum_final_peak_to_peak_relative_range: float = 0.10
    post_clearance_grace_period_s: float = 0.05
    crisis_voltage_threshold_pu: float = 0.70
    low_voltage_threshold_pu: float = 0.80
    maximum_cumulative_low_voltage_duration_s: float = 1.0


@dataclass(frozen=True)
class StabilityAssessment:
    """Voltage recovery criteria and supporting response measurements."""

    pre_fault_reference_value: float
    fault_on_area_pu_s: float
    recovery_area_pu_s: float
    unified_response_area_pu_s: float
    final_reference_value: float
    final_settlement_peak_absolute_relative_error: float
    final_settlement_peak_to_peak_relative_range: float
    is_settled: bool
    crisis_decision_time_s: float
    crisis_decision_duration_s: float
    crisis_decision_event: str
    is_crisis_voltage_compliant: bool
    cumulative_low_voltage_duration_s: float
    is_low_voltage_duration_compliant: bool
    criteria: StabilityCriteria

    @property
    def is_stable(self) -> bool:
        """Return the combined status of all active voltage criteria."""
        return (
            self.is_crisis_voltage_compliant
            and self.is_low_voltage_duration_compliant
            and self.is_settled
        )

    @property
    def failed_criteria(self) -> tuple[str, ...]:
        """Return every active voltage criterion that this run does not meet."""
        failures = []
        if not self.is_crisis_voltage_compliant:
            failures.append("Final recovery above 0.70 pu")
        if not self.is_low_voltage_duration_compliant:
            failures.append("Cumulative time below 0.80 pu")
        if not self.is_settled:
            failures.append("Final settlement")
        return tuple(failures)


@dataclass(frozen=True)
class FrequencyCriteria:
    """Reserved configuration for a future frequency assessment.

    Frequency is catalogued and plotted with the bus analysis, but no frequency
    compliance decision is active yet. Thresholds will be added only after the
    applicable contingency profile and installation assumptions are defined.
    """


@dataclass(frozen=True)
class FrequencyAssessment:
    """Placeholder result for a future per-run frequency assessment."""

    criteria: FrequencyCriteria
    is_evaluated: bool = False
    status: str = "Not configured"


@dataclass(frozen=True)
class SpectralCriteria:
    """Frequency bands used to describe an oscillatory signal segment."""

    slow_frequency_limit_hz: float = 10.0
    fast_frequency_limit_hz: float = 30.0


@dataclass(frozen=True)
class SpectralDiagnostic:
    """Detrended spectral-energy measurements for one signal segment."""

    dominant_frequency_hz: float
    slow_energy_ratio: float
    intermediate_energy_ratio: float
    high_energy_ratio: float
    spectral_group: str


def assess_initialization(
    time_values: Sequence[float],
    signal_values: Sequence[float],
    criteria: InitializationCriteria | None = None,
) -> InitializationAssessment:
    """Assess whether a signal settles before the contingency begins.

    A centered moving average estimates the slow trajectory. The residual is
    measured over the full evaluation window, so a slow voltage evolution is
    allowed while high-frequency initialization oscillations are rejected.
    """
    criteria = criteria or InitializationCriteria()
    if criteria.evaluation_start_time_s >= criteria.evaluation_end_time_s:
        raise ValueError("Initialization evaluation start must precede its end.")
    if criteria.smoothing_duration_s <= 0:
        raise ValueError("Initialization smoothing duration must be positive.")
    if criteria.maximum_fast_relative_error_percentile_99 <= 0:
        raise ValueError("Fast-oscillation limit must be positive.")
    if len(time_values) != len(signal_values):
        raise ValueError("Time and signal values must have the same length.")

    samples = [
        (float(time), float(value)) for time, value in zip(time_values, signal_values)
    ]
    if len(samples) < 2 or not all(
        isfinite(time) and isfinite(value) for time, value in samples
    ):
        raise ValueError("Signal analysis requires finite time and signal values.")
    if any(right[0] <= left[0] for left, right in zip(samples, samples[1:])):
        raise ValueError("Time samples must be strictly increasing.")
    sample_step = median(
        right[0] - left[0] for left, right in zip(samples, samples[1:])
    )
    # Allow one sample step because the contingency instant is excluded.
    if (
        samples[0][0] > criteria.evaluation_start_time_s + sample_step
        or samples[-1][0] < criteria.evaluation_end_time_s - sample_step
    ):
        raise ValueError("Signal does not cover the initialization window.")

    evaluation_samples = [
        (time, value)
        for time, value in samples
        if criteria.evaluation_start_time_s <= time < criteria.evaluation_end_time_s
    ]
    if len(evaluation_samples) < 4:
        raise ValueError("At least four initialization samples are required.")
    if any(
        right[0] - left[0] > 3 * sample_step
        for left, right in zip(evaluation_samples, evaluation_samples[1:])
    ):
        raise ValueError("The initialization window contains missing time samples.")
    reference_value = median(value for _, value in evaluation_samples)
    if reference_value == 0:
        raise ValueError("The initialization reference value must not be zero.")

    import numpy as np

    smoothing_samples = max(3, round(criteria.smoothing_duration_s / sample_step))
    if smoothing_samples % 2 == 0:
        smoothing_samples += 1
    if smoothing_samples >= len(evaluation_samples):
        raise ValueError(
            "The smoothing window must fit inside the evaluation interval."
        )

    values = np.asarray([value for _, value in evaluation_samples], dtype=float)
    trend = np.convolve(
        values,
        np.full(smoothing_samples, 1.0 / smoothing_samples),
        mode="valid",
    )
    edge_samples = smoothing_samples // 2
    fast_relative_errors = np.abs(values[edge_samples:-edge_samples] - trend) / abs(
        reference_value
    )
    fast_relative_error_percentile_99 = float(np.quantile(fast_relative_errors, 0.99))
    fast_rms_relative_error = float(np.sqrt(np.mean(fast_relative_errors**2)))
    rejection_reasons = []
    if (
        fast_relative_error_percentile_99
        > criteria.maximum_fast_relative_error_percentile_99
    ):
        rejection_reasons.append("fast_oscillation")
    is_well_initialized = not rejection_reasons

    return InitializationAssessment(
        reference_value=reference_value,
        fast_relative_error_percentile_99=fast_relative_error_percentile_99,
        fast_rms_relative_error=fast_rms_relative_error,
        is_well_initialized=is_well_initialized,
        rejection_reasons=tuple(rejection_reasons),
        criteria=criteria,
    )


def _samples_from_time(
    samples: Sequence[tuple[float, float]], start_time_s: float
) -> list[tuple[float, float]]:
    """Return samples from an exact start time, linearly interpolating its value."""
    if start_time_s < samples[0][0] or start_time_s > samples[-1][0]:
        raise ValueError(
            "The signal does not cover the requested assessment start time."
        )

    for index, (time, value) in enumerate(samples):
        if time == start_time_s:
            return list(samples[index:])
        if time > start_time_s:
            previous_time, previous_value = samples[index - 1]
            fraction = (start_time_s - previous_time) / (time - previous_time)
            start_value = previous_value + fraction * (value - previous_value)
            return [(start_time_s, start_value), *samples[index:]]

    return [(start_time_s, samples[-1][1])]


def _cumulative_time_below_threshold(
    samples: Sequence[tuple[float, float]], threshold: float
) -> float:
    """Measure duration below a threshold using linear crossings between samples."""
    duration_s = 0.0
    for (left_time, left_value), (right_time, right_value) in zip(samples, samples[1:]):
        interval_s = right_time - left_time
        left_is_below = left_value < threshold
        right_is_below = right_value < threshold
        if left_is_below and right_is_below:
            duration_s += interval_s
        elif left_is_below != right_is_below:
            crossing_fraction = (threshold - left_value) / (right_value - left_value)
            crossing_time = left_time + crossing_fraction * interval_s
            if left_is_below:
                duration_s += crossing_time - left_time
            else:
                duration_s += right_time - crossing_time
    return duration_s


def _final_exit_time_above_threshold(
    samples: Sequence[tuple[float, float]], threshold: float
) -> float | None:
    """Return the final exit from ``value <= threshold`` into ``value > threshold``.

    A threshold contact counts as critical.  Therefore, a later contact or
    descending crossing replaces an earlier recovery; only the final upward
    exit can establish permanent recovery. ``None`` means the signal ends in
    the critical region and never completes its final recovery.
    """
    is_critical = [value <= threshold for _, value in samples]
    if not any(is_critical):
        return samples[0][0]

    final_exit_time_s = None
    for (left_time, left_value), (right_time, right_value) in zip(samples, samples[1:]):
        left_is_critical = left_value <= threshold
        right_is_critical = right_value <= threshold
        if not left_is_critical or right_is_critical:
            continue
        if left_value == threshold:
            final_exit_time_s = left_time
        else:
            crossing_fraction = (threshold - left_value) / (right_value - left_value)
            final_exit_time_s = left_time + crossing_fraction * (right_time - left_time)

    if samples[-1][1] <= threshold:
        return None
    return final_exit_time_s


def _first_entry_time_into_critical_region(
    samples: Sequence[tuple[float, float]], threshold: float
) -> float | None:
    """Return the first post-window entry into ``value <= threshold``."""
    if samples[0][1] <= threshold:
        return samples[0][0]
    for (left_time, left_value), (right_time, right_value) in zip(samples, samples[1:]):
        if left_value > threshold and right_value <= threshold:
            if right_value == threshold:
                return right_time
            crossing_fraction = (threshold - left_value) / (right_value - left_value)
            return left_time + crossing_fraction * (right_time - left_time)
    return None


def assess_stability(
    time_values: Sequence[float],
    signal_values: Sequence[float],
    criteria: StabilityCriteria | None = None,
    pre_fault_reference_value: float | None = None,
) -> StabilityAssessment:
    """Assess three event areas, voltage recovery, and final settlement."""
    criteria = criteria or StabilityCriteria()
    if criteria.fault_time_s < 0 or criteria.clearance_time_s <= criteria.fault_time_s:
        raise ValueError("The fault-on interval must have positive duration.")
    if criteria.response_end_time_s <= criteria.clearance_time_s:
        raise ValueError("The recovery interval must have positive duration.")
    if (
        criteria.final_reference_duration_s <= 0
        or criteria.final_settlement_duration_s <= 0
    ):
        raise ValueError("Final reference and settlement durations must be positive.")
    if criteria.minimum_simulation_end_time_s < criteria.response_end_time_s:
        raise ValueError(
            "The minimum simulation end time must cover the response interval."
        )
    if (
        criteria.minimum_final_reference_start_time_s < 0
        or criteria.minimum_simulation_end_time_s
        < criteria.minimum_final_reference_start_time_s
        + criteria.final_reference_duration_s
    ):
        raise ValueError("The simulation must include the final-reference interval.")
    if criteria.maximum_final_absolute_relative_error <= 0:
        raise ValueError("The final settlement limit must be positive.")
    if criteria.maximum_final_peak_to_peak_relative_range <= 0:
        raise ValueError("The final settlement range limit must be positive.")
    if criteria.post_clearance_grace_period_s < 0:
        raise ValueError("The post-clearance grace period must not be negative.")
    if criteria.crisis_voltage_threshold_pu <= 0:
        raise ValueError("The 0.70 pu recovery threshold must be positive.")
    if criteria.low_voltage_threshold_pu <= 0:
        raise ValueError("The 0.80 pu threshold must be positive.")
    if criteria.maximum_cumulative_low_voltage_duration_s < 0:
        raise ValueError(
            "The cumulative low-voltage duration limit must not be negative."
        )
    if len(time_values) != len(signal_values):
        raise ValueError("Time and signal values must have the same length.")

    samples = [
        (float(time), float(value)) for time, value in zip(time_values, signal_values)
    ]
    if len(samples) < 2 or not all(
        isfinite(time) and isfinite(value) for time, value in samples
    ):
        raise ValueError("Signal analysis requires finite time and signal values.")
    if any(right[0] <= left[0] for left, right in zip(samples, samples[1:])):
        raise ValueError("Time samples must be strictly increasing.")

    end_time_s = samples[-1][0]
    if end_time_s < criteria.minimum_simulation_end_time_s:
        raise ValueError(
            "The signal ends before the minimum stability evaluation time."
        )
    crisis_deadline_time_s = (
        criteria.clearance_time_s + criteria.post_clearance_grace_period_s
    )
    if end_time_s < criteria.clearance_time_s:
        raise ValueError("The signal ends before the 0.70 pu recovery evaluation time.")

    def interval(start_time_s: float, end_time_s: float) -> list[tuple[float, float]]:
        values = _samples_from_time(samples, start_time_s)
        result = [sample for sample in values if sample[0] < end_time_s]
        for time, value in values:
            if time == end_time_s:
                result.append((time, value))
                break
            if time > end_time_s:
                previous_time, previous_value = result[-1]
                fraction = (end_time_s - previous_time) / (time - previous_time)
                result.append(
                    (end_time_s, previous_value + fraction * (value - previous_value))
                )
                break
        else:
            if values[-1][0] == end_time_s:
                result.append(values[-1])
        return result

    fault_on_samples = interval(criteria.fault_time_s, criteria.clearance_time_s)
    recovery_samples = interval(criteria.clearance_time_s, criteria.response_end_time_s)
    unified_samples = interval(criteria.fault_time_s, criteria.response_end_time_s)
    reference_start_time_s = end_time_s - criteria.final_reference_duration_s
    if reference_start_time_s < criteria.minimum_final_reference_start_time_s:
        raise ValueError("The final-reference interval starts too early.")
    final_settlement_start_time_s = end_time_s - criteria.final_settlement_duration_s
    reference_values = [
        value for time, value in samples if reference_start_time_s <= time <= end_time_s
    ]
    final_settlement_values = [
        value
        for time, value in samples
        if final_settlement_start_time_s <= time <= end_time_s
    ]
    low_voltage_samples = _samples_from_time(samples, criteria.fault_time_s)
    if (
        len(fault_on_samples) < 2
        or len(recovery_samples) < 2
        or len(unified_samples) < 2
        or not reference_values
        or not final_settlement_values
        or len(low_voltage_samples) < 2
    ):
        raise ValueError("No samples are available for the voltage assessment.")

    if pre_fault_reference_value is None:
        pre_fault_values = [
            value for time, value in samples if time < criteria.fault_time_s
        ]
        if not pre_fault_values:
            raise ValueError("No samples are available before the fault.")
        pre_fault_reference_value = median(pre_fault_values)
    if not isfinite(pre_fault_reference_value) or pre_fault_reference_value == 0:
        raise ValueError("The pre-fault reference value must be finite and nonzero.")

    final_reference_value = median(reference_values)
    if final_reference_value == 0:
        raise ValueError("The final stability reference value must not be zero.")

    import numpy as np

    def response_area(segment: Sequence[tuple[float, float]]) -> float:
        times = np.asarray([time for time, _ in segment], dtype=float)
        errors = np.abs(
            np.asarray([value for _, value in segment], dtype=float)
            - pre_fault_reference_value
        )
        return float(np.trapezoid(errors, times))

    fault_on_area_pu_s = response_area(fault_on_samples)
    recovery_area_pu_s = response_area(recovery_samples)
    unified_response_area_pu_s = response_area(unified_samples)
    final_settlement_peak_absolute_relative_error = max(
        abs(value / final_reference_value - 1.0) for value in final_settlement_values
    )
    final_settlement_peak_to_peak_relative_range = (
        max(final_settlement_values) - min(final_settlement_values)
    ) / abs(final_reference_value)
    is_settled = (
        final_settlement_peak_absolute_relative_error
        <= criteria.maximum_final_absolute_relative_error
        and final_settlement_peak_to_peak_relative_range
        <= criteria.maximum_final_peak_to_peak_relative_range
    )
    crisis_recovery_samples = interval(
        criteria.clearance_time_s,
        crisis_deadline_time_s,
    )
    final_recovery_time_s = _final_exit_time_above_threshold(
        crisis_recovery_samples,
        criteria.crisis_voltage_threshold_pu,
    )
    if final_recovery_time_s is None or isclose(
        final_recovery_time_s,
        crisis_deadline_time_s,
        abs_tol=1e-9,
    ):
        crisis_decision_time_s = crisis_deadline_time_s
        crisis_decision_event = "deadline_violation"
        is_crisis_voltage_compliant = False
    else:
        post_deadline_samples = _samples_from_time(samples, crisis_deadline_time_s)
        late_critical_entry_time_s = _first_entry_time_into_critical_region(
            post_deadline_samples,
            criteria.crisis_voltage_threshold_pu,
        )
        if late_critical_entry_time_s is not None:
            crisis_decision_time_s = late_critical_entry_time_s
            crisis_decision_event = "late_critical_entry"
            is_crisis_voltage_compliant = False
        else:
            crisis_decision_time_s = final_recovery_time_s
            is_critical_during_recovery = any(
                value <= criteria.crisis_voltage_threshold_pu
                for _, value in crisis_recovery_samples
            )
            crisis_decision_event = (
                "recovery_crossing"
                if is_critical_during_recovery
                else "already_above_threshold"
            )
            is_crisis_voltage_compliant = True
    crisis_decision_duration_s = crisis_decision_time_s - criteria.clearance_time_s
    cumulative_low_voltage_duration_s = _cumulative_time_below_threshold(
        low_voltage_samples, criteria.low_voltage_threshold_pu
    )
    is_low_voltage_duration_compliant = (
        cumulative_low_voltage_duration_s
        <= criteria.maximum_cumulative_low_voltage_duration_s
    )
    return StabilityAssessment(
        pre_fault_reference_value=pre_fault_reference_value,
        fault_on_area_pu_s=fault_on_area_pu_s,
        recovery_area_pu_s=recovery_area_pu_s,
        unified_response_area_pu_s=unified_response_area_pu_s,
        final_reference_value=final_reference_value,
        final_settlement_peak_absolute_relative_error=final_settlement_peak_absolute_relative_error,
        final_settlement_peak_to_peak_relative_range=final_settlement_peak_to_peak_relative_range,
        is_settled=is_settled,
        crisis_decision_time_s=crisis_decision_time_s,
        crisis_decision_duration_s=crisis_decision_duration_s,
        crisis_decision_event=crisis_decision_event,
        is_crisis_voltage_compliant=is_crisis_voltage_compliant,
        cumulative_low_voltage_duration_s=cumulative_low_voltage_duration_s,
        is_low_voltage_duration_compliant=is_low_voltage_duration_compliant,
        criteria=criteria,
    )


def analyze_spectrum(
    time_values: Sequence[float],
    signal_values: Sequence[float],
    criteria: SpectralCriteria | None = None,
) -> SpectralDiagnostic:
    """Measure detrended spectral energy and group a signal by its peak frequency."""
    import numpy as np

    criteria = criteria or SpectralCriteria()
    time = np.asarray(time_values, dtype=float)
    values = np.asarray(signal_values, dtype=float)
    if len(time) < 4:
        raise ValueError("At least four samples are required for spectral analysis.")
    if not np.all(np.isfinite(values)):
        raise ValueError("The spectral window contains non-finite signal values.")

    sample_interval_s = float(np.median(np.diff(time)))
    if sample_interval_s <= 0:
        raise ValueError("Time samples must be strictly increasing.")

    relative_time = time - time[0]
    trend = np.polyval(np.polyfit(relative_time, values, deg=1), relative_time)
    windowed_values = (values - trend) * np.hanning(len(values))
    frequencies = np.fft.rfftfreq(len(windowed_values), d=sample_interval_s)
    power = np.abs(np.fft.rfft(windowed_values)) ** 2
    positive_frequency = frequencies > 0
    total_power = float(power[positive_frequency].sum())
    if total_power == 0:
        raise ValueError("The spectral window has no oscillatory signal content.")

    slow_band = (frequencies > 0) & (frequencies < criteria.slow_frequency_limit_hz)
    intermediate_band = (frequencies >= criteria.slow_frequency_limit_hz) & (
        frequencies < criteria.fast_frequency_limit_hz
    )
    high_band = frequencies >= criteria.fast_frequency_limit_hz
    dominant_index = np.flatnonzero(positive_frequency)[
        np.argmax(power[positive_frequency])
    ]
    dominant_frequency_hz = float(frequencies[dominant_index])

    if dominant_frequency_hz < criteria.slow_frequency_limit_hz:
        spectral_group = "slow"
    elif dominant_frequency_hz < criteria.fast_frequency_limit_hz:
        spectral_group = "intermediate"
    else:
        spectral_group = "fast"

    return SpectralDiagnostic(
        dominant_frequency_hz=dominant_frequency_hz,
        slow_energy_ratio=float(power[slow_band].sum() / total_power),
        intermediate_energy_ratio=float(power[intermediate_band].sum() / total_power),
        high_energy_ratio=float(power[high_band].sum() / total_power),
        spectral_group=spectral_group,
    )
