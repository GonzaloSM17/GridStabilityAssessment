"""Reusable models and calculations for individual time-series signals."""

from dataclasses import dataclass
from math import isfinite, sqrt
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
    """IDD, post-clearance area, and final-settlement criteria.

    The persistence area integrates absolute deviation from the final-voltage
    reference from 2 s to the simulation end, capped at 10 s. The final
    settlement decision uses the last one-second tail of the simulation.
    The IDD follows NTSyCS Anexo Técnico art. 3-10: it integrates the absolute
    deviation from the pre-fault voltage during the fault transient.
    """

    fault_time_s: float = 1.0
    dynamic_performance_index_end_time_s: float = 1.12
    maximum_dynamic_performance_index_pu_s: float = 0.16
    response_area_end_time_s: float = 2.0
    post_fault_persistence_start_time_s: float = 2.0
    # This is a cap: the actual endpoint is min(simulation end, this time).
    post_fault_persistence_max_end_time_s: float = 10.0
    clearance_time_s: float = 1.12
    evaluation_duration_s: float = 1.0
    final_reference_duration_s: float = 1.0
    final_settlement_duration_s: float = 1.0
    minimum_final_reference_start_time_s: float = 5.0
    minimum_simulation_end_time_s: float = 6.0
    maximum_relative_error_area_s: float = 0.06
    maximum_final_absolute_relative_error: float = 0.10
    maximum_final_peak_to_peak_relative_range: float = 0.10
    post_evaluation_voltage_threshold_pu: float = 0.8


@dataclass(frozen=True)
class StabilityAssessment:
    """IDD, post-contingency area, recovery, and final-settlement measurements."""

    pre_fault_reference_value: float
    dynamic_performance_index_pu_s: float
    is_dynamic_performance_index_compliant: bool
    recovery_area_pu_s: float
    unified_response_area_pu_s: float
    post_fault_persistence_area_pu_s: float
    post_fault_persistence_end_time_s: float
    final_reference_value: float
    relative_error_area_s: float
    post_clear_peak_absolute_relative_error: float
    is_area_within_limit: bool
    final_settlement_peak_absolute_relative_error: float
    final_settlement_peak_to_peak_relative_range: float
    is_settled: bool
    minimum_voltage_after_evaluation_pu: float
    remains_above_voltage_threshold_after_evaluation: bool
    criteria: StabilityCriteria


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

    samples = [(float(time), float(value)) for time, value in zip(time_values, signal_values)]
    if len(samples) < 2 or not all(isfinite(time) and isfinite(value) for time, value in samples):
        raise ValueError("Signal analysis requires finite time and signal values.")
    if any(right[0] <= left[0] for left, right in zip(samples, samples[1:])):
        raise ValueError("Time samples must be strictly increasing.")
    sample_step = median(right[0] - left[0] for left, right in zip(samples, samples[1:]))
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
        raise ValueError("The smoothing window must fit inside the evaluation interval.")

    values = np.asarray([value for _, value in evaluation_samples], dtype=float)
    trend = np.convolve(
        values,
        np.full(smoothing_samples, 1.0 / smoothing_samples),
        mode="valid",
    )
    edge_samples = smoothing_samples // 2
    fast_relative_errors = np.abs(
        values[edge_samples:-edge_samples] - trend
    ) / abs(reference_value)
    fast_relative_error_percentile_99 = float(
        np.quantile(fast_relative_errors, 0.99)
    )
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


def assess_stability(
    time_values: Sequence[float],
    signal_values: Sequence[float],
    criteria: StabilityCriteria | None = None,
    pre_fault_reference_value: float | None = None,
) -> StabilityAssessment:
    """Assess IDD, recovery, and final settling for one voltage signal.

    IDD uses the supplied pre-fault RMS reference across the fault transient.
    The post-clearance area uses the final-voltage reference across the first
    second after clearance. The final reference is the median in the final
    reference window. A signal is evaluable only when it reaches the requested
    minimum simulation duration. This keeps an early simulation stop from being
    mistaken for a settled result.
    """
    criteria = criteria or StabilityCriteria()
    if (
        criteria.fault_time_s < 0
        or criteria.dynamic_performance_index_end_time_s <= criteria.fault_time_s
    ):
        raise ValueError("The IDD interval must start at the fault and have positive duration.")
    if (
        criteria.response_area_end_time_s
        <= criteria.dynamic_performance_index_end_time_s
    ):
        raise ValueError("The recovery-area interval must have positive duration.")
    if (
        criteria.post_fault_persistence_start_time_s
        < criteria.response_area_end_time_s
        or criteria.post_fault_persistence_max_end_time_s
        <= criteria.post_fault_persistence_start_time_s
    ):
        raise ValueError(
            "The post-fault persistence interval must start after recovery."
        )
    if criteria.maximum_dynamic_performance_index_pu_s <= 0:
        raise ValueError("The IDD limit must be positive.")
    if (
        criteria.evaluation_duration_s <= 0
        or criteria.final_reference_duration_s <= 0
        or criteria.final_settlement_duration_s <= 0
    ):
        raise ValueError("Stability evaluation, reference, and settlement durations must be positive.")
    if criteria.minimum_simulation_end_time_s < criteria.clearance_time_s + criteria.evaluation_duration_s:
        raise ValueError("The minimum simulation end time must cover the evaluation window.")
    if criteria.minimum_simulation_end_time_s <= criteria.post_fault_persistence_start_time_s:
        raise ValueError(
            "The minimum simulation end time must extend past the persistence start."
        )
    if (
        criteria.minimum_final_reference_start_time_s < 0
        or criteria.minimum_simulation_end_time_s
        < criteria.minimum_final_reference_start_time_s
        + criteria.final_reference_duration_s
    ):
        raise ValueError(
            "The simulation must include the complete final-reference interval."
        )
    if criteria.maximum_relative_error_area_s <= 0:
        raise ValueError("The post-clearance area limit must be positive.")
    if criteria.maximum_final_absolute_relative_error <= 0:
        raise ValueError("The final settlement limit must be positive.")
    if criteria.maximum_final_peak_to_peak_relative_range <= 0:
        raise ValueError("The final settlement range limit must be positive.")
    if criteria.post_evaluation_voltage_threshold_pu <= 0:
        raise ValueError("The post-evaluation voltage threshold must be positive.")
    if len(time_values) != len(signal_values):
        raise ValueError("Time and signal values must have the same length.")

    samples = [(float(time), float(value)) for time, value in zip(time_values, signal_values)]
    if len(samples) < 2 or not all(isfinite(time) and isfinite(value) for time, value in samples):
        raise ValueError("Signal analysis requires finite time and signal values.")
    if any(right[0] <= left[0] for left, right in zip(samples, samples[1:])):
        raise ValueError("Time samples must be strictly increasing.")

    end_time_s = samples[-1][0]
    if end_time_s < criteria.minimum_simulation_end_time_s:
        raise ValueError("The signal ends before the minimum stability evaluation time.")

    post_fault_persistence_end_time_s = min(
        end_time_s,
        criteria.post_fault_persistence_max_end_time_s,
    )

    evaluation_end_time_s = criteria.clearance_time_s + criteria.evaluation_duration_s
    if end_time_s < evaluation_end_time_s:
        raise ValueError("The signal does not cover the post-clearance evaluation window.")

    evaluation_samples = [
        (time, value)
        for time, value in samples
        if criteria.clearance_time_s <= time <= evaluation_end_time_s
    ]
    dynamic_performance_index_samples = [
        (time, value)
        for time, value in samples
        if criteria.fault_time_s <= time <= criteria.dynamic_performance_index_end_time_s
    ]
    recovery_area_samples = [
        (time, value)
        for time, value in samples
        if (
            criteria.dynamic_performance_index_end_time_s
            <= time
            <= criteria.response_area_end_time_s
        )
    ]
    unified_response_area_samples = [
        (time, value)
        for time, value in samples
        if criteria.fault_time_s <= time <= criteria.response_area_end_time_s
    ]
    post_fault_persistence_samples = [
        (time, value)
        for time, value in samples
        if (
            criteria.post_fault_persistence_start_time_s
            <= time
            <= post_fault_persistence_end_time_s
        )
    ]
    reference_start_time_s = end_time_s - criteria.final_reference_duration_s
    if reference_start_time_s < criteria.minimum_final_reference_start_time_s:
        raise ValueError("The final-reference interval starts too early.")
    final_settlement_start_time_s = end_time_s - criteria.final_settlement_duration_s
    reference_values = [
        value
        for time, value in samples
        if reference_start_time_s <= time <= end_time_s
    ]
    final_settlement_values = [
        value
        for time, value in samples
        if final_settlement_start_time_s <= time <= end_time_s
    ]
    post_evaluation_values = [
        value for time, value in samples if time >= evaluation_end_time_s
    ]
    if (
        len(dynamic_performance_index_samples) < 2
        or len(recovery_area_samples) < 2
        or len(unified_response_area_samples) < 2
        or len(post_fault_persistence_samples) < 2
        or len(evaluation_samples) < 2
        or not reference_values
        or not final_settlement_values
        or not post_evaluation_values
    ):
        raise ValueError("No samples are available for the post-clearance assessment.")

    if pre_fault_reference_value is None:
        pre_fault_values = [
            value for time, value in samples if time < criteria.fault_time_s
        ]
        if not pre_fault_values:
            raise ValueError("No samples are available before the fault for the IDD.")
        pre_fault_reference_value = median(pre_fault_values)
    if not isfinite(pre_fault_reference_value):
        raise ValueError("The pre-fault IDD reference value must be finite.")

    final_reference_value = median(reference_values)
    if final_reference_value == 0:
        raise ValueError("The final stability reference value must not be zero.")

    import numpy as np

    dynamic_performance_index_times = np.asarray(
        [time for time, _ in dynamic_performance_index_samples], dtype=float
    )
    dynamic_performance_index_errors = np.abs(
        np.asarray(
            [value for _, value in dynamic_performance_index_samples], dtype=float
        )
        - pre_fault_reference_value
    )
    dynamic_performance_index_pu_s = float(
        np.trapezoid(dynamic_performance_index_errors, dynamic_performance_index_times)
    )
    is_dynamic_performance_index_compliant = (
        dynamic_performance_index_pu_s
        < criteria.maximum_dynamic_performance_index_pu_s
    )
    recovery_area_times = np.asarray(
        [time for time, _ in recovery_area_samples], dtype=float
    )
    recovery_area_pu_s = float(
        np.trapezoid(
            np.abs(
                np.asarray(
                    [value for _, value in recovery_area_samples], dtype=float
                )
                - pre_fault_reference_value
            ),
            recovery_area_times,
        )
    )
    unified_response_area_times = np.asarray(
        [time for time, _ in unified_response_area_samples], dtype=float
    )
    unified_response_area_pu_s = float(
        np.trapezoid(
            np.abs(
                np.asarray(
                    [value for _, value in unified_response_area_samples], dtype=float
                )
                - pre_fault_reference_value
            ),
            unified_response_area_times,
        )
    )
    post_fault_persistence_times = np.asarray(
        [time for time, _ in post_fault_persistence_samples], dtype=float
    )
    post_fault_persistence_area_pu_s = float(
        np.trapezoid(
            np.abs(
                np.asarray(
                    [value for _, value in post_fault_persistence_samples], dtype=float
                )
                - final_reference_value
            ),
            post_fault_persistence_times,
        )
    )
    evaluation_times = np.asarray([time for time, _ in evaluation_samples], dtype=float)
    evaluation_errors = np.abs(
        np.asarray([value for _, value in evaluation_samples], dtype=float)
        / final_reference_value
        - 1.0
    )
    relative_error_area_s = float(
        np.trapezoid(evaluation_errors, evaluation_times)
    )
    post_clear_peak_absolute_relative_error = max(
        evaluation_errors
    )
    is_area_within_limit = (
        relative_error_area_s <= criteria.maximum_relative_error_area_s
    )
    final_settlement_peak_absolute_relative_error = max(
        abs(value / final_reference_value - 1.0)
        for value in final_settlement_values
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
    minimum_voltage_after_evaluation_pu = min(post_evaluation_values)
    remains_above_voltage_threshold_after_evaluation = (
        minimum_voltage_after_evaluation_pu
        >= criteria.post_evaluation_voltage_threshold_pu
    )

    return StabilityAssessment(
        pre_fault_reference_value=pre_fault_reference_value,
        dynamic_performance_index_pu_s=dynamic_performance_index_pu_s,
        is_dynamic_performance_index_compliant=(
            is_dynamic_performance_index_compliant
        ),
        recovery_area_pu_s=recovery_area_pu_s,
        unified_response_area_pu_s=unified_response_area_pu_s,
        post_fault_persistence_area_pu_s=post_fault_persistence_area_pu_s,
        post_fault_persistence_end_time_s=post_fault_persistence_end_time_s,
        final_reference_value=final_reference_value,
        relative_error_area_s=relative_error_area_s,
        post_clear_peak_absolute_relative_error=post_clear_peak_absolute_relative_error,
        is_area_within_limit=is_area_within_limit,
        final_settlement_peak_absolute_relative_error=(
            final_settlement_peak_absolute_relative_error
        ),
        final_settlement_peak_to_peak_relative_range=(
            final_settlement_peak_to_peak_relative_range
        ),
        is_settled=is_settled,
        minimum_voltage_after_evaluation_pu=minimum_voltage_after_evaluation_pu,
        remains_above_voltage_threshold_after_evaluation=(
            remains_above_voltage_threshold_after_evaluation
        ),
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
    intermediate_band = (
        (frequencies >= criteria.slow_frequency_limit_hz)
        & (frequencies < criteria.fast_frequency_limit_hz)
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
