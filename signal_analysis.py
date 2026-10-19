"""Reusable models and calculations for individual time-series signals."""

from dataclasses import dataclass
from math import ceil, isfinite, sqrt
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class InitializationCriteria:
    """Settling requirements for a pre-contingency signal."""

    evaluation_start_time_s: float = 0.05
    evaluation_end_time_s: float = 1.0
    reference_start_time_s: float = 0.9
    reference_end_time_s: float = 1.0
    relative_error_limit: float = 0.01
    maximum_out_of_tolerance_fraction: float = 0.01
    minimum_settled_duration_s: float = 0.1


@dataclass(frozen=True)
class InitializationAssessment:
    """Pre-contingency settling metrics for one signal."""

    reference_value: float
    maximum_absolute_relative_error: float
    percentile_95_absolute_relative_error: float
    rms_relative_error: float
    out_of_tolerance_fraction: float
    settled_from_time_s: float | None
    is_well_initialized: bool


@dataclass(frozen=True)
class StabilityCriteria:
    """Settling requirements after a contingency has been cleared."""

    clearance_time_s: float = 1.12
    tail_duration_s: float = 1.0
    reference_duration_s: float = 0.25
    relative_oscillation_limit: float = 0.01
    relative_drift_limit: float = 0.005


@dataclass(frozen=True)
class StabilityAssessment:
    """Post-contingency settling metrics for one signal."""

    final_reference_value: float
    post_clear_peak_absolute_relative_error: float
    tail_percentile_95_absolute_relative_error: float
    tail_relative_drift: float
    is_stable: bool


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

    The reference is the median value in the final pre-contingency interval. A
    signal is considered well initialized when it enters and remains within the
    configured relative-error tolerance for the required final duration.
    """
    criteria = criteria or InitializationCriteria()
    if criteria.evaluation_start_time_s >= criteria.evaluation_end_time_s:
        raise ValueError("Initialization evaluation start must precede its end.")
    if criteria.reference_start_time_s >= criteria.reference_end_time_s:
        raise ValueError("Initialization reference start must precede its end.")
    if criteria.relative_error_limit <= 0:
        raise ValueError("Initialization relative-error limit must be positive.")
    if not 0 <= criteria.maximum_out_of_tolerance_fraction <= 1:
        raise ValueError("Maximum out-of-tolerance fraction must be between zero and one.")
    if criteria.minimum_settled_duration_s <= 0:
        raise ValueError("Minimum settled duration must be positive.")
    if len(time_values) != len(signal_values):
        raise ValueError("Time and signal values must have the same length.")

    samples = [(float(time), float(value)) for time, value in zip(time_values, signal_values)]
    if len(samples) < 2 or not all(isfinite(time) and isfinite(value) for time, value in samples):
        raise ValueError("Signal analysis requires finite time and signal values.")

    evaluation_samples = [
        (time, value)
        for time, value in samples
        if criteria.evaluation_start_time_s <= time < criteria.evaluation_end_time_s
    ]
    reference_values = [
        value
        for time, value in samples
        if criteria.reference_start_time_s <= time < criteria.reference_end_time_s
    ]
    if not evaluation_samples:
        raise ValueError("No samples are available for the initialization interval.")
    if not reference_values:
        raise ValueError("No samples are available for the initialization reference.")

    reference_value = median(reference_values)
    if reference_value == 0:
        raise ValueError("The initialization reference value must not be zero.")

    errors = [abs(value / reference_value - 1.0) for _, value in evaluation_samples]
    sorted_errors = sorted(errors)
    percentile_95_index = ceil(0.95 * len(sorted_errors)) - 1
    rms_relative_error = sqrt(sum(error**2 for error in errors) / len(errors))
    out_of_tolerance_indices = [
        index
        for index, error in enumerate(errors)
        if error > criteria.relative_error_limit
    ]

    if out_of_tolerance_indices:
        next_index = out_of_tolerance_indices[-1] + 1
        settled_from_time_s = (
            evaluation_samples[next_index][0]
            if next_index < len(evaluation_samples)
            else None
        )
    else:
        settled_from_time_s = evaluation_samples[0][0]

    final_sample_time_s = evaluation_samples[-1][0]
    is_well_initialized = (
        settled_from_time_s is not None
        and final_sample_time_s - settled_from_time_s
        >= criteria.minimum_settled_duration_s
        and len(out_of_tolerance_indices) / len(errors)
        <= criteria.maximum_out_of_tolerance_fraction
    )

    return InitializationAssessment(
        reference_value=reference_value,
        maximum_absolute_relative_error=max(errors),
        percentile_95_absolute_relative_error=sorted_errors[percentile_95_index],
        rms_relative_error=rms_relative_error,
        out_of_tolerance_fraction=len(out_of_tolerance_indices) / len(errors),
        settled_from_time_s=settled_from_time_s,
        is_well_initialized=is_well_initialized,
    )


def assess_stability(
    time_values: Sequence[float],
    signal_values: Sequence[float],
    criteria: StabilityCriteria | None = None,
) -> StabilityAssessment:
    """Assess whether a post-clearance signal has settled by simulation end.

    Stability here means the voltage is both low-oscillation and low-drift in its
    final tail. The final reference permits the post-contingency operating point
    to differ from the pre-contingency voltage.
    """
    criteria = criteria or StabilityCriteria()
    if criteria.tail_duration_s <= 0 or criteria.reference_duration_s <= 0:
        raise ValueError("Stability tail and reference durations must be positive.")
    if criteria.relative_oscillation_limit <= 0 or criteria.relative_drift_limit <= 0:
        raise ValueError("Stability limits must be positive.")
    if len(time_values) != len(signal_values):
        raise ValueError("Time and signal values must have the same length.")

    samples = [(float(time), float(value)) for time, value in zip(time_values, signal_values)]
    if len(samples) < 2 or not all(isfinite(time) and isfinite(value) for time, value in samples):
        raise ValueError("Signal analysis requires finite time and signal values.")

    end_time_s = samples[-1][0]
    if end_time_s <= criteria.clearance_time_s:
        raise ValueError("The signal ends before the contingency clearance time.")
    tail_start_time_s = max(criteria.clearance_time_s, end_time_s - criteria.tail_duration_s)
    reference_start_time_s = max(
        criteria.clearance_time_s,
        end_time_s - criteria.reference_duration_s,
    )
    tail_samples = [
        (time, value) for time, value in samples if tail_start_time_s <= time <= end_time_s
    ]
    reference_values = [
        value
        for time, value in samples
        if reference_start_time_s <= time <= end_time_s
    ]
    post_clear_values = [
        value for time, value in samples if criteria.clearance_time_s <= time <= end_time_s
    ]
    if not tail_samples or not reference_values or not post_clear_values:
        raise ValueError("No samples are available for the post-clearance assessment.")

    final_reference_value = median(reference_values)
    if final_reference_value == 0:
        raise ValueError("The final stability reference value must not be zero.")

    tail_errors = [
        abs(value / final_reference_value - 1.0) for _, value in tail_samples
    ]
    sorted_tail_errors = sorted(tail_errors)
    percentile_95_index = ceil(0.95 * len(sorted_tail_errors)) - 1
    post_clear_peak_absolute_relative_error = max(
        abs(value / final_reference_value - 1.0) for value in post_clear_values
    )

    tail_midpoint_s = tail_start_time_s + (end_time_s - tail_start_time_s) / 2
    early_tail_values = [
        value for time, value in tail_samples if time < tail_midpoint_s
    ]
    late_tail_values = [
        value for time, value in tail_samples if time >= tail_midpoint_s
    ]
    if not early_tail_values or not late_tail_values:
        raise ValueError("The stability tail must contain two non-empty halves.")
    tail_relative_drift = abs(median(late_tail_values) - median(early_tail_values)) / abs(
        final_reference_value
    )

    tail_percentile_95_absolute_relative_error = sorted_tail_errors[percentile_95_index]
    is_stable = (
        tail_percentile_95_absolute_relative_error
        <= criteria.relative_oscillation_limit
        and tail_relative_drift <= criteria.relative_drift_limit
    )

    return StabilityAssessment(
        final_reference_value=final_reference_value,
        post_clear_peak_absolute_relative_error=post_clear_peak_absolute_relative_error,
        tail_percentile_95_absolute_relative_error=(
            tail_percentile_95_absolute_relative_error
        ),
        tail_relative_drift=tail_relative_drift,
        is_stable=is_stable,
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
