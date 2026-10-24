"""Shared pre-contingency initialization assessment for RMS-voltage signals."""

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class InitializationCriteria:
    """RMS-voltage initialization limits based on fast residual oscillations."""

    evaluation_start_time_s: float = 0.5
    evaluation_end_time_s: float = 1.0
    smoothing_duration_s: float = 0.05
    maximum_fast_relative_error_percentile_99: float = 0.001


@dataclass(frozen=True)
class InitializationAssessment:
    """Pre-contingency settling metrics for one RMS-voltage signal."""

    reference_value: float
    fast_relative_error_percentile_99: float
    fast_rms_relative_error: float
    is_well_initialized: bool
    rejection_reasons: tuple[str, ...]
    criteria: InitializationCriteria


def assess_initialization(
    time_values: Sequence[float],
    signal_values: Sequence[float],
    criteria: InitializationCriteria | None = None,
) -> InitializationAssessment:
    """Assess whether an RMS-voltage signal settles before the contingency.

    A centred moving average estimates the slow voltage trend. The residual is
    measured over the whole evaluation window, allowing slow variation while
    rejecting the fast saw-tooth behaviour characteristic of bad initialization.
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
    trend = np.convolve(values, np.full(smoothing_samples, 1.0 / smoothing_samples), mode="valid")
    edge_samples = smoothing_samples // 2
    fast_relative_errors = np.abs(values[edge_samples:-edge_samples] - trend) / abs(reference_value)
    percentile_99 = float(np.quantile(fast_relative_errors, 0.99))
    rms_error = float(np.sqrt(np.mean(fast_relative_errors**2)))
    rejection_reasons = (
        ("fast_oscillation",)
        if percentile_99 > criteria.maximum_fast_relative_error_percentile_99
        else ()
    )
    return InitializationAssessment(
        reference_value=reference_value,
        fast_relative_error_percentile_99=percentile_99,
        fast_rms_relative_error=rms_error,
        is_well_initialized=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        criteria=criteria,
    )
