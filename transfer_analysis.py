"""Electromechanical-mode damping assessment for active-power transfers."""

from dataclasses import dataclass
from math import log, pi, sqrt
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class DampingCriteria:
    """Configuration for identifying one dominant electromechanical power mode."""

    pre_fault_start_time_s: float = 0.9
    fault_time_s: float = 1.0
    clearance_time_s: float = 1.12
    post_clearance_guard_time_s: float = 0.05
    final_reference_duration_s: float = 1.0
    minimum_simulation_end_time_s: float = 5.0
    minimum_pre_fault_power: float = 1.0
    opened_final_power_ratio: float = 0.05
    mode_minimum_frequency_hz: float = 0.2
    mode_maximum_frequency_hz: float = 2.0
    mode_filter_half_width_hz: float = 0.25
    minimum_peak_relative_amplitude: float = 0.02
    dominant_pair_amplitude_fraction: float = 0.50
    late_mode_start_time_s: float = 2.0
    minimum_damping_ratio: float = 0.05


@dataclass(frozen=True)
class DampingPair:
    """Two consecutive same-sign extrema of the isolated power mode."""

    polarity: str
    first_time_s: float
    second_time_s: float
    first_amplitude: float
    second_amplitude: float
    period_s: float
    damping_ratio: float


@dataclass(frozen=True)
class TransferRunAssessment:
    """One transfer's dominant electromechanical-mode result in one run."""

    run_id: int
    transfer_name: str
    pre_fault_power: float
    final_power: float
    status: str
    dominant_frequency_hz: float | None
    warning: str | None
    damping_pairs: tuple[DampingPair, ...]
    selected_damping_pair: DampingPair | None
    criteria: DampingCriteria
    comment: str | None = None

    @property
    def damping_ratio(self) -> float | None:
        """Return the damping of the selected dominant-mode pair."""
        if self.selected_damping_pair is None:
            return None
        return self.selected_damping_pair.damping_ratio

    @property
    def is_compliant(self) -> bool | None:
        """Return the 5-percent compliance result when the mode is evaluable."""
        if self.damping_ratio is None:
            return None
        return self.damping_ratio >= self.criteria.minimum_damping_ratio


@dataclass(frozen=True)
class TransferStudyAnalysis:
    """All transfer-mode assessments in one Monte Carlo study."""

    assessments: tuple[TransferRunAssessment, ...]

    @property
    def transfer_names(self) -> tuple[str, ...]:
        return tuple(sorted({assessment.transfer_name for assessment in self.assessments}))

    def assessments_for_transfer(self, transfer_name: str) -> tuple[TransferRunAssessment, ...]:
        return tuple(
            assessment
            for assessment in self.assessments
            if assessment.transfer_name == transfer_name
        )

    def assessment_for(self, run_id: int, transfer_name: str) -> TransferRunAssessment | None:
        return next(
            (
                assessment
                for assessment in self.assessments
                if assessment.run_id == run_id and assessment.transfer_name == transfer_name
            ),
            None,
        )


def assess_transfer_damping(
    time_values: Sequence[float],
    power_values: Sequence[float],
    *,
    run_id: int,
    transfer_name: str,
    criteria: DampingCriteria | None = None,
) -> TransferRunAssessment:
    """Identify and assess the dominant 0.2-2.0 Hz power oscillation mode.

    The global study stability result is deliberately not affected by this
    assessment.  It is an independent transfer-level normative diagnostic.
    """
    criteria = criteria or DampingCriteria()
    _validate_criteria(criteria)
    if len(time_values) != len(power_values) or len(time_values) < 3:
        raise ValueError("Transfer damping requires matching time and power samples.")

    samples = [(float(time), float(power)) for time, power in zip(time_values, power_values)]
    if any(right[0] <= left[0] for left, right in zip(samples, samples[1:])):
        raise ValueError("Transfer time samples must be strictly increasing.")
    if samples[-1][0] < criteria.minimum_simulation_end_time_s:
        raise ValueError("Transfer signal ends before final-reference evaluation time.")

    pre_fault_values = [
        value
        for time, value in samples
        if criteria.pre_fault_start_time_s <= time < criteria.fault_time_s
    ]
    if not pre_fault_values:
        raise ValueError("Transfer signal does not cover the pre-fault window.")
    pre_fault_power = median(pre_fault_values)

    final_start_time_s = samples[-1][0] - criteria.final_reference_duration_s
    final_values = [value for time, value in samples if time >= final_start_time_s]
    if not final_values:
        raise ValueError("Transfer signal does not cover the final-reference window.")
    final_power = median(final_values)

    common = dict(
        run_id=run_id,
        transfer_name=transfer_name,
        pre_fault_power=pre_fault_power,
        final_power=final_power,
        criteria=criteria,
    )
    if abs(pre_fault_power) < criteria.minimum_pre_fault_power:
        return TransferRunAssessment(
            status="inactive_pre_fault",
            dominant_frequency_hz=None,
            warning=None,
            damping_pairs=(),
            selected_damping_pair=None,
            comment="Transfer is inactive before the contingency.",
            **common,
        )
    if abs(final_power) <= abs(pre_fault_power) * criteria.opened_final_power_ratio:
        return TransferRunAssessment(
            status="opened",
            dominant_frequency_hz=None,
            warning=None,
            damping_pairs=(),
            selected_damping_pair=None,
            comment="Transfer is open in the final operating condition.",
            **common,
        )

    mode_start_time_s = criteria.clearance_time_s + criteria.post_clearance_guard_time_s
    mode_samples = [
        (time, value - final_power)
        for time, value in samples
        if time >= mode_start_time_s
    ]
    frequency_hz, filtered_response = _isolate_dominant_mode(mode_samples, criteria)
    if frequency_hz is None or filtered_response is None:
        return TransferRunAssessment(
            status="no_identifiable_electromechanical_mode",
            dominant_frequency_hz=None,
            warning=(
                "No dominant electromechanical mode was identified within "
                f"{criteria.mode_minimum_frequency_hz:.1f}-{criteria.mode_maximum_frequency_hz:.1f} Hz."
            ),
            damping_pairs=(),
            selected_damping_pair=None,
            **common,
        )

    amplitude_floor = abs(pre_fault_power) * criteria.minimum_peak_relative_amplitude
    positive_peaks, negative_peaks = _same_sign_extrema(
        filtered_response,
        amplitude_floor,
        minimum_peak_distance_s=0.5 / criteria.mode_maximum_frequency_hz,
    )
    minimum_period_s = 1.0 / criteria.mode_maximum_frequency_hz
    maximum_period_s = 1.0 / criteria.mode_minimum_frequency_hz
    damping_pairs = tuple(
        [
            *_damping_pairs(
                positive_peaks,
                "positive",
                minimum_period_s,
                maximum_period_s,
            ),
            *_damping_pairs(
                negative_peaks,
                "negative",
                minimum_period_s,
                maximum_period_s,
            ),
        ]
    )
    if not damping_pairs:
        available_duration_s = mode_samples[-1][0] - mode_samples[0][0]
        required_duration_s = 1.0 / frequency_hz
        if available_duration_s >= required_duration_s:
            return TransferRunAssessment(
                status="no_identifiable_electromechanical_mode",
                dominant_frequency_hz=frequency_hz,
                warning=(
                    f"A {frequency_hz:.2f} Hz candidate was found, but no two "
                    "significant same-sign extrema were retained for damping evaluation."
                ),
                damping_pairs=(),
                selected_damping_pair=None,
                **common,
            )
        return TransferRunAssessment(
            status="insufficient_mode_cycles",
            dominant_frequency_hz=frequency_hz,
            warning=(
                f"Dominant mode {frequency_hz:.2f} Hz needs at least "
                f"{required_duration_s:.2f} s for one complete cycle; "
                f"only {available_duration_s:.2f} s are available."
            ),
            damping_pairs=(),
            selected_damping_pair=None,
            **common,
        )

    selected_pair = _select_dominant_pair(damping_pairs, criteria)
    warning = None
    if selected_pair.first_time_s > criteria.late_mode_start_time_s:
        warning = (
            f"The selected {frequency_hz:.2f} Hz mode begins late "
            f"at {selected_pair.first_time_s:.2f} s."
        )
    if selected_pair.damping_ratio < 0.0:
        status = "warning_growing_mode"
        warning = _append_warning(
            warning,
            "The selected dominant-mode envelope grows between the measured extrema.",
        )
    elif selected_pair.damping_ratio < criteria.minimum_damping_ratio:
        status = "warning_below_damping_limit"
        warning = _append_warning(
            warning,
            f"Damping is below the {criteria.minimum_damping_ratio * 100:.0f}% limit.",
        )
    else:
        status = "pass"

    return TransferRunAssessment(
        status=status,
        dominant_frequency_hz=frequency_hz,
        warning=warning,
        damping_pairs=damping_pairs,
        selected_damping_pair=selected_pair,
        **common,
    )


def _validate_criteria(criteria: DampingCriteria) -> None:
    if not 0.0 < criteria.mode_minimum_frequency_hz < criteria.mode_maximum_frequency_hz:
        raise ValueError("The electromechanical mode frequency limits are invalid.")
    if criteria.post_clearance_guard_time_s < 0.0:
        raise ValueError("The post-clearance guard time cannot be negative.")
    if not 0.0 < criteria.dominant_pair_amplitude_fraction <= 1.0:
        raise ValueError("The dominant-pair amplitude fraction must be in (0, 1].")


def _isolate_dominant_mode(
    samples: Sequence[tuple[float, float]],
    criteria: DampingCriteria,
) -> tuple[float | None, list[tuple[float, float]] | None]:
    """Find the highest-energy target-band mode and isolate it around its frequency."""
    import numpy as np
    from scipy import signal

    if len(samples) < 16:
        return None, None
    time = np.asarray([sample[0] for sample in samples], dtype=float)
    response = np.asarray([sample[1] for sample in samples], dtype=float)
    time_step_s = float(np.median(np.diff(time)))
    if time_step_s <= 0.0:
        return None, None
    sample_frequency_hz = 1.0 / time_step_s
    if sample_frequency_hz <= 2.5 * criteria.mode_maximum_frequency_hz:
        return None, None

    uniform_time = np.arange(time[0], time[-1] + time_step_s / 2.0, time_step_s)
    uniform_response = np.interp(uniform_time, time, response)
    detrended_response = signal.detrend(uniform_response, type="linear")
    frequencies, powers = signal.periodogram(
        detrended_response,
        fs=sample_frequency_hz,
        window="hann",
        scaling="spectrum",
    )
    candidates = (
        (frequencies >= criteria.mode_minimum_frequency_hz)
        & (frequencies <= criteria.mode_maximum_frequency_hz)
    )
    if not np.any(candidates):
        return None, None
    candidate_frequencies = frequencies[candidates]
    candidate_powers = powers[candidates]
    if not np.any(candidate_powers > 0.0):
        return None, None
    dominant_frequency_hz = float(candidate_frequencies[np.argmax(candidate_powers)])

    half_width_hz = max(
        criteria.mode_filter_half_width_hz,
        0.5 / (uniform_time[-1] - uniform_time[0]),
    )
    lower_frequency_hz = max(0.05, dominant_frequency_hz - half_width_hz)
    upper_frequency_hz = min(
        sample_frequency_hz * 0.45,
        dominant_frequency_hz + half_width_hz,
    )
    if lower_frequency_hz >= upper_frequency_hz:
        return None, None
    filter_sections = signal.butter(
        3,
        (lower_frequency_hz, upper_frequency_hz),
        btype="bandpass",
        fs=sample_frequency_hz,
        output="sos",
    )
    try:
        filtered_response = signal.sosfiltfilt(filter_sections, uniform_response)
    except ValueError:
        return None, None
    return dominant_frequency_hz, list(zip(uniform_time.tolist(), filtered_response.tolist()))


def _same_sign_extrema(
    response: Sequence[tuple[float, float]],
    amplitude_floor: float,
    *,
    minimum_peak_distance_s: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return distinct positive peaks and negative valleys of one filtered mode."""
    from scipy.signal import find_peaks
    import numpy as np

    time = np.asarray([sample[0] for sample in response], dtype=float)
    values = np.asarray([sample[1] for sample in response], dtype=float)
    time_step_s = float(np.median(np.diff(time)))
    distance = max(1, round(minimum_peak_distance_s / time_step_s))
    positive_indices, _ = find_peaks(values, height=amplitude_floor, distance=distance)
    negative_indices, _ = find_peaks(-values, height=amplitude_floor, distance=distance)
    positive_peaks = [(float(time[index]), float(values[index])) for index in positive_indices]
    negative_peaks = [(float(time[index]), float(-values[index])) for index in negative_indices]
    return positive_peaks, negative_peaks


def _damping_pairs(
    peaks: Sequence[tuple[float, float]],
    polarity: str,
    minimum_period_s: float,
    maximum_period_s: float,
) -> list[DampingPair]:
    """Calculate logarithmic decrement only for target-band same-sign pairs."""
    pairs = []
    for (first_time_s, first_amplitude), (second_time_s, second_amplitude) in zip(peaks, peaks[1:]):
        period_s = second_time_s - first_time_s
        if not minimum_period_s <= period_s <= maximum_period_s:
            continue
        decrement = log(first_amplitude / second_amplitude)
        damping_ratio = decrement / sqrt((2.0 * pi) ** 2 + decrement**2)
        pairs.append(
            DampingPair(
                polarity=polarity,
                first_time_s=first_time_s,
                second_time_s=second_time_s,
                first_amplitude=first_amplitude,
                second_amplitude=second_amplitude,
                period_s=period_s,
                damping_ratio=damping_ratio,
            )
        )
    return pairs


def _select_dominant_pair(
    pairs: Sequence[DampingPair],
    criteria: DampingCriteria,
) -> DampingPair:
    """Choose the largest first excursion representative of the dominant mode."""
    maximum_amplitude = max(
        max(pair.first_amplitude, pair.second_amplitude) for pair in pairs
    )
    representative_pairs = [
        pair
        for pair in pairs
        if max(pair.first_amplitude, pair.second_amplitude)
        >= criteria.dominant_pair_amplitude_fraction * maximum_amplitude
    ]
    return max(
        representative_pairs,
        key=lambda pair: (pair.first_amplitude, -pair.first_time_s),
    )


def _append_warning(existing_warning: str | None, addition: str) -> str:
    return f"{existing_warning} {addition}" if existing_warning else addition
