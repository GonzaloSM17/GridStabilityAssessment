"""Reusable damping assessment for published active-power transfers."""

from dataclasses import dataclass
from math import log, pi, sqrt
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class DampingCriteria:
    """Time windows and limits used to assess one active-power transfer."""

    pre_fault_start_time_s: float = 0.9
    fault_time_s: float = 1.0
    clearance_time_s: float = 1.12
    final_reference_duration_s: float = 1.0
    minimum_simulation_end_time_s: float = 5.0
    minimum_pre_fault_power: float = 1.0
    opened_final_power_ratio: float = 0.05
    minimum_peak_relative_amplitude: float = 0.02
    minimum_damping_ratio: float = 0.05


@dataclass(frozen=True)
class DampingPair:
    """Two consecutive extrema of the same sign and their damping estimate."""

    polarity: str
    first_time_s: float
    second_time_s: float
    first_amplitude: float
    second_amplitude: float
    period_s: float
    damping_ratio: float


@dataclass(frozen=True)
class TransferRunAssessment:
    """The damping result for one transfer monitored in one simulation run."""

    run_id: int
    transfer_name: str
    pre_fault_power: float
    final_power: float
    status: str
    damping_pairs: tuple[DampingPair, ...]
    worst_damping_pair: DampingPair | None
    criteria: DampingCriteria

    @property
    def damping_ratio(self) -> float | None:
        """Return the lowest reliable damping ratio, when available."""
        if self.worst_damping_pair is None:
            return None
        return self.worst_damping_pair.damping_ratio

    @property
    def is_compliant(self) -> bool | None:
        """Return compliance only when damping can be evaluated."""
        if self.damping_ratio is None:
            return None
        return self.damping_ratio >= self.criteria.minimum_damping_ratio


@dataclass(frozen=True)
class TransferStudyAnalysis:
    """All local transfer damping assessments in a Monte Carlo study."""

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
    """Assess all reliable same-sign peak pairs after contingency clearance."""
    criteria = criteria or DampingCriteria()
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
        return TransferRunAssessment(status="inactive_pre_fault", damping_pairs=(), worst_damping_pair=None, **common)
    if abs(final_power) <= abs(pre_fault_power) * criteria.opened_final_power_ratio:
        return TransferRunAssessment(status="opened", damping_pairs=(), worst_damping_pair=None, **common)

    analysis_end_time_s = final_start_time_s
    response = [
        (time, value - final_power)
        for time, value in samples
        if criteria.clearance_time_s <= time <= analysis_end_time_s
    ]
    amplitude_floor = abs(pre_fault_power) * criteria.minimum_peak_relative_amplitude
    positive_peaks, negative_peaks = _same_sign_extrema(response, amplitude_floor)
    damping_pairs = tuple(
        [*_damping_pairs(positive_peaks, "positive"), *_damping_pairs(negative_peaks, "negative")]
    )
    if not damping_pairs:
        return TransferRunAssessment(status="no_identifiable_oscillation", damping_pairs=(), worst_damping_pair=None, **common)

    worst_damping_pair = min(damping_pairs, key=lambda pair: pair.damping_ratio)
    status = (
        "pass"
        if worst_damping_pair.damping_ratio >= criteria.minimum_damping_ratio
        else "below_damping_limit"
    )
    return TransferRunAssessment(
        status=status,
        damping_pairs=damping_pairs,
        worst_damping_pair=worst_damping_pair,
        **common,
    )


def _same_sign_extrema(
    response: Sequence[tuple[float, float]], amplitude_floor: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return usable positive maxima and negative minima around final power."""
    positive_peaks = []
    negative_peaks = []
    for left, middle, right in zip(response, response[1:], response[2:]):
        if middle[1] >= left[1] and middle[1] > right[1] and middle[1] >= amplitude_floor:
            positive_peaks.append(middle)
        if middle[1] <= left[1] and middle[1] < right[1] and middle[1] <= -amplitude_floor:
            negative_peaks.append((middle[0], abs(middle[1])))
    return positive_peaks, negative_peaks


def _damping_pairs(
    peaks: Sequence[tuple[float, float]], polarity: str
) -> list[DampingPair]:
    """Compute logarithmic-decrement damping for consecutive same-sign peaks."""
    pairs = []
    for (first_time_s, first_amplitude), (second_time_s, second_amplitude) in zip(peaks, peaks[1:]):
        if second_amplitude <= 0 or first_amplitude <= 0:
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
                period_s=second_time_s - first_time_s,
                damping_ratio=damping_ratio,
            )
        )
    return pairs
