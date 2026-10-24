"""Electromechanical-mode damping assessment for active-power transfers."""

from dataclasses import dataclass
from math import log, pi, sqrt
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class DampingCriteria:
    """Configuration for post-contingency electromechanical mode assessment."""

    pre_fault_start_time_s: float = 0.9
    fault_time_s: float = 1.0
    clearance_time_s: float = 1.12
    post_clearance_guard_time_s: float = 0.05
    analysis_start_time_s: float = 2.0
    final_reference_duration_s: float = 1.0
    minimum_simulation_end_time_s: float = 5.0
    minimum_pre_fault_power: float = 1.0
    opened_final_power_ratio: float = 0.05
    mode_minimum_frequency_hz: float = 0.2
    mode_maximum_frequency_hz: float = 2.5
    minimum_damping_ratio: float = 0.05


@dataclass(frozen=True)
class DampingPair:
    """Two consecutive local extrema of the same kind in an isolated mode."""

    extremum: str
    first_time_s: float
    second_time_s: float
    first_amplitude: float
    second_amplitude: float
    period_s: float
    damping_ratio: float
    first_value: float
    second_value: float

    @property
    def amplitude_ratio(self) -> float:
        return self.second_amplitude / self.first_amplitude


@dataclass(frozen=True)
class ElectromechanicalMode:
    """One spectral candidate with its band and consecutive damping pairs."""

    frequency_hz: float
    lower_band_hz: float
    upper_band_hz: float
    status: str
    damping_pairs: tuple[DampingPair, ...]
    selected_pair: DampingPair | None
    envelope_peak_time_s: float | None
    envelope_peak_amplitude: float | None
    envelope_peak_extremum: str | None
    tail_pair_count: int
    terminal_sequence_pair_count: int
    message: str | None = None

    @property
    def damping_ratio(self) -> float | None:
        return self.selected_pair.damping_ratio if self.selected_pair else None

    @property
    def selected_extremum(self) -> str | None:
        return self.selected_pair.extremum if self.selected_pair else None

    @property
    def classification(self) -> str:
        return {
            "damped": "Damped",
            "damped_under": "Damped under",
            "growing": "Growing",
            "not_measurable": "Not measurable",
        }[self.status]


@dataclass(frozen=True)
class TransferRunAssessment:
    """All observable electromechanical modes of one transfer in one run."""

    run_id: int
    transfer_name: str
    pre_fault_power: float
    final_power: float
    status: str
    modes: tuple[ElectromechanicalMode, ...]
    criteria: DampingCriteria
    warning: str | None = None
    comment: str | None = None

    @property
    def critical_mode(self) -> ElectromechanicalMode | None:
        """Return the observed mode with the smallest damping ratio."""
        measurable = [
            mode
            for mode in self.modes
            if mode.status != "not_measurable" and mode.damping_ratio is not None
        ]
        return min(measurable, key=lambda mode: mode.damping_ratio) if measurable else None

    @property
    def damping_ratio(self) -> float | None:
        mode = self.critical_mode
        return mode.damping_ratio if mode else None

    @property
    def is_compliant(self) -> bool | None:
        if self.status in {"not_measurable", "inactive_pre_fault", "opened"}:
            return None
        return self.status == "damped"

    @property
    def classification(self) -> str:
        return {
            "damped": "Damped",
            "damped_under": "Damped under",
            "growing": "Growing",
            "not_measurable": "Not measurable",
            "inactive_pre_fault": "Not applicable",
            "opened": "Not applicable",
        }[self.status]


@dataclass(frozen=True)
class TransferStudyAnalysis:
    """All transfer-mode assessments in one Monte Carlo study."""

    assessments: tuple[TransferRunAssessment, ...]

    @property
    def transfer_names(self) -> tuple[str, ...]:
        return tuple(sorted({assessment.transfer_name for assessment in self.assessments}))

    def assessments_for_transfer(self, transfer_name: str) -> tuple[TransferRunAssessment, ...]:
        return tuple(item for item in self.assessments if item.transfer_name == transfer_name)

    def assessment_for(self, run_id: int, transfer_name: str) -> TransferRunAssessment | None:
        return next((item for item in self.assessments if item.run_id == run_id and item.transfer_name == transfer_name), None)


def assess_transfer_damping(
    time_values: Sequence[float],
    power_values: Sequence[float],
    *,
    run_id: int,
    transfer_name: str,
    criteria: DampingCriteria | None = None,
) -> TransferRunAssessment:
    """Assess every identifiable 0.2-2.5 Hz mode of one power transfer.

    ``P_final`` is retained as the physical post-contingency reference. The
    assessed signal is ``P(t) - P_final``. Modes are discovered in the whole
    target band and measured independently; mixed responses get no single xi.
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

    pre_fault_values = [value for time, value in samples if criteria.pre_fault_start_time_s <= time < criteria.fault_time_s]
    if not pre_fault_values:
        raise ValueError("Transfer signal does not cover the pre-fault window.")
    final_start = samples[-1][0] - criteria.final_reference_duration_s
    final_values = [value for time, value in samples if time >= final_start]
    if not final_values:
        raise ValueError("Transfer signal does not cover the final-reference window.")
    pre_fault_power = median(pre_fault_values)
    final_power = median(final_values)
    common = dict(run_id=run_id, transfer_name=transfer_name, pre_fault_power=pre_fault_power, final_power=final_power, criteria=criteria)
    if abs(pre_fault_power) < criteria.minimum_pre_fault_power:
        return TransferRunAssessment(status="inactive_pre_fault", modes=(), comment="Transfer is inactive before the contingency.", **common)
    if abs(final_power) <= abs(pre_fault_power) * criteria.opened_final_power_ratio:
        return TransferRunAssessment(status="opened", modes=(), comment="Transfer is open in the final operating condition.", **common)

    mode_start = max(criteria.analysis_start_time_s, criteria.clearance_time_s + criteria.post_clearance_guard_time_s)
    deviation = [(time, value - final_power) for time, value in samples if time >= mode_start]
    mode_bands = _detect_mode_bands(deviation, criteria)
    if not mode_bands:
        return TransferRunAssessment(
            status="not_measurable", modes=(),
            warning=("No identifiable electromechanical mode was found within "
                     f"{criteria.mode_minimum_frequency_hz:.1f}-{criteria.mode_maximum_frequency_hz:.1f} Hz."),
            **common,
        )
    modes = tuple(
        _measure_mode(deviation, frequency, lower, upper, criteria)
        for frequency, lower, upper in mode_bands
    )
    measurable = [
        mode
        for mode in modes
        if mode.status != "not_measurable" and mode.damping_ratio is not None
    ]
    if not measurable:
        return TransferRunAssessment(
            status="not_measurable", modes=modes,
            warning="Detected electromechanical content could not be measured over two local extrema of the same kind.",
            **common,
        )
    if any(mode.status == "growing" for mode in measurable):
        status, warning = "growing", "At least one observed electromechanical mode has non-positive damping."
    elif any(mode.status == "damped_under" for mode in measurable):
        status, warning = "damped_under", f"At least one observed mode is below the {criteria.minimum_damping_ratio * 100:.0f}% limit."
    else:
        status, warning = "damped", None
    return TransferRunAssessment(status=status, modes=modes, warning=warning, **common)


def _validate_criteria(criteria: DampingCriteria) -> None:
    if not 0.0 < criteria.mode_minimum_frequency_hz < criteria.mode_maximum_frequency_hz:
        raise ValueError("The electromechanical mode frequency limits are invalid.")
    if criteria.post_clearance_guard_time_s < 0.0:
        raise ValueError("The post-clearance guard duration cannot be negative.")
    if criteria.analysis_start_time_s < criteria.clearance_time_s:
        raise ValueError("Damping analysis must not start before contingency clearance.")


def _detect_mode_bands(
    samples: Sequence[tuple[float, float]],
    criteria: DampingCriteria,
) -> tuple[tuple[float, float, float], ...]:
    """Return every local spectral maximum with its spectral-basin band.

    The outer band limits come from the accepted electromechanical range. The
    nearest spectral valleys delimit each local candidate, so no prominence
    threshold, maximum count, or fixed filter width is introduced.
    """
    import numpy as np
    from scipy import signal

    if len(samples) < 16:
        return ()
    time = np.asarray([sample[0] for sample in samples], dtype=float)
    response = np.asarray([sample[1] for sample in samples], dtype=float)
    time_step = float(np.median(np.diff(time)))
    if time_step <= 0.0:
        return ()
    sample_frequency = 1.0 / time_step
    if sample_frequency <= 2.5 * criteria.mode_maximum_frequency_hz:
        return ()
    uniform_time = np.arange(time[0], time[-1] + time_step / 2.0, time_step)
    uniform_response = np.interp(uniform_time, time, response)
    frequencies, powers = signal.periodogram(signal.detrend(uniform_response, type="linear"), fs=sample_frequency, window="hann", scaling="spectrum")
    in_band = (frequencies >= criteria.mode_minimum_frequency_hz) & (frequencies <= criteria.mode_maximum_frequency_hz)
    band_frequencies, band_powers = frequencies[in_band], powers[in_band]
    if len(band_frequencies) == 0 or not np.any(band_powers > 0.0):
        return ()
    peak_indices, _ = signal.find_peaks(band_powers)
    candidate_indices = set(peak_indices.tolist())
    # The global maximum may be at a band edge, where find_peaks has no local
    # neighbour on both sides. It remains a candidate by definition.
    candidate_indices.add(int(np.argmax(band_powers)))
    return _spectral_mode_bands(
        band_frequencies,
        band_powers,
        tuple(sorted(candidate_indices)),
        criteria,
    )


def _spectral_mode_bands(
    frequencies,
    powers,
    candidate_indices: Sequence[int],
    criteria: DampingCriteria,
) -> tuple[tuple[float, float, float], ...]:
    """Bound each candidate by the nearest spectral valleys around it.

    The valleys define the candidate's own spectral basin. Only when no valley
    exists on one side is the accepted electromechanical band boundary used.
    """
    from scipy import signal

    valley_indices, _ = signal.find_peaks(-powers)
    bands = []
    for index in candidate_indices:
        left_valleys = valley_indices[valley_indices < index]
        right_valleys = valley_indices[valley_indices > index]
        lower = (
            float(frequencies[left_valleys[-1]])
            if len(left_valleys)
            else criteria.mode_minimum_frequency_hz
        )
        upper = (
            float(frequencies[right_valleys[0]])
            if len(right_valleys)
            else criteria.mode_maximum_frequency_hz
        )
        frequency = float(frequencies[index])
        if lower < frequency < upper:
            bands.append((frequency, lower, upper))
    return tuple(bands)


def _measure_mode(
    samples: Sequence[tuple[float, float]],
    frequency_hz: float,
    lower: float,
    upper: float,
    criteria: DampingCriteria,
) -> ElectromechanicalMode:
    """Measure all consecutive local-extrema pairs of one detected mode."""
    filtered = isolate_mode(samples, lower, upper)
    if filtered is None:
        return ElectromechanicalMode(
            frequency_hz, lower, upper, "not_measurable", (), None,
            None, None, None, 0, 0,
            "The mode could not be isolated from the sampled response.",
        )
    maxima, minima = _local_extrema(filtered, frequency_hz)
    usable = [("maximum", maxima), ("minimum", minima)]
    usable = [(extremum, points) for extremum, points in usable if len(points) >= 2]
    if not usable:
        return ElectromechanicalMode(
            frequency_hz, lower, upper, "not_measurable", (), None,
            None, None, None, 0, 0,
            "Fewer than two local extrema of the same kind are available.",
        )
    pairs = tuple(
        pair
        for extremum, points in usable
        for pair in _consecutive_damping_pairs(points, extremum)
    )
    if not pairs:
        return ElectromechanicalMode(
            frequency_hz, lower, upper, "not_measurable", (), None,
            None, None, None, 0, 0,
            "No consecutive local-extrema pair is available.",
        )

    peak_extremum, peak_time, peak_amplitude = max(
        (
            (extremum, time, amplitude)
            for extremum, points in usable
            for time, amplitude, _ in points
        ),
        key=lambda item: item[2],
    )
    peak_tail_pairs = tuple(
        pair
        for pair in pairs
        if pair.extremum == peak_extremum and pair.first_time_s >= peak_time
    )
    if not peak_tail_pairs:
        return ElectromechanicalMode(
            frequency_hz, lower, upper, "not_measurable", pairs, None,
            peak_time, peak_amplitude, peak_extremum, 0, 0,
            "The envelope maximum has no following consecutive pair of the same kind.",
        )

    terminal_pairs = _terminal_direction_sequence(peak_tail_pairs)
    terminal_is_growing = terminal_pairs[-1].damping_ratio <= 0.0
    if terminal_is_growing:
        selected_pair = min(terminal_pairs, key=lambda pair: pair.damping_ratio)
        status = "growing"
    else:
        selected_pair = max(terminal_pairs, key=lambda pair: pair.damping_ratio)
        status = (
            "damped"
            if selected_pair.damping_ratio >= criteria.minimum_damping_ratio
            else "damped_under"
        )
    return ElectromechanicalMode(
        frequency_hz, lower, upper, status, pairs, selected_pair,
        peak_time, peak_amplitude, peak_extremum, len(peak_tail_pairs),
        len(terminal_pairs),
    )


def _terminal_direction_sequence(
    pairs: Sequence[DampingPair],
) -> tuple[DampingPair, ...]:
    """Return the uninterrupted terminal chain with the final pair's direction.

    A positive damping ratio denotes a decreasing envelope. A zero or negative
    ratio denotes a non-decreasing envelope. The search stops at the first pair
    with the opposite direction, so a better earlier pair cannot bypass a local
    recovery or growth interval.
    """
    if not pairs:
        return ()
    terminal_is_decreasing = pairs[-1].damping_ratio > 0.0
    sequence = [pairs[-1]]
    for pair in reversed(pairs[:-1]):
        pair_is_decreasing = pair.damping_ratio > 0.0
        if pair_is_decreasing != terminal_is_decreasing:
            break
        sequence.append(pair)
    return tuple(reversed(sequence))


def _consecutive_damping_pairs(
    extrema: Sequence[tuple[float, float, float]],
    extremum: str,
) -> tuple[DampingPair, ...]:
    """Calculate logarithmic decrement from consecutive raw local extrema."""
    pairs = []
    for (
        (first_time, first_amplitude, first_value),
        (second_time, second_amplitude, second_value),
    ) in zip(extrema, extrema[1:]):
        period_s = second_time - first_time
        if period_s <= 0.0 or first_amplitude <= 0.0 or second_amplitude <= 0.0:
            continue
        decrement = log(first_amplitude / second_amplitude)
        damping_ratio = decrement / sqrt((2.0 * pi) ** 2 + decrement**2)
        pairs.append(DampingPair(
            extremum=extremum,
            first_time_s=first_time,
            second_time_s=second_time,
            first_amplitude=first_amplitude,
            second_amplitude=second_amplitude,
            period_s=period_s,
            damping_ratio=damping_ratio,
            first_value=first_value,
            second_value=second_value,
        ))
    return tuple(pairs)


def isolate_mode(samples: Sequence[tuple[float, float]], lower_frequency_hz: float, upper_frequency_hz: float) -> list[tuple[float, float]] | None:
    """Band-pass a single spectrum-detected mode of the deviation signal."""
    import numpy as np
    from scipy import signal

    if lower_frequency_hz >= upper_frequency_hz:
        return None
    time = np.asarray([sample[0] for sample in samples], dtype=float)
    response = np.asarray([sample[1] for sample in samples], dtype=float)
    time_step = float(np.median(np.diff(time)))
    if time_step <= 0.0:
        return None
    sample_frequency = 1.0 / time_step
    uniform_time = np.arange(time[0], time[-1] + time_step / 2.0, time_step)
    uniform_response = np.interp(uniform_time, time, response)
    sections = signal.butter(3, (lower_frequency_hz, upper_frequency_hz), btype="bandpass", fs=sample_frequency, output="sos")
    try:
        response_filtered = signal.sosfiltfilt(sections, uniform_response)
    except ValueError:
        return None
    return list(zip(uniform_time.tolist(), response_filtered.tolist()))


def _local_extrema(
    response: Sequence[tuple[float, float]],
    frequency_hz: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Return all local maxima and minima of an isolated EM response.

    Each point retains its raw signed value for plotting and its absolute
    amplitude relative to ``P_final`` for the logarithmic decrement.  A zero
    crossing is therefore part of the measured response; it never excludes or
    splits an extrema sequence.
    """
    import numpy as np
    from scipy.signal import find_peaks

    time = np.asarray([sample[0] for sample in response], dtype=float)
    values = np.asarray([sample[1] for sample in response], dtype=float)
    time_step = float(np.median(np.diff(time)))
    distance = max(1, round((0.5 / frequency_hz) / time_step))
    maximum_indices, _ = find_peaks(values, distance=distance)
    minimum_indices, _ = find_peaks(-values, distance=distance)
    maxima = [
        (float(time[index]), float(abs(values[index])), float(values[index]))
        for index in maximum_indices
    ]
    minima = [
        (float(time[index]), float(abs(values[index])), float(values[index]))
        for index in minimum_indices
    ]
    return maxima, minima
