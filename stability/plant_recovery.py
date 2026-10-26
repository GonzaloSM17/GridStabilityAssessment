"""Shared operating-point recovery assessment for converter-based plants."""

from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PlantRecoveryCriteria:
    """Define the observed recovery comparison for one contingency.

    The pre-fault operating point is sampled at a defined instant.  Recovery is
    observed from one second after fault clearance onward, while the reported
    decision is deliberately the terminal sample of the simulation.
    """

    pre_fault_reference_time_s: float = 0.5
    recovery_evaluation_delay_s: float = 1.0
    active_power_tolerance_mw: float = 5.0
    reactive_power_tolerance_mvar: float = 5.0

    def __post_init__(self) -> None:
        if self.pre_fault_reference_time_s < 0.0:
            raise ValueError("The pre-fault reference time cannot be negative.")
        if self.recovery_evaluation_delay_s < 0.0:
            raise ValueError("The recovery evaluation delay cannot be negative.")
        if self.active_power_tolerance_mw < 0.0:
            raise ValueError("The active-power tolerance cannot be negative.")
        if self.reactive_power_tolerance_mvar < 0.0:
            raise ValueError("The reactive-power tolerance cannot be negative.")


@dataclass(frozen=True)
class PlantRecoveryAssessment:
    """Terminal P/Q recovery of one plant against its pre-fault operation."""

    plant_name: str
    technology: str
    reference_time_s: float
    evaluation_start_time_s: float
    final_time_s: float
    pre_fault_active_power_mw: float
    final_active_power_mw: float
    active_power_error_mw: float
    maximum_active_power_absolute_error_mw: float
    maximum_active_power_error_mw: float
    maximum_active_power_error_time_s: float
    pre_fault_reactive_power_mvar: float
    final_reactive_power_mvar: float
    reactive_power_error_mvar: float
    maximum_reactive_power_absolute_error_mvar: float
    maximum_reactive_power_error_mvar: float
    maximum_reactive_power_error_time_s: float
    is_active_power_recovered: bool
    is_reactive_power_recovered: bool

    @property
    def is_recovered(self) -> bool:
        """Return true only when both active and reactive powers recover."""
        return self.is_active_power_recovered and self.is_reactive_power_recovered

    @property
    def status(self) -> str:
        """Return a concise, traceable terminal recovery classification."""
        if self.is_recovered:
            return "Recovered"
        if not self.is_active_power_recovered and not self.is_reactive_power_recovered:
            return "P and Q not recovered"
        if not self.is_active_power_recovered:
            return "P not recovered"
        return "Q not recovered"


def assess_plant_recovery(
    time_values: Sequence[float],
    active_power_mw: Sequence[float],
    reactive_power_mvar: Sequence[float],
    *,
    plant_name: str,
    technology: str,
    clearance_time_s: float,
    criteria: PlantRecoveryCriteria = PlantRecoveryCriteria(),
) -> PlantRecoveryAssessment:
    """Assess a plant's terminal P/Q return to its value at 0.5 s.

    ``time_values`` must describe the complete simulation.  The assessment
    validates that the post-clearance observation period exists; it retains no
    arbitrary filtering or averaging and uses the actual final simulation
    sample as the decision value.
    """
    if not time_values:
        raise ValueError("Plant recovery requires at least one time sample.")
    if not (
        len(time_values) == len(active_power_mw) == len(reactive_power_mvar)
    ):
        raise ValueError("Plant recovery time, active-power, and reactive-power lengths must match.")

    times = tuple(float(value) for value in time_values)
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("Plant recovery time values must be strictly increasing.")

    reference_time_s = criteria.pre_fault_reference_time_s
    evaluation_start_time_s = clearance_time_s + criteria.recovery_evaluation_delay_s
    final_time_s = times[-1]
    if reference_time_s < times[0] or reference_time_s > final_time_s:
        raise ValueError("The plant-recovery reference time is outside the simulation.")
    if evaluation_start_time_s > final_time_s:
        raise ValueError(
            "The simulation ends before the plant-recovery evaluation period begins."
        )

    pre_fault_p = _interpolate(times, active_power_mw, reference_time_s)
    pre_fault_q = _interpolate(times, reactive_power_mvar, reference_time_s)
    final_p = float(active_power_mw[-1])
    final_q = float(reactive_power_mvar[-1])
    active_error = final_p - pre_fault_p
    reactive_error = final_q - pre_fault_q
    post_evaluation_samples = _samples_from_time(
        times,
        active_power_mw,
        reactive_power_mvar,
        evaluation_start_time_s,
    )
    maximum_active = max(
        (
            (time, active_value - pre_fault_p)
            for time, active_value, _ in post_evaluation_samples
        ),
        key=lambda item: abs(item[1]),
    )
    maximum_reactive = max(
        (
            (time, reactive_value - pre_fault_q)
            for time, _, reactive_value in post_evaluation_samples
        ),
        key=lambda item: abs(item[1]),
    )
    return PlantRecoveryAssessment(
        plant_name=plant_name,
        technology=technology,
        reference_time_s=reference_time_s,
        evaluation_start_time_s=evaluation_start_time_s,
        final_time_s=final_time_s,
        pre_fault_active_power_mw=pre_fault_p,
        final_active_power_mw=final_p,
        active_power_error_mw=active_error,
        maximum_active_power_absolute_error_mw=abs(maximum_active[1]),
        maximum_active_power_error_mw=maximum_active[1],
        maximum_active_power_error_time_s=maximum_active[0],
        pre_fault_reactive_power_mvar=pre_fault_q,
        final_reactive_power_mvar=final_q,
        reactive_power_error_mvar=reactive_error,
        maximum_reactive_power_absolute_error_mvar=abs(maximum_reactive[1]),
        maximum_reactive_power_error_mvar=maximum_reactive[1],
        maximum_reactive_power_error_time_s=maximum_reactive[0],
        is_active_power_recovered=(
            abs(active_error) <= criteria.active_power_tolerance_mw
        ),
        is_reactive_power_recovered=(
            abs(reactive_error) <= criteria.reactive_power_tolerance_mvar
        ),
    )


def _interpolate(
    time_values: Sequence[float],
    values: Sequence[float],
    target_time_s: float,
) -> float:
    """Linearly interpolate one sampled value at an explicit time."""
    index = bisect_left(time_values, target_time_s)
    if index < len(time_values) and time_values[index] == target_time_s:
        return float(values[index])
    if index == 0 or index == len(time_values):
        raise ValueError("Interpolation target is outside the sampled time range.")
    lower_time, upper_time = time_values[index - 1], time_values[index]
    fraction = (target_time_s - lower_time) / (upper_time - lower_time)
    return float(values[index - 1]) + fraction * (
        float(values[index]) - float(values[index - 1])
    )


def _samples_from_time(
    time_values: Sequence[float],
    active_power_mw: Sequence[float],
    reactive_power_mvar: Sequence[float],
    start_time_s: float,
) -> tuple[tuple[float, float, float], ...]:
    """Return samples from an explicit start, including its interpolated value."""
    index = bisect_left(time_values, start_time_s)
    samples = tuple(
        (float(time), float(active), float(reactive))
        for time, active, reactive in zip(
            time_values[index:], active_power_mw[index:], reactive_power_mvar[index:]
        )
    )
    if index < len(time_values) and time_values[index] == start_time_s:
        return samples
    return (
        (
            start_time_s,
            _interpolate(time_values, active_power_mw, start_time_s),
            _interpolate(time_values, reactive_power_mvar, start_time_s),
        ),
        *samples,
    )
