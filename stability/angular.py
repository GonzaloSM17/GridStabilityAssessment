"""Centre-of-inertia angular-excursion calculation shared by all study flows."""

from dataclasses import dataclass
from math import ceil, floor, log10
from typing import Mapping, Sequence


ANGULAR_LIMIT_DEGREES = 120.0
ANGULAR_REFERENCE_NOTE = (
    "Angular excursions are referred to the centre of inertia calculated at every "
    "time step with the H·S weight from Assets. The ±120° lines are the active "
    "angular-separation reference limits."
)


def angular_excursion_limit(excursions: Sequence[Sequence[float]]) -> float:
    """Return a symmetric axis of at least ±140°, extended in 10° steps."""
    maximum_excursion = max(
        (abs(value) for excursion in excursions for value in excursion),
        default=0.0,
    )
    return float(max(140, ceil(maximum_excursion / 10.0) * 10))


def angular_excursion_ticks(limit_degrees: float) -> tuple[float, ...]:
    """Return symmetric readable ticks that explicitly include both axis bounds."""
    if limit_degrees <= 0.0:
        raise ValueError("The angular-excursion limit must be positive.")
    target_step = limit_degrees / 6.0
    magnitude = 10.0 ** floor(log10(target_step))
    candidates = tuple(value * magnitude for value in (1.0, 2.0, 5.0, 10.0))
    step = min(candidates, key=lambda value: abs(value - target_step))
    positive_ticks = [0.0]
    value = step
    while value < limit_degrees:
        positive_ticks.append(float(value))
        value += step
    if positive_ticks[-1] != limit_degrees:
        positive_ticks.append(float(limit_degrees))
    return tuple(-value for value in reversed(positive_ticks[1:])) + tuple(positive_ticks)


@dataclass(frozen=True)
class AngularMachineInput:
    """One machine and the H·S data needed for a COI calculation."""

    machine_name: str
    signal_name: str
    inertia_s: float
    nominal_power_mva: float

    @property
    def inertia_weight_mva_s(self) -> float:
        return self.inertia_s * self.nominal_power_mva


@dataclass(frozen=True)
class AngularMachineAssessment:
    """One machine's COI-referenced angular excursion and strict ±120° result."""

    machine_name: str
    signal_name: str
    inertia_s: float
    nominal_power_mva: float
    inertia_weight_mva_s: float
    minimum_excursion_degrees: float
    maximum_excursion_degrees: float
    maximum_absolute_excursion_degrees: float
    first_limit_violation_time_s: float | None

    @property
    def is_within_limit(self) -> bool:
        return self.first_limit_violation_time_s is None


@dataclass(frozen=True)
class AngularStabilityAnalysis:
    """Machine excursions relative to the H·S weighted centre-of-inertia angle."""

    machines: tuple[AngularMachineAssessment, ...]
    centre_of_inertia_minimum_degrees: float
    centre_of_inertia_maximum_degrees: float

    @property
    def is_stable(self) -> bool:
        return all(machine.is_within_limit for machine in self.machines)


@dataclass(frozen=True)
class AngularCalculation:
    """Full sample-by-sample centre-of-inertia calculation."""

    analysis: AngularStabilityAnalysis
    time_values: tuple[float, ...]
    centre_of_inertia: tuple[float, ...]
    excursions: tuple[tuple[float, ...], ...]


def assess_angular_stability(
    time_values: Sequence[float],
    angle_values_by_machine: Mapping[str, Sequence[float]],
    machines: Sequence[AngularMachineInput],
) -> AngularCalculation:
    """Calculate COI and every machine excursion at every supplied time sample."""
    import numpy as np

    if not machines:
        raise ValueError("At least one synchronous machine is required for COI analysis.")
    time = np.asarray(time_values, dtype=float)
    if len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("Angular stability requires strictly increasing time samples.")
    missing = [machine.machine_name for machine in machines if machine.machine_name not in angle_values_by_machine]
    if missing:
        raise KeyError(f"Machine angles are missing for: {', '.join(missing)}")
    weights = np.asarray([machine.inertia_weight_mva_s for machine in machines], dtype=float)
    if np.any(weights <= 0.0):
        raise ValueError("Every angular machine must have a positive H·S weight.")
    angles = np.asarray(
        [angle_values_by_machine[machine.machine_name] for machine in machines],
        dtype=float,
    )
    if angles.shape != (len(machines), len(time)) or not np.all(np.isfinite(angles)):
        raise ValueError("Each machine angle must be finite and cover every time sample.")
    centre_of_inertia = np.average(angles, axis=0, weights=weights)
    excursions = angles - centre_of_inertia
    assessments = []
    for machine, excursion in zip(machines, excursions):
        violation = np.flatnonzero(
            (excursion >= ANGULAR_LIMIT_DEGREES)
            | (excursion <= -ANGULAR_LIMIT_DEGREES)
        )
        assessments.append(
            AngularMachineAssessment(
                machine_name=machine.machine_name,
                signal_name=machine.signal_name,
                inertia_s=machine.inertia_s,
                nominal_power_mva=machine.nominal_power_mva,
                inertia_weight_mva_s=machine.inertia_weight_mva_s,
                minimum_excursion_degrees=float(np.min(excursion)),
                maximum_excursion_degrees=float(np.max(excursion)),
                maximum_absolute_excursion_degrees=float(np.max(np.abs(excursion))),
                first_limit_violation_time_s=(
                    float(time[violation[0]]) if len(violation) else None
                ),
            )
        )
    analysis = AngularStabilityAnalysis(
        machines=tuple(assessments),
        centre_of_inertia_minimum_degrees=float(np.min(centre_of_inertia)),
        centre_of_inertia_maximum_degrees=float(np.max(centre_of_inertia)),
    )
    return AngularCalculation(
        analysis=analysis,
        time_values=tuple(time.tolist()),
        centre_of_inertia=tuple(centre_of_inertia.tolist()),
        excursions=tuple(tuple(excursion.tolist()) for excursion in excursions),
    )
