"""Combined contingency dynamic-stability decision for one simulation run."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicStabilityAssessment:
    """Combine voltage and active-power transfer modal-damping results for one run."""

    run_id: int
    status: str
    voltage_status: str
    modal_damping_status: str
    angular_status: str | None = None
    reason: str | None = None


def assess_dynamic_stability(
    run_id: int,
    voltage_status: str,
    modal_damping_status: str,
    angular_status: str | None = None,
) -> DynamicStabilityAssessment:
    """Apply active voltage/damping criteria and an optional angular criterion.

    Focused single-run assessments supply the centre-of-inertia angular result.
    Monte Carlo uses assess_monte_carlo_stability instead, with its own binary
    voltage-or-Growing rule and no angular evaluation.
    """
    if voltage_status == "Not evaluated":
        status = "Not evaluated"
        reason = "Voltage stability was not evaluated."
    elif modal_damping_status in {"Not measurable", "Not applicable"}:
        status = "Not evaluated"
        reason = "Active-power transfer modal damping was not evaluated."
    elif voltage_status == "Unstable":
        status = "Unstable"
        reason = "Voltage stability criteria failed."
    elif modal_damping_status != "Damped":
        status = "Unstable"
        reason = "Active-power transfer modal damping criteria failed."
    elif angular_status == "Not evaluated":
        status = "Not evaluated"
        reason = "Angular stability was not evaluated."
    elif angular_status not in {None, "Stable"}:
        status = "Unstable"
        reason = "Angular stability criterion failed."
    else:
        status = "Stable"
        reason = None
    return DynamicStabilityAssessment(
        run_id=run_id,
        status=status,
        voltage_status=voltage_status,
        modal_damping_status=modal_damping_status,
        angular_status=angular_status,
        reason=reason,
    )


def assess_monte_carlo_stability(
    run_id: int,
    voltage_status: str,
    modal_damping_status: str,
    angular_status: str | None = None,
) -> DynamicStabilityAssessment:
    """Apply the Monte Carlo voltage, damping, and optional angular criteria.

    The binary Stable group includes ``Damped under``. A known voltage, Growing,
    or angular failure is decisive even when another component is unavailable.
    ``angular_status=None`` preserves the historical no-angular assessment for
    compatibility with previously generated Monte Carlo outputs.
    """
    if voltage_status == "Unstable":
        status, reason = "Unstable", "Voltage stability criteria failed."
    elif modal_damping_status == "Growing":
        status, reason = "Unstable", "Active-power transfer oscillations are Growing."
    elif angular_status == "Unstable":
        status, reason = "Unstable", "Angular stability criterion failed."
    elif voltage_status != "Stable":
        status, reason = "Not evaluated", "Voltage stability was not evaluated."
    elif modal_damping_status not in {"Damped", "Damped under"}:
        status, reason = "Not evaluated", "Active-power transfer modal damping was not evaluated."
    elif angular_status == "Not evaluated":
        status, reason = "Not evaluated", "Angular stability was not evaluated."
    else:
        status = "Stable"
        reason = ("Positive damping below 5% (Damped under); included in the Stable group."
                  if modal_damping_status == "Damped under" else None)
    return DynamicStabilityAssessment(
        run_id, status, voltage_status, modal_damping_status,
        angular_status=angular_status, reason=reason,
    )
