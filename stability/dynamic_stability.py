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

    Monte Carlo currently omits ``angular_status`` and therefore preserves its
    existing voltage-plus-damping decision. A focused single-run assessment can
    provide it to add the strict centre-of-inertia angular result.
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
