"""Combined contingency dynamic-stability decision for one simulation run."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicStabilityAssessment:
    """Combine voltage and active-power transfer modal-damping results for one run."""

    run_id: int
    status: str
    voltage_status: str
    modal_damping_status: str
    reason: str | None = None


def assess_dynamic_stability(
    run_id: int,
    voltage_status: str,
    modal_damping_status: str,
) -> DynamicStabilityAssessment:
    """Apply the active voltage and modal-damping decision to one run."""
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
    else:
        status = "Stable"
        reason = None
    return DynamicStabilityAssessment(
        run_id=run_id,
        status=status,
        voltage_status=voltage_status,
        modal_damping_status=modal_damping_status,
        reason=reason,
    )
