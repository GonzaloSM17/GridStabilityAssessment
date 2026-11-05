"""Scenario-level Grid Code compliance and dynamic-stability conclusions.

The same voltage, modal-damping and angular measurements support two distinct
conclusions. Grid Code compliance applies every configured threshold. The
dynamic-stability diagnosis describes whether the response settles, retains
synchronism and contains only decaying electromechanical modes.
"""

from dataclasses import dataclass


GRID_CODE_COMPLIANT = "Compliant"
GRID_CODE_NON_COMPLIANT = "Non-compliant"
GRID_CODE_NOT_EVALUATED = "Not evaluated"

DYNAMIC_STABLE = "Stable"
DYNAMIC_UNSTABLE = "Unstable"
DYNAMIC_INCONCLUSIVE = "Inconclusive"


@dataclass(frozen=True)
class DynamicStabilityAssessment:
    """Traceable Grid Code and technical conclusion for one simulation run.

    ``status`` and ``reason`` are the dynamic-diagnosis fields. New population
    consumers use ``grid_code_status`` and ``grid_code_reason``.
    """

    run_id: int
    status: str
    voltage_status: str
    modal_damping_status: str
    angular_status: str | None = None
    reason: str | None = None
    grid_code_status: str = GRID_CODE_NOT_EVALUATED
    grid_code_reason: str | None = None
    dynamic_status: str | None = None
    dynamic_reason: str | None = None

    @property
    def is_grid_code_compliant(self) -> bool:
        return self.grid_code_status == GRID_CODE_COMPLIANT

    @property
    def is_dynamically_stable(self) -> bool:
        return (self.dynamic_status or self.status) == DYNAMIC_STABLE


def assess_dynamic_stability(
    run_id: int,
    voltage_status: str,
    modal_damping_status: str,
    angular_status: str | None = None,
) -> DynamicStabilityAssessment:
    """Assess Grid Code compliance and dynamic stability for one run.

    Voltage, damping and angular criteria all participate in Grid Code
    compliance. A positive but under-limit damping ratio is non-compliant but
    dynamically stable; non-positive damping is a growing, unstable mode.
    """
    grid_code_failures: list[str] = []
    grid_code_missing: list[str] = []
    if voltage_status == "Unstable":
        grid_code_failures.append("Voltage criteria failed.")
    elif voltage_status != "Stable":
        grid_code_missing.append("Voltage was not evaluated.")
    if modal_damping_status == "Growing":
        grid_code_failures.append("Active-power transfer damping is growing.")
    elif modal_damping_status == "Damped under":
        grid_code_failures.append(
            "Active-power transfer damping is below the Grid Code limit."
        )
    elif modal_damping_status != "Damped":
        grid_code_missing.append("Active-power transfer damping was not evaluated.")
    if angular_status == "Unstable":
        grid_code_failures.append("Angular separation criterion failed.")
    elif angular_status != "Stable":
        grid_code_missing.append("Angular stability was not evaluated.")

    if grid_code_failures:
        grid_code_status = GRID_CODE_NON_COMPLIANT
        grid_code_reason = " ".join(grid_code_failures)
    elif grid_code_missing:
        grid_code_status = GRID_CODE_NOT_EVALUATED
        grid_code_reason = " ".join(grid_code_missing)
    else:
        grid_code_status = GRID_CODE_COMPLIANT
        grid_code_reason = None

    dynamic_failures: list[str] = []
    dynamic_missing: list[str] = []
    if voltage_status == "Unstable":
        dynamic_failures.append("Voltage did not settle under the assessment rule.")
    elif voltage_status != "Stable":
        dynamic_missing.append("Voltage was not evaluated.")
    if modal_damping_status == "Growing":
        dynamic_failures.append("Active-power transfer oscillations are growing.")
    elif modal_damping_status not in {"Damped", "Damped under"}:
        dynamic_missing.append("Active-power transfer damping was not evaluated.")
    if angular_status == "Unstable":
        dynamic_failures.append("Angular synchronism criterion failed.")
    elif angular_status != "Stable":
        dynamic_missing.append("Angular stability was not evaluated.")

    if dynamic_failures:
        status = DYNAMIC_UNSTABLE
        reason = " ".join(dynamic_failures)
    elif dynamic_missing:
        status = DYNAMIC_INCONCLUSIVE
        reason = " ".join(dynamic_missing)
    else:
        status = DYNAMIC_STABLE
        reason = (
            "Positive damping below the Grid Code limit (Damped under)."
            if modal_damping_status == "Damped under"
            else None
        )

    return DynamicStabilityAssessment(
        run_id=run_id,
        status=status,
        voltage_status=voltage_status,
        modal_damping_status=modal_damping_status,
        angular_status=angular_status,
        reason=reason,
        grid_code_status=grid_code_status,
        grid_code_reason=grid_code_reason,
        dynamic_status=status,
        dynamic_reason=reason,
    )


def assess_monte_carlo_stability(
    run_id: int,
    voltage_status: str,
    modal_damping_status: str,
    angular_status: str | None = None,
) -> DynamicStabilityAssessment:
    """Backward-compatible entry point using the common scenario assessment."""
    return assess_dynamic_stability(
        run_id,
        voltage_status,
        modal_damping_status,
        angular_status,
    )
