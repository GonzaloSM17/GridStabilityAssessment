"""Scenario and evaluation-profile models for contingency analyses."""

from dataclasses import dataclass

from stability.initialization import InitializationCriteria
from stability.voltage import FrequencyCriteria, StabilityCriteria
from stability.power_transfer import DampingCriteria, HighFrequencyCriteria


@dataclass(frozen=True)
class ContingencyScenario:
    """Describe the physical contingency represented by one Monte Carlo study."""

    name: str
    severity: int
    fault_type: str
    fault_time_s: float
    clearance_time_s: float
    location: str | None = None
    operating_state: str = "normal"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A contingency scenario must have a name.")
        if self.severity <= 0:
            raise ValueError("Contingency severity must be positive.")
        if not self.fault_type.strip():
            raise ValueError("A contingency scenario must identify its fault type.")
        if self.fault_time_s < 0.0:
            raise ValueError("Fault time cannot be negative.")
        if self.clearance_time_s <= self.fault_time_s:
            raise ValueError("Clearance time must follow the fault time.")


@dataclass(frozen=True)
class EvaluationProfile:
    """Criteria applied to one explicitly defined contingency scenario.

    The scenario identifies the disturbance; the three criteria objects contain
    the active voltage, frequency, and transfer-damping rules. Future severity
    profiles can be introduced without changing any analysis implementation.
    """

    scenario: ContingencyScenario
    initialization_criteria: InitializationCriteria
    voltage_criteria: StabilityCriteria
    frequency_criteria: FrequencyCriteria
    transfer_damping_criteria: DampingCriteria
    high_frequency_criteria: HighFrequencyCriteria

    @classmethod
    def for_scenario(cls, scenario: ContingencyScenario) -> "EvaluationProfile":
        """Create the current criteria profile while preserving scenario timing."""
        initialization_window_duration_s = 0.5
        pre_fault_power_window_duration_s = 0.1
        response_window_after_clearance_s = 2.0 - 1.12
        damping_start_after_clearance_s = 2.0 - 1.12
        return cls(
            scenario=scenario,
            initialization_criteria=InitializationCriteria(
                evaluation_start_time_s=max(
                    0.0, scenario.fault_time_s - initialization_window_duration_s
                ),
                evaluation_end_time_s=scenario.fault_time_s,
            ),
            voltage_criteria=StabilityCriteria(
                fault_time_s=scenario.fault_time_s,
                clearance_time_s=scenario.clearance_time_s,
                response_end_time_s=(
                    scenario.clearance_time_s + response_window_after_clearance_s
                ),
            ),
            frequency_criteria=FrequencyCriteria(),
            transfer_damping_criteria=DampingCriteria(
                pre_fault_start_time_s=(
                    scenario.fault_time_s - pre_fault_power_window_duration_s
                ),
                fault_time_s=scenario.fault_time_s,
                clearance_time_s=scenario.clearance_time_s,
                analysis_start_time_s=(
                    scenario.clearance_time_s + damping_start_after_clearance_s
                ),
                mode_minimum_frequency_hz=0.1,
                mode_maximum_frequency_hz=2.5,
            ),
            high_frequency_criteria=HighFrequencyCriteria(
                minimum_frequency_hz=2.5,
                maximum_frequency_hz=50.0,
                analysis_start_time_s=(
                    scenario.clearance_time_s + damping_start_after_clearance_s
                ),
            ),
        )


DEFAULT_CONTINGENCY_SCENARIO = ContingencyScenario(
    name="CC-2PH-G severity 4",
    severity=4,
    fault_type="two_phase_to_ground",
    fault_time_s=1.0,
    clearance_time_s=1.12,
)
