"""Native spatial-controller and stability result models."""
from dataclasses import dataclass
from pathlib import Path
from stability.dynamic_stability import DynamicStabilityAssessment


@dataclass(frozen=True)
class SpatialControllerPoint:
    case_id: str
    run_id: int
    device_id: str
    generator_name: str
    asset_name: str
    x: float
    y: float
    nominal_power_mva: float
    kp: float
    ki: float
    xi: float
    fn_hz: float


@dataclass(frozen=True)
class SpatialCaseAssessment:
    case_id: str
    assessment: DynamicStabilityAssessment
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpatialStabilityAnalysis:
    points: tuple[SpatialControllerPoint, ...]
    cases: tuple[SpatialCaseAssessment, ...]
    excluded_devices: tuple[dict, ...] = ()
    skipped_runs: tuple["SpatialSkippedRun", ...] = ()
    voltage_scope: str = "All available 220/500 kV buses"

    @property
    def include_angle(self):
        return any(case.assessment.angular_status is not None for case in self.cases)

    @property
    def criterion_description(self):
        if self.include_angle:
            return "Native dynamic stability: voltage + modal damping + angle"
        return ("Monte Carlo: Unstable if voltage is Unstable or modal damping is Growing; "
                "Damped under is grouped with Stable; angular criterion is not evaluated")


@dataclass(frozen=True)
class SpatialSkippedRun:
    """One run omitted from spatial maps without invalidating other analyses."""

    case_id: str
    reason: str


@dataclass(frozen=True)
class SpatialStabilityResult:
    output_directory: Path
    analysis: SpatialStabilityAnalysis | None
    figure_paths: tuple[Path, ...]
    status: str = "Completed"
    reason: str | None = None
