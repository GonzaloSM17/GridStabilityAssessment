"""Native spatial-controller and Grid Code compliance result models."""
from dataclasses import dataclass, replace
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

    def __post_init__(self) -> None:
        """Expose Grid Code compliance as the spatial classification status.

        Spatial maps, distributions, clustering and characterization predate the
        separate technical diagnosis and consistently consume ``status``. Keep
        their data contract intact while retaining the diagnosis in the added
        ``dynamic_status`` and ``dynamic_reason`` fields of the assessment.
        """
        if self.assessment.status != self.assessment.grid_code_status:
            object.__setattr__(
                self,
                "assessment",
                replace(
                    self.assessment,
                    status=self.assessment.grid_code_status,
                    reason=self.assessment.grid_code_reason,
                ),
            )


@dataclass(frozen=True)
class SpatialStabilityAnalysis:
    points: tuple[SpatialControllerPoint, ...]
    cases: tuple[SpatialCaseAssessment, ...]
    excluded_devices: tuple[dict, ...] = ()
    skipped_runs: tuple["SpatialSkippedRun", ...] = ()
    voltage_scope: str = "All available 220/500 kV buses"
    monte_carlo_policy: bool = False

    @property
    def include_angle(self):
        return any(case.assessment.angular_status is not None for case in self.cases)

    @property
    def criterion_description(self):
        return (
            "Grid Code compliance: voltage, modal damping and angular criteria "
            "are all required. Damped under and Growing are non-compliant; "
            "missing required evidence is not evaluated."
        )


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
