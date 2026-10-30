"""Domain models for the deterministic extreme-case comparison."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtremeCaseDefinition:
    """One deliberately selected common-controller operating point."""

    directory_name: str
    label: str
    fn_hz: float
    xi: float

    def __post_init__(self) -> None:
        if not self.directory_name.strip() or not self.label.strip():
            raise ValueError("Extreme-case names must not be empty.")
        if self.fn_hz <= 0.0:
            raise ValueError("Natural frequency must be positive.")
        if self.xi <= 0.0:
            raise ValueError("Damping ratio must be positive.")


EXTREME_CASE_DEFINITIONS = (
    ExtremeCaseDefinition("Fronte Lenta", "Slow boundary", 2.0, 0.60),
    ExtremeCaseDefinition("Rápido", "Fast", 20.0, 0.95),
    ExtremeCaseDefinition("Frontera Rápida", "Fast boundary", 20.0, 0.60),
)
