"""Minimal study adapter exposing extreme cases to the Monte Carlo pipeline."""

from dataclasses import dataclass

from data.run import MonteCarloRun
from data.signals import Bus, PowerTransfer, SignalCatalog


@dataclass(frozen=True)
class ExtremeCasesRunContext:
    """Ordered deterministic cases presented through the batch-analysis API."""

    runs: tuple[MonteCarloRun, ...]


@dataclass
class ExtremeCasesStudy:
    """Signal catalog and runs required by non-spatial Monte Carlo artefacts."""

    context: ExtremeCasesRunContext
    signal_catalog: SignalCatalog
    assets: object | None = None

    @classmethod
    def from_runs(cls, runs: tuple[MonteCarloRun, ...]) -> "ExtremeCasesStudy":
        dataset = next((run.dataset for run in runs if run.dataset is not None), None)
        if dataset is None:
            raise ValueError("Extreme cases do not contain a readable signal dataset.")
        return cls(ExtremeCasesRunContext(runs), SignalCatalog.from_dataset(dataset))

    def bus(self, name: str) -> Bus:
        return self.signal_catalog.bus(name)

    @property
    def power_transfers(self) -> tuple[PowerTransfer, ...]:
        return self.signal_catalog.power_transfers
