"""The complete input model for one Monte Carlo analysis study."""

from dataclasses import dataclass, field

from data.assets import Assets
from data.signals import Bus, PowerTransfer, SignalCatalog
from montecarlo.context import MonteCarloContext


@dataclass
class MonteCarloStudy:
    """Combine Monte Carlo runs, Assets metadata, and the bus signal catalog."""

    context: MonteCarloContext
    assets: Assets
    signal_catalog: SignalCatalog = field(init=False)

    def __post_init__(self) -> None:
        representative_dataset = next(
            (run.dataset for run in self.context.runs if run.dataset is not None),
            None,
        )
        if representative_dataset is None:
            raise ValueError("The study does not contain a Parquet result dataset.")
        self.signal_catalog = SignalCatalog.from_dataset(representative_dataset)

    def bus(self, name: str) -> Bus:
        """Return one bus discovered in the published study signals."""
        return self.signal_catalog.bus(name)

    @property
    def power_transfers(self) -> tuple[PowerTransfer, ...]:
        """Return the explicitly published active-power transfer monitors."""
        return self.signal_catalog.power_transfers
