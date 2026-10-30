"""Minimal NET parser used only to catalog published MTR and PQ_NC meters."""

from dataclasses import dataclass
from pathlib import Path
import re


_MTR_PATTERN = re.compile(
    r"^@CEN_Meter_V_f_[^;]*;(?P<name>[^;]+_MTR);", re.MULTILINE
)
_PQ_PATTERN = re.compile(
    r"^@CEN_Meter_PQ_[^;]*;(?P<name>PQ_NC_[^;]+);", re.MULTILINE
)


@dataclass(frozen=True)
class NetSignalCatalog:
    """Meter names published by one EMTP NET model.

    The NET remains a static catalog only. It is never used to reproduce or
    modify the simulation; numerical samples come exclusively from
    ``signals.npy``.
    """

    net_path: Path
    voltage_frequency_meters: tuple[str, ...]
    power_transfer_meters: tuple[str, ...]

    @classmethod
    def from_net(cls, net_path: str | Path) -> "NetSignalCatalog":
        path = Path(net_path).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".net":
            raise FileNotFoundError(f"NET model not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        mtr = tuple(match.group("name") for match in _MTR_PATTERN.finditer(text))
        transfers = tuple(match.group("name") for match in _PQ_PATTERN.finditer(text))
        if not mtr:
            raise ValueError(f"No CEN MTR voltage/frequency meters found: {path}")
        if not transfers:
            raise ValueError(f"No PQ_NC transfer meters found: {path}")
        if len(set(mtr)) != len(mtr):
            raise ValueError(f"Duplicate MTR names found in {path}")
        if len(set(transfers)) != len(transfers):
            raise ValueError(f"Duplicate PQ_NC names found in {path}")
        return cls(path, mtr, transfers)

    @property
    def signature(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Ordered static meter signature used to compare case models."""
        return self.voltage_frequency_meters, self.power_transfer_meters

    @property
    def expected_signal_names(self) -> tuple[str, ...]:
        """Return conventional analysis names expected from these meters."""
        names = []
        for meter in self.voltage_frequency_meters:
            names.extend((f"{meter}/u_rms", f"{meter}/Frequency_Meter/f"))
        for meter in self.power_transfer_meters:
            names.extend((f"{meter}/P", f"{meter}/Q"))
        return tuple(names)
