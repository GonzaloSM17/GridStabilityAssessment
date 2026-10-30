"""Native signal-name index for raw deterministic-case NPY results."""

from dataclasses import dataclass
import json
from pathlib import Path

from extreme_cases.net_catalog import NetSignalCatalog
from extreme_cases.npy_signals import NpySignalMatrix, SignalLayout


@dataclass(frozen=True)
class SignalIndex:
    """Validated native mapping from signal name to raw NPY column index."""

    path: Path
    column_by_signal: dict[str, int]

    @classmethod
    def from_json(cls, path: str | Path) -> "SignalIndex":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Signal index not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Signal index JSON must contain an object.")
        if payload.get("time") != 0:
            raise ValueError("Signal index must map the time column to index 0.")
        mapping = {
            str(name): int(index)
            for name, index in payload.items()
            if name != "time"
        }
        if not mapping:
            raise ValueError("Signal index does not contain any named signals.")
        indices = tuple(mapping.values())
        if len(set(mapping)) != len(mapping) or len(set(indices)) != len(indices):
            raise ValueError("Signal index contains duplicate names or column indices.")
        if set(indices) != set(range(1, len(indices) + 1)):
            raise ValueError(
                "Signal index columns must be consecutive one-based indices after time."
            )
        return cls(source, mapping)

    @property
    def signal_count(self) -> int:
        return len(self.column_by_signal)

    def bind(self, matrix: NpySignalMatrix) -> SignalLayout:
        """Bind this exact native index to one signals.npy matrix."""
        if self.signal_count != matrix.signal_count:
            raise ValueError(
                "Signal index and signals NPY have different signal counts: "
                f"{self.signal_count} versus {matrix.signal_count}."
            )
        layout = SignalLayout.from_column_mapping(self.column_by_signal)
        layout.validate(self.column_by_signal, matrix)
        return layout

    def validate_net_catalog(self, catalog: NetSignalCatalog) -> None:
        """Check that MTR/PQ monitors from the NET are present in this index."""
        missing = tuple(
            name
            for name in catalog.expected_signal_names
            if name not in self.column_by_signal
        )
        if missing:
            raise ValueError(
                "Signal index is missing NET-published signals: "
                + ", ".join(missing[:10])
            )
