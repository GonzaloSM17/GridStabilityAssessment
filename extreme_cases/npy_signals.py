"""Raw NPY signal access and explicit named-layout binding."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data.time_series import TimeSeries


@dataclass(frozen=True)
class NpySignalMatrix:
    """One raw ``signals.npy`` matrix, kept memory-mapped until read."""

    path: Path
    row_count: int
    column_count: int
    dtype: str
    time_start_s: float
    time_end_s: float
    time_step_s: float

    @classmethod
    def from_file(cls, path: str | Path) -> "NpySignalMatrix":
        import numpy as np

        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Signals NPY not found: {resolved}")
        values = np.load(resolved, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("Signals NPY must be a two-dimensional time-plus-signal matrix.")
        if values.dtype.kind != "f":
            raise ValueError("Signals NPY must use a floating-point dtype.")
        time = values[:, 0]
        time_step = float(np.median(np.diff(time)))
        if time_step <= 0.0 or not np.all(np.diff(time) > 0.0):
            raise ValueError("Signals NPY time column must be strictly increasing.")
        return cls(
            path=resolved,
            row_count=int(values.shape[0]),
            column_count=int(values.shape[1]),
            dtype=str(values.dtype),
            time_start_s=float(time[0]),
            time_end_s=float(time[-1]),
            time_step_s=time_step,
        )

    @property
    def signal_count(self) -> int:
        """Return raw numeric channels excluding the time column."""
        return self.column_count - 1

    @property
    def schema_signature(self) -> tuple[int, int, str, float, float, float]:
        """Return the layout identity used to compare deterministic cases."""
        return (
            self.row_count,
            self.column_count,
            self.dtype,
            self.time_start_s,
            self.time_end_s,
            self.time_step_s,
        )

    def read_columns(self, column_indices: Iterable[int]):
        """Read time plus selected raw signal columns into NumPy arrays."""
        import numpy as np

        indices = tuple(int(index) for index in column_indices)
        if any(index <= 0 or index >= self.column_count for index in indices):
            raise IndexError(
                f"Signal columns must be within 1 and {self.column_count - 1}."
            )
        values = np.load(self.path, mmap_mode="r", allow_pickle=False)
        return np.asarray(values[:, 0]), {
            index: np.asarray(values[:, index]) for index in indices
        }


@dataclass(frozen=True)
class SignalLayout:
    """Explicit mapping from conventional signal name to a NPY column index.

    NPY files store only numeric matrices. The extreme-case flow creates this
    layout from the native ordered ``signal_index.json`` catalog.
    """

    column_by_signal: dict[str, int]

    @classmethod
    def from_ordered_signal_names(cls, signal_names: Iterable[str]) -> "SignalLayout":
        """Bind ordered physical signal names to one-based NPY columns."""
        names = tuple(str(name) for name in signal_names)
        if len(set(names)) != len(names):
            raise ValueError("The reference signal schema contains duplicate names.")
        return cls({name: index for index, name in enumerate(names, start=1)})

    @classmethod
    def from_column_mapping(cls, column_by_signal: dict[str, int]) -> "SignalLayout":
        """Create a layout from the native signal-name-to-column mapping."""
        return cls({str(name): int(index) for name, index in column_by_signal.items()})

    def validate(self, expected_signal_names: Iterable[str], matrix: NpySignalMatrix) -> None:
        expected = tuple(expected_signal_names)
        missing = tuple(name for name in expected if name not in self.column_by_signal)
        unexpected = tuple(name for name in self.column_by_signal if name not in expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={', '.join(missing[:5])}")
            if unexpected:
                details.append(f"unexpected={', '.join(unexpected[:5])}")
            raise ValueError("Signal layout is incomplete: " + "; ".join(details))
        indices = tuple(self.column_by_signal[name] for name in expected)
        if len(set(indices)) != len(indices):
            raise ValueError("Each expected signal must use a distinct NPY column.")
        if any(index <= 0 or index >= matrix.column_count for index in indices):
            raise ValueError(
                f"Signal layout contains a column outside 1..{matrix.column_count - 1}."
            )


class NamedNpyDataset:
    """Parquet-compatible subset used by existing shared analysis artefacts."""

    def __init__(self, matrix: NpySignalMatrix, layout: SignalLayout) -> None:
        self.matrix = matrix
        self.layout = layout
        self.time_column = "time"
        self.series = [TimeSeries(name=name) for name in layout.column_by_signal]

    def inspect(self) -> None:
        """Match the metadata-discovery API exposed by ``TimeSeriesDataset``."""

    def get_series(self, name: str) -> TimeSeries:
        for series in self.series:
            if series.name == name:
                return series
        raise KeyError(f"Time series not found: {name}")

    def read_series(self, name: str):
        return self.read_series_group((name,))

    def read_series_group(self, names: Iterable[str]):
        import polars as pl

        requested = tuple(names)
        for name in requested:
            self.get_series(name)
        indices = tuple(self.layout.column_by_signal[name] for name in requested)
        time, columns = self.matrix.read_columns(indices)
        return pl.DataFrame(
            {
                "time": time,
                **{
                    name: columns[index]
                    for name, index in zip(requested, indices)
                },
            }
        )

    def read_series_window(self, names: Iterable[str], start_time_s: float, end_time_s: float):
        if start_time_s >= end_time_s:
            raise ValueError("The time-window start must precede its end.")
        import polars as pl

        return self.read_series_group(names).filter(
            (pl.col("time") >= start_time_s) & (pl.col("time") <= end_time_s)
        )

    def maximum_time_s(self) -> float:
        return self.matrix.time_end_s
