"""Parquet-backed time-series models for Monte Carlo results."""

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TimeSeries:
    """Represents one named signal in a time-series dataset."""

    name: str


@dataclass
class TimeSeriesDataset:
    """Represents the Parquet result produced by one simulation run."""

    parquet_path: Path
    time_column: str | None = None
    series: list[TimeSeries] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.parquet_path = Path(self.parquet_path).expanduser().resolve()

        if not self.parquet_path.is_file():
            raise FileNotFoundError(f"Parquet result not found: {self.parquet_path}")
        if self.parquet_path.suffix.lower() != ".parquet":
            raise ValueError(
                f"Dataset path must point to a Parquet file: {self.parquet_path}"
            )

    def inspect(self) -> None:
        """Read Parquet metadata and build the dataset's time-series index."""
        import polars as pl

        column_names = pl.read_parquet_schema(self.parquet_path).names()

        if "time" not in column_names:
            raise ValueError(
                f"Parquet result does not contain a 'time' column: {self.parquet_path}"
            )

        self.time_column = "time"
        self.series = [
            TimeSeries(name=column_name)
            for column_name in column_names
            if column_name != self.time_column
        ]

    def get_series(self, name: str) -> TimeSeries:
        """Return one time series by its exact Parquet column name."""
        if self.time_column is None:
            self.inspect()

        for series in self.series:
            if series.name == name:
                return series

        raise KeyError(f"Time series not found: {name}")

    def read_series(self, name: str):
        """Read only time and one requested time series from the Parquet file."""
        series = self.get_series(name)

        import polars as pl

        return (
            pl.scan_parquet(self.parquet_path)
            .select([self.time_column, series.name])
            .collect()
        )

    def read_series_window(
        self,
        names: list[str] | tuple[str, ...],
        start_time_s: float,
        end_time_s: float,
    ):
        """Read several named series within an inclusive time window."""
        if start_time_s >= end_time_s:
            raise ValueError("The time-window start must precede its end.")
        series = [self.get_series(name) for name in names]

        import polars as pl

        return (
            pl.scan_parquet(self.parquet_path)
            .filter(
                (pl.col(self.time_column) >= start_time_s)
                & (pl.col(self.time_column) <= end_time_s)
            )
            .select([self.time_column, *(item.name for item in series)])
            .collect()
        )

    def maximum_time_s(self) -> float:
        """Return the final simulation time without loading any signal series."""
        if self.time_column is None:
            self.inspect()

        import polars as pl

        maximum_time = (
            pl.scan_parquet(self.parquet_path)
            .select(pl.col(self.time_column).max().alias("maximum_time"))
            .collect()
            .item(0, "maximum_time")
        )
        if maximum_time is None:
            raise ValueError(f"Dataset contains no time samples: {self.parquet_path}")
        return float(maximum_time)

    def read_series_group(self, names: list[str] | tuple[str, ...]):
        """Read time and several named series without applying a time window."""
        series = [self.get_series(name) for name in names]

        import polars as pl

        return (
            pl.scan_parquet(self.parquet_path)
            .select([self.time_column, *(item.name for item in series)])
            .collect()
        )

