"""Parquet-backed time-series models for Monte Carlo results."""

from dataclasses import dataclass, field
from pathlib import Path

from signal_analysis import (
    InitializationAssessment,
    InitializationCriteria,
    StabilityAssessment,
    StabilityCriteria,
    assess_initialization,
    assess_stability,
)


@dataclass
class TimeSeries:
    """Represents one named signal in a time-series dataset."""

    name: str


@dataclass
class SignalDeviation:
    """Maximum absolute deviation of one signal within a time window."""

    max_absolute_delta: float
    signed_delta_at_peak: float
    peak_time_s: float
    reference_value: float


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

    def calculate_max_absolute_delta(
        self,
        signal_name: str,
        start_time: float,
        end_time: float,
        reference_time: float = 0.0,
    ) -> SignalDeviation:
        """Calculate a signal's largest absolute delta within a time window."""
        if start_time > end_time:
            raise ValueError("start_time must not be greater than end_time.")

        import polars as pl

        data = self.read_series(signal_name)
        reference_rows = data.filter(pl.col("time") == reference_time)
        if reference_rows.is_empty():
            raise ValueError(
                f"Reference time {reference_time} is not present for {signal_name}."
            )

        reference_value = float(reference_rows[signal_name][0])
        window = data.filter(
            (pl.col("time") >= start_time) & (pl.col("time") <= end_time)
        )
        if window.is_empty():
            raise ValueError(
                f"No samples for {signal_name} between {start_time} and {end_time} s."
            )

        delta_column = "_delta"
        absolute_delta_column = "_absolute_delta"
        peak = (
            window.with_columns(
                (pl.col(signal_name) - reference_value).alias(delta_column)
            )
            .with_columns(pl.col(delta_column).abs().alias(absolute_delta_column))
            .sort(absolute_delta_column, descending=True)
            .row(0, named=True)
        )

        return SignalDeviation(
            max_absolute_delta=float(peak[absolute_delta_column]),
            signed_delta_at_peak=float(peak[delta_column]),
            peak_time_s=float(peak["time"]),
            reference_value=reference_value,
        )

    def evaluate_initialization(
        self,
        signal_name: str,
        criteria: InitializationCriteria | None = None,
    ) -> InitializationAssessment:
        """Evaluate whether a signal settles before the contingency begins."""
        data = self.read_series(signal_name)
        return assess_initialization(
            data["time"].to_list(),
            data[signal_name].to_list(),
            criteria,
        )

    def evaluate_stability(
        self,
        signal_name: str,
        criteria: StabilityCriteria | None = None,
    ) -> StabilityAssessment:
        """Evaluate whether a signal settles after the contingency clearance."""
        data = self.read_series(signal_name)
        return assess_stability(
            data["time"].to_list(),
            data[signal_name].to_list(),
            criteria,
        )
