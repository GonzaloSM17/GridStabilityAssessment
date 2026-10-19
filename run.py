"""Representation of one Monte Carlo simulation run."""

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
import re
from typing import TYPE_CHECKING

from time_series import TimeSeriesDataset

if TYPE_CHECKING:
    from assets import Assets


RUN_DIRECTORY_PATTERN = re.compile(r"run_(\d{6})$")


@dataclass
class WeightedControllerParameters:
    """Power-weighted controller parameters for one simulation run."""

    fn_weighted_hz: float
    xi_weighted: float
    total_nominal_power_mva: float
    unit_count: int


@dataclass
class MonteCarloRun:
    """Represents one published simulation run."""

    run_id: int
    run_path: Path
    dataset: TimeSeriesDataset | None

    @classmethod
    def from_directory(cls, run_path: str | Path) -> "MonteCarloRun":
        """Create a run from its directory and locate its Parquet result."""
        path = Path(run_path).expanduser().resolve()

        if not path.is_dir():
            raise NotADirectoryError(f"Run directory not found: {path}")

        match = RUN_DIRECTORY_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(
                "Run directory names must follow the format 'run_XXXXXX': "
                f"{path.name}"
            )

        parquet_files = sorted(path.glob("*.parquet"))
        if len(parquet_files) > 1:
            raise ValueError(f"Run contains more than one Parquet file: {path}")

        dataset = TimeSeriesDataset(parquet_files[0]) if parquet_files else None

        return cls(
            run_id=int(match.group(1)),
            run_path=path,
            dataset=dataset,
        )

    def read_parameters(self):
        """Read the parameter table saved with this run."""
        parameters_path = self.run_path / "parameters.csv"
        if not parameters_path.is_file():
            raise FileNotFoundError(f"Run parameters not found: {parameters_path}")

        import polars as pl

        return pl.read_csv(parameters_path, separator=";")

    def calculate_weighted_controller_parameters(
        self,
        assets: "Assets",
    ) -> WeightedControllerParameters:
        """Calculate nominal-power-weighted fn and xi for this run."""
        import polars as pl

        parameters = self.read_parameters()
        required_columns = {"GeneratorName", "InService", "Kp", "Ki"}
        missing_columns = required_columns.difference(parameters.columns)
        if missing_columns:
            raise ValueError(
                f"Run parameters are missing columns: {sorted(missing_columns)}"
            )

        active_parameters = parameters.filter(
            (pl.col("InService") == 1)
            & pl.col("Kp").is_not_null()
            & pl.col("Ki").is_not_null()
        )
        if active_parameters.is_empty():
            raise ValueError(f"Run {self.run_id} has no active controller parameters.")

        total_nominal_power_mva = 0.0
        weighted_fn_hz = 0.0
        weighted_xi = 0.0
        missing_assets: list[str] = []

        for parameter in active_parameters.iter_rows(named=True):
            asset_name = str(parameter["GeneratorName"])
            try:
                nominal_power_mva = assets.get_nominal_power_mva(asset_name)
            except KeyError:
                missing_assets.append(asset_name)
                continue

            kp = float(parameter["Kp"])
            ki = float(parameter["Ki"])
            if ki <= 0.0:
                raise ValueError(
                    f"Run {self.run_id} has a non-positive Ki for {asset_name}."
                )
            if nominal_power_mva <= 0.0:
                raise ValueError(
                    f"Asset {asset_name} has non-positive nominal power."
                )

            fn_hz = sqrt(ki) / (2.0 * pi)
            xi = kp / (2.0 * sqrt(ki))

            total_nominal_power_mva += nominal_power_mva
            weighted_fn_hz += nominal_power_mva * fn_hz
            weighted_xi += nominal_power_mva * xi

        if missing_assets:
            missing_asset_names = ", ".join(sorted(set(missing_assets)))
            raise KeyError(
                f"Run {self.run_id} has assets missing from the workbook: "
                f"{missing_asset_names}"
            )
        if total_nominal_power_mva == 0.0:
            raise ValueError(f"Run {self.run_id} has no usable nominal power.")

        return WeightedControllerParameters(
            fn_weighted_hz=weighted_fn_hz / total_nominal_power_mva,
            xi_weighted=weighted_xi / total_nominal_power_mva,
            total_nominal_power_mva=total_nominal_power_mva,
            unit_count=active_parameters.height,
        )
