"""Representation of one Monte Carlo simulation run."""

from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
import re
from typing import TYPE_CHECKING

from data.time_series import TimeSeriesDataset

if TYPE_CHECKING:
    from data.assets import Assets


RUN_DIRECTORY_PATTERN = re.compile(r"run_(\d{6})$")


@dataclass(frozen=True)
class ActiveControllerParameters:
    """Derived controller parameters for one active machine in a run."""

    generator_name: str
    fn_hz: float
    xi: float
    nominal_power_mva: float


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

    def read_active_controller_parameters(
        self,
        assets: "Assets",
    ) -> list[ActiveControllerParameters]:
        """Return fn and xi for every active machine in this run."""
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

        controllers: list[ActiveControllerParameters] = []
        missing_assets: list[str] = []
        for parameter in active_parameters.iter_rows(named=True):
            generator_name = str(parameter["GeneratorName"])
            try:
                nominal_power_mva = assets.get_nominal_power_mva(generator_name)
            except KeyError:
                missing_assets.append(generator_name)
                continue

            kp = float(parameter["Kp"])
            ki = float(parameter["Ki"])
            if ki <= 0.0:
                raise ValueError(
                    f"Run {self.run_id} has a non-positive Ki for {generator_name}."
                )
            if nominal_power_mva <= 0.0:
                raise ValueError(
                    f"Asset {generator_name} has non-positive nominal power."
                )

            controllers.append(
                ActiveControllerParameters(
                    generator_name=generator_name,
                    fn_hz=sqrt(ki) / (2.0 * pi),
                    xi=kp / (2.0 * sqrt(ki)),
                    nominal_power_mva=nominal_power_mva,
                )
            )

        if missing_assets:
            missing_asset_names = ", ".join(sorted(set(missing_assets)))
            raise KeyError(
                f"Run {self.run_id} has assets missing from the workbook: "
                f"{missing_asset_names}"
            )
        if not controllers:
            raise ValueError(f"Run {self.run_id} has no usable controller parameters.")

        return controllers
