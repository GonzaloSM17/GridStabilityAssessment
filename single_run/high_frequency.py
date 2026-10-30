"""Observation-only high-frequency transfer diagnostics for one simulation run."""

from dataclasses import dataclass
from math import ceil
from pathlib import Path

from data.run import MonteCarloRun
from data.signals import (
    PowerTransfer,
    SignalCatalog,
    order_power_transfers_north_to_south,
)
from stability.plotting import close_y_axis
from stability.power_transfer import (
    HighFrequencyTransferAssessment,
    assess_high_frequency_components,
    spectral_power_spectrum,
)
from stability.scenario import EvaluationProfile
from utils.figure_localization import save_figure_with_spanish_copy


@dataclass(frozen=True)
class HighFrequencyRecord:
    """One transfer and its observation-only high-frequency result."""

    transfer: PowerTransfer
    assessment: HighFrequencyTransferAssessment
    time_values: tuple[float, ...]
    power_values: tuple[float, ...]


@dataclass(frozen=True)
class HighFrequencyAnalysis:
    """High-frequency components detected without a stability decision."""

    records: tuple[HighFrequencyRecord, ...]

    @property
    def warning_records(self) -> tuple[HighFrequencyRecord, ...]:
        return tuple(record for record in self.records if record.assessment.components)


@dataclass(frozen=True)
class HighFrequencyResult:
    """Files created by the high-frequency diagnostic."""

    output_directory: Path
    analysis: HighFrequencyAnalysis
    figure_paths: tuple[Path, ...]


class SingleRunHighFrequency:
    """Adapt the shared high-frequency detector to one published run."""

    def __init__(self, run: MonteCarloRun, profile: EvaluationProfile) -> None:
        if run.dataset is None:
            raise ValueError(f"Run {run.run_id} does not contain a Parquet result.")
        self.run = run
        self.profile = profile

    def analyze(self) -> HighFrequencyAnalysis:
        """Observe 2.5–50 Hz transfer components without assessing stability."""
        catalog = SignalCatalog.from_dataset(self.run.dataset)
        transfers = order_power_transfers_north_to_south(catalog.power_transfers)
        if not transfers:
            raise ValueError("No PQ_* active-power transfer monitors were found.")
        data = self.run.dataset.read_series_group(
            tuple(transfer.column_name for transfer in transfers)
        )
        time_values = tuple(float(value) for value in data["time"].to_list())
        records = []
        for transfer in transfers:
            power_values = tuple(float(value) for value in data[transfer.column_name].to_list())
            assessment = assess_high_frequency_components(
                time_values,
                power_values,
                run_id=self.run.run_id,
                transfer_name=transfer.name,
                criteria=self.profile.high_frequency_criteria,
            )
            records.append(HighFrequencyRecord(transfer, assessment, time_values, power_values))
        return HighFrequencyAnalysis(tuple(records))

    def create(self, output_directory: str | Path) -> HighFrequencyResult:
        """Save one spectrum grid under the damping-analysis output folder."""
        import matplotlib.pyplot as plt

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        analysis = self.analyze()
        figure = self._plot_spectrum(analysis)
        path = output_path / "06_high_frequency_spectrum.png"
        save_figure_with_spanish_copy(figure, path, dpi=150)
        plt.close(figure)
        return HighFrequencyResult(output_path, analysis, (path,))

    def _plot_spectrum(self, analysis: HighFrequencyAnalysis):
        import matplotlib.pyplot as plt
        import numpy as np

        columns = 3
        rows = ceil(len(analysis.records) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(16.5, 3.65 * rows), squeeze=False)
        axes = np.asarray(axes).reshape(-1)
        criteria = self.profile.high_frequency_criteria
        for axis, record in zip(axes, analysis.records):
            samples = tuple(
                (time, value - record.assessment.final_power)
                for time, value in zip(record.time_values, record.power_values)
                if time >= criteria.analysis_start_time_s
            )
            spectrum = spectral_power_spectrum(
                samples,
                criteria.minimum_frequency_hz,
                criteria.maximum_frequency_hz,
            )
            if spectrum is None:
                axis.text(0.5, 0.5, "No observable spectrum", ha="center", va="center", transform=axis.transAxes)
            else:
                frequencies, powers = spectrum
                axis.plot(frequencies, powers, color="#4C78A8", linewidth=0.9)
                close_y_axis(axis)
            axis.set_title(self._display_name(record.transfer.name), fontsize=8.8, fontweight="normal", pad=5)
            axis.set(xlim=(criteria.minimum_frequency_hz, criteria.maximum_frequency_hz), xlabel="Frequency (Hz)", ylabel="Spectral power")
            axis.set_facecolor("white")
            axis.grid(visible=True, color="#D6D9DC", linewidth=0.55, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        for axis in axes[len(analysis.records):]:
            axis.set_visible(False)
        figure.suptitle("High-Frequency Transfer Spectrum — Observation Only", fontsize=14, fontweight="normal", y=0.975)
        figure.text(0.5, 0.015, "Detected components are warnings only and do not alter the dynamic-stability result.", ha="center", va="bottom", fontsize=9, color="#404040")
        figure.subplots_adjust(left=0.07, right=0.96, bottom=0.075, top=0.92, hspace=0.42, wspace=0.24)
        return figure

    @staticmethod
    def _display_name(name: str) -> str:
        return name.removeprefix("NC_").removeprefix("SC_")
