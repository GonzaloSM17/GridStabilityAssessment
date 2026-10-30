"""RMS-voltage initialization presentation for one published simulation run."""

from dataclasses import dataclass
from pathlib import Path

from data.run import MonteCarloRun
from data.signals import Bus, SignalCatalog
from stability.initialization import (
    InitializationAssessment,
    InitializationCriteria,
    assess_initialization,
)
from utils.figure_localization import save_figure_with_spanish_copy


@dataclass(frozen=True)
class VoltageInitializationRecord:
    """One 220/500 kV RMS-voltage initialization assessment."""

    bus: Bus
    voltage_level_kv: int
    assessment: InitializationAssessment


@dataclass(frozen=True)
class InitializationAnalysis:
    """Initialization results for every available 220/500 kV RMS-voltage signal."""

    records: tuple[VoltageInitializationRecord, ...]
    criteria: InitializationCriteria


@dataclass(frozen=True)
class InitializationResult:
    """Files created by the single-run initialization presentation."""

    output_directory: Path
    analysis: InitializationAnalysis
    figure_paths: tuple[Path, ...]


class SingleRunInitialization:
    """Apply the common RMS-voltage initialization mechanism to one run."""

    def __init__(self, run: MonteCarloRun, criteria: InitializationCriteria) -> None:
        if run.dataset is None:
            raise ValueError(f"Run {run.run_id} does not contain a Parquet result.")
        self.run = run
        self.criteria = criteria

    def analyze(self) -> InitializationAnalysis:
        """Assess the pre-fault RMS-voltage settling of all 220/500 kV buses."""
        catalog = SignalCatalog.from_dataset(self.run.dataset)
        selected = tuple(
            (bus, level)
            for level in (220, 500)
            for bus in catalog.buses
            if bus.name.endswith(f"_{level}") or bus.name.endswith(str(level))
        )
        if not selected:
            raise ValueError("No 220 kV or 500 kV RMS-voltage signals were found.")
        columns = tuple(bus.voltage_rms().column_name for bus, _ in selected)
        data = self.run.dataset.read_series_group(columns)
        time_values = data["time"].to_list()
        records = []
        for bus, level in selected:
            signal_name = bus.voltage_rms().column_name
            records.append(
                VoltageInitializationRecord(
                    bus=bus,
                    voltage_level_kv=level,
                    assessment=assess_initialization(
                        time_values,
                        data[signal_name].to_list(),
                        self.criteria,
                    ),
                )
            )
        return InitializationAnalysis(tuple(records), self.criteria)

    def create(self, output_directory: str | Path) -> InitializationResult:
        """Create one compact 220/500 kV pre-fault voltage view."""
        import matplotlib.pyplot as plt

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        analysis = self.analyze()
        figure = self._plot(analysis)
        path = output_path / "01_voltage_initialization.png"
        save_figure_with_spanish_copy(figure, path, dpi=150)
        plt.close(figure)
        return InitializationResult(output_path, analysis, (path,))

    def _plot(self, analysis: InitializationAnalysis):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True)
        for axis, level in zip(axes, (220, 500)):
            records = tuple(
                item for item in analysis.records if item.voltage_level_kv == level
            )
            axis.axvspan(
                self.criteria.evaluation_start_time_s,
                self.criteria.evaluation_end_time_s,
                color="#F0B323",
                alpha=0.16,
                zorder=0,
            )
            names = tuple(item.bus.voltage_rms().column_name for item in records)
            if names:
                data = self.run.dataset.read_series_window(
                    names,
                    0.0,
                    self.criteria.evaluation_end_time_s,
                )
                for item in records:
                    color = "#009E73" if item.assessment.is_well_initialized else "#D73027"
                    signal_name = item.bus.voltage_rms().column_name
                    axis.plot(data["time"], data[signal_name], color=color, linewidth=0.75, alpha=0.70)
                axis.relim()
                axis.autoscale(axis="y")
            else:
                axis.text(0.5, 0.5, "No RMS-voltage signals", ha="center", va="center", transform=axis.transAxes)
            axis.set(
                xlim=(0.0, self.criteria.evaluation_end_time_s),
                xlabel="Time (s)",
                ylabel="Voltage (pu)",
                title=f"{level} kV RMS Voltage",
            )
            axis.set_facecolor("white")
            axis.grid(visible=True, color="#D6D9DC", linewidth=0.55, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        figure.legend(
            handles=(
                Line2D((), (), color="#009E73", label="Well initialized"),
                Line2D((), (), color="#D73027", label="Initialization warning"),
                Line2D((), (), color="#F0B323", linewidth=7, alpha=0.35, label="Evaluation window"),
            ),
            loc="lower center",
            ncol=3,
            fontsize=8,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="#AEB4BA",
            facecolor="white",
        )
        figure.suptitle("RMS-Voltage Initialization Assessment", fontsize=13, fontweight="normal", y=0.965)
        figure.subplots_adjust(left=0.085, right=0.95, bottom=0.16, top=0.86, wspace=0.18)
        return figure
