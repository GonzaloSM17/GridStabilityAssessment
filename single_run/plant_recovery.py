"""P/Q operating-point recovery views for converter-based plants in one run."""

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import re

from data.run import MonteCarloRun
from stability.plant_recovery import (
    PlantRecoveryAssessment,
    PlantRecoveryCriteria,
    assess_plant_recovery,
)
from stability.plotting import close_y_axis
from stability.scenario import ContingencyScenario
from utils.figure_localization import save_figure_with_spanish_copy

PLANT_TYPES = (
    ("PV", "PFV"),
    ("WF", "PE"),
    ("BESS", "BESS"),
)
PLANT_POWER_SIGNAL = re.compile(
    r"^(?P<source>(?P<prefix>PFV|PE|BESS)_.+)/(?:PQpark|Converter)/(?P<quantity>P|Q)$"
)


@dataclass(frozen=True)
class PlantRecoveryRecord:
    """One discovered plant, its P/Q signal pair, and its recovery result."""

    plant_name: str
    technology: str
    active_power_signal: str
    reactive_power_signal: str
    assessment: PlantRecoveryAssessment
    time_values: tuple[float, ...]
    active_power_mw: tuple[float, ...]
    reactive_power_mvar: tuple[float, ...]


@dataclass(frozen=True)
class PlantRecoveryAnalysis:
    """Recovery result for all complete PV, WF, and BESS P/Q signal pairs."""

    records: tuple[PlantRecoveryRecord, ...]
    criteria: PlantRecoveryCriteria

    @property
    def non_recovered_records(self) -> tuple[PlantRecoveryRecord, ...]:
        """Return only plants outside either terminal recovery tolerance."""
        return tuple(
            record for record in self.records if not record.assessment.is_recovered
        )


@dataclass(frozen=True)
class PlantRecoveryResult:
    """Files created for one focused plant-recovery assessment."""

    output_directory: Path
    analysis: PlantRecoveryAnalysis
    figure_paths: tuple[Path, ...]


class SingleRunPlantRecovery:
    """Assess terminal P/Q return of PV, WF, and BESS plants in one run."""

    def __init__(
        self,
        run: MonteCarloRun,
        scenario: ContingencyScenario,
        criteria: PlantRecoveryCriteria | None = None,
    ) -> None:
        if run.dataset is None:
            raise ValueError(f"Run {run.run_id} does not contain a Parquet result.")
        self.run = run
        self.scenario = scenario
        self.criteria = criteria or PlantRecoveryCriteria()

    def analyze(self) -> PlantRecoveryAnalysis:
        """Discover complete plant P/Q pairs and assess their terminal recovery."""
        dataset = self.run.dataset
        dataset.inspect()
        signals = self._discover_plant_power_signals(
            tuple(series.name for series in dataset.series)
        )
        if not signals:
            return PlantRecoveryAnalysis((), self.criteria)

        columns = tuple(
            signal
            for _, quantities in signals
            for signal in (quantities["P"], quantities["Q"])
        )
        data = dataset.read_series_group(columns)
        time_values = tuple(float(value) for value in data["time"].to_list())
        records = []
        for (technology, source_name), quantities in signals:
            active_power_mw = tuple(
                float(value) * 1e-6 for value in data[quantities["P"]].to_list()
            )
            reactive_power_mvar = tuple(
                float(value) * 1e-6 for value in data[quantities["Q"]].to_list()
            )
            plant_name = self._display_plant_name(source_name, technology)
            assessment = assess_plant_recovery(
                time_values,
                active_power_mw,
                reactive_power_mvar,
                plant_name=plant_name,
                technology=technology,
                clearance_time_s=self.scenario.clearance_time_s,
                criteria=self.criteria,
            )
            records.append(
                PlantRecoveryRecord(
                    plant_name=plant_name,
                    technology=technology,
                    active_power_signal=quantities["P"],
                    reactive_power_signal=quantities["Q"],
                    assessment=assessment,
                    time_values=time_values,
                    active_power_mw=active_power_mw,
                    reactive_power_mvar=reactive_power_mvar,
                )
            )
        return PlantRecoveryAnalysis(tuple(records), self.criteria)

    def create(self, output_directory: str | Path) -> PlantRecoveryResult:
        """Create one P/Q figure for PV, WF, and BESS recovery exceptions."""
        import matplotlib.pyplot as plt

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        analysis = self.analyze()
        figures = []
        for index, (technology, _) in enumerate(PLANT_TYPES, start=1):
            figure = self._plot_technology(analysis, technology)
            path = output_path / f"{index:02d}_{technology.lower()}_recovery.png"
            save_figure_with_spanish_copy(figure, path, dpi=150)
            plt.close(figure)
            figures.append(path)
        return PlantRecoveryResult(output_path, analysis, tuple(figures))

    def _plot_technology(self, analysis: PlantRecoveryAnalysis, technology: str):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        records = tuple(
            record
            for record in analysis.non_recovered_records
            if record.technology == technology
        )
        figure, axes = plt.subplots(2, 1, figsize=(13.2, 8.0), sharex=True)
        maximum_time_s = self.run.dataset.maximum_time_s()
        evaluation_start_time_s = (
            self.scenario.clearance_time_s + self.criteria.recovery_evaluation_delay_s
        )
        for axis in axes:
            self._draw_event_windows(axis, evaluation_start_time_s)
            axis.set_xlim(0.0, maximum_time_s)
            axis.set_facecolor("white")
            axis.grid(visible=True, color="#D6D9DC", linewidth=0.55, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        if not records:
            for axis in axes:
                axis.text(
                    0.5,
                    0.5,
                    f"All assessed {technology} plants recover P and Q at tmax.",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="#404040",
                )
                axis.set_ylim(-1.0, 1.0)
            legend_handles = []
        else:
            colors = plt.get_cmap("tab20")
            legend_handles = []
            for index, record in enumerate(records):
                color = colors(index % colors.N)
                for axis, values, reference_value in (
                    (
                        axes[0],
                        record.active_power_mw,
                        record.assessment.pre_fault_active_power_mw,
                    ),
                    (
                        axes[1],
                        record.reactive_power_mvar,
                        record.assessment.pre_fault_reactive_power_mvar,
                    ),
                ):
                    axis.plot(
                        record.time_values,
                        values,
                        color=color,
                        linewidth=1.0,
                        alpha=0.9,
                    )
                    axis.axhline(
                        reference_value,
                        color=color,
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.85,
                    )
                    axis.plot(
                        record.assessment.final_time_s,
                        values[-1],
                        marker="o",
                        markersize=3.4,
                        color=color,
                    )
                legend_handles.append(
                    Line2D((), (), color=color, linewidth=1.3, label=record.plant_name)
                )
            for axis in axes:
                close_y_axis(axis)

        axes[0].set(title="Active Power", ylabel="Active power (MW)")
        axes[1].set(
            title="Reactive Power", xlabel="Time (s)", ylabel="Reactive power (MVAr)"
        )
        figure.suptitle(
            f"{technology} Plant Recovery — Terminal Exceptions",
            fontsize=13,
            fontweight="normal",
            y=0.965,
        )
        reference_handle = Line2D(
            (),
            (),
            color="#404040",
            linestyle="--",
            linewidth=0.9,
            label=(
                "Pre-fault reference "
                f"({self.criteria.pre_fault_reference_time_s:.2f} s)"
            ),
        )
        evaluation_handle = Line2D(
            (),
            (),
            color="#404040",
            linestyle=":",
            linewidth=1.0,
            label=f"Recovery evaluation starts ({evaluation_start_time_s:.2f} s)",
        )
        legend_columns = min(3, max(1, len(legend_handles)))
        legend_rows = (
            ceil(len(legend_handles) / legend_columns) if legend_handles else 0
        )
        footer_y = 0.130 + 0.045 * legend_rows
        figure.legend(
            handles=(reference_handle, evaluation_handle),
            title="References",
            loc="upper center",
            bbox_to_anchor=(0.765, 0.50),
            fontsize=8,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="#AEB4BA",
            facecolor="white",
        )
        if legend_handles:
            figure.legend(
                handles=legend_handles,
                title="Parks",
                loc="lower center",
                bbox_to_anchor=(0.415, 0.020),
                ncol=legend_columns,
                fontsize=8,
                frameon=True,
                fancybox=False,
                framealpha=0.96,
                edgecolor="#AEB4BA",
                facecolor="white",
            )
        figure.text(
            0.415,
            footer_y,
            (
                "Terminal decision at tmax: "
                f"|ΔP| ≤ {self.criteria.active_power_tolerance_mw:g} MW and "
                f"|ΔQ| ≤ {self.criteria.reactive_power_tolerance_mvar:g} MVAr."
            ),
            ha="center",
            va="center",
            fontsize=8.5,
            color="#404040",
        )
        figure.subplots_adjust(
            left=0.09,
            right=0.735,
            bottom=footer_y + 0.060,
            top=0.89,
            hspace=0.32,
        )
        return figure

    def _draw_event_windows(self, axis, evaluation_start_time_s: float) -> None:
        axis.axvspan(
            self.scenario.fault_time_s,
            self.scenario.clearance_time_s,
            color="#F0B323",
            alpha=0.15,
            zorder=0,
        )
        axis.axvspan(
            self.scenario.clearance_time_s,
            evaluation_start_time_s,
            color="#F7E7A8",
            alpha=0.20,
            zorder=0,
        )
        axis.axvline(
            evaluation_start_time_s,
            color="#404040",
            linestyle=":",
            linewidth=1.0,
            alpha=0.9,
        )

    @staticmethod
    def _discover_plant_power_signals(
        available_names: tuple[str, ...],
    ) -> tuple[tuple[tuple[str, str], dict[str, str]], ...]:
        """Return complete P/Q pairs supported by the published signal schema."""
        discovered: dict[tuple[str, str], dict[str, str]] = {}
        technology_by_prefix = {
            prefix: technology for technology, prefix in PLANT_TYPES
        }
        for name in available_names:
            match = PLANT_POWER_SIGNAL.fullmatch(name)
            if match is None:
                continue
            technology = technology_by_prefix[match.group("prefix")]
            key = technology, match.group("source")
            discovered.setdefault(key, {})[match.group("quantity")] = name
        complete_pairs = (
            (key, quantities)
            for key, quantities in discovered.items()
            if {"P", "Q"}.issubset(quantities)
        )
        return tuple(sorted(complete_pairs, key=lambda item: (item[0][0], item[0][1])))

    @staticmethod
    def _display_plant_name(source_name: str, technology: str) -> str:
        """Use the internal device path, not the enclosing subcircuit, in legends."""
        _, separator, internal_name = source_name.partition("/")
        if separator:
            return internal_name.replace("/", "_")
        prefix = next(prefix for label, prefix in PLANT_TYPES if label == technology)
        return source_name.removeprefix(f"{prefix}_")
