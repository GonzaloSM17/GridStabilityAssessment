"""Generate the fixed general overview for one published simulation run."""

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, floor, log10
from pathlib import Path
import re

from data.run import MonteCarloRun
from stability.angular import AngularCalculation, angular_excursion_limit
from stability.scenario import ContingencyScenario


SignalSelector = Callable[[str], bool]


@dataclass(frozen=True)
class GeneralPlotDefinition:
    """Describe one subplot and the published signals that belong to it."""

    title: str
    y_label: str
    selector: SignalSelector
    show_legend: bool = False
    show_voltage_references: bool = False
    show_angular_references: bool = False
    use_coi_angular_excursions: bool = False
    value_scale: float = 1.0
    y_limits: tuple[float, float] | None = None


@dataclass(frozen=True)
class GeneralFigureDefinition:
    """Describe one saved overview figure and its ordered subplots."""

    filename: str
    title: str
    plots: tuple[GeneralPlotDefinition, ...]
    columns: int = 2


@dataclass(frozen=True)
class GeneralOverviewResult:
    """Files produced by one general-overview rendering operation."""

    output_directory: Path
    figure_paths: tuple[Path, ...]


def _matches(pattern: str) -> SignalSelector:
    expression = re.compile(pattern)
    return lambda name: expression.search(name) is not None


def _power_selector(prefix: str, quantity: str) -> SignalSelector:
    pattern = rf"^{prefix}_.*/(?:PQpark|Converter)/{quantity}$"
    return _matches(pattern)


SYNCHRONOUS_MACHINE_PREFIX = r"(?:HE|HP|GEO|TER)"


GENERAL_FIGURES = (
    GeneralFigureDefinition(
        filename="01_sm_power.png",
        title="SM Power",
        plots=(
            GeneralPlotDefinition(
                "Active Power", "Active power (MW)",
                _matches(rf"^{SYNCHRONOUS_MACHINE_PREFIX}_[^/]+/P$"),
            ),
            GeneralPlotDefinition(
                "Reactive Power", "Reactive power (MVAr)",
                _matches(rf"^{SYNCHRONOUS_MACHINE_PREFIX}_[^/]+/Q$"),
            ),
        ),
    ),
    GeneralFigureDefinition(
        filename="02_sm_electrical_mechanical_power.png",
        title="SM Electrical and Mechanical Power",
        plots=(
            GeneralPlotDefinition(
                "Electrical Power", "Electrical power (pu)",
                _matches(rf"^{SYNCHRONOUS_MACHINE_PREFIX}_[^/]+/Pe_"),
            ),
            GeneralPlotDefinition(
                "Mechanical Power", "Mechanical power (pu)",
                _matches(rf"^{SYNCHRONOUS_MACHINE_PREFIX}_[^/]+/Control_SG/Pmss$"),
            ),
        ),
    ),
    GeneralFigureDefinition(
        filename="03_sm_terminal_voltage.png",
        title="SM Terminal Voltage",
        plots=(
            GeneralPlotDefinition(
                "Terminal Voltage", "Voltage (pu)",
                _matches(rf"^{SYNCHRONOUS_MACHINE_PREFIX}_[^/]+/Control_SG/Vt$"),
                show_voltage_references=True,
                y_limits=(0.0, 1.5),
            ),
        ),
        columns=1,
    ),
    GeneralFigureDefinition(
        filename="04_sm_dynamics.png",
        title="SM Dynamics",
        plots=(
            GeneralPlotDefinition(
                "Rotor Speed", "Rotor speed (pu)",
                _matches(rf"^{SYNCHRONOUS_MACHINE_PREFIX}_[^/]+/Omega_"),
            ),
            GeneralPlotDefinition(
                "Angular Excursion", "Angular excursion, Δδ (deg)",
                _matches(rf"^{SYNCHRONOUS_MACHINE_PREFIX}_[^/]+/Teta_1_"),
                show_angular_references=True,
                use_coi_angular_excursions=True,
            ),
        ),
    ),
    GeneralFigureDefinition(
        filename="05_500_kv_buses.png",
        title="500 kV Buses",
        plots=(
            GeneralPlotDefinition(
                "RMS Voltage", "Voltage (pu)",
                _matches(r"^(?!PMGD_)[^/]*500_MTR/u_rms$"),
                show_voltage_references=True,
                y_limits=(0.0, 1.5),
            ),
            GeneralPlotDefinition(
                "Frequency", "Frequency (Hz)",
                _matches(r"^(?!PMGD_)[^/]*500_MTR/Frequency_Meter/f$"),
                y_limits=(47.5, 52.5),
            ),
        ),
    ),
    GeneralFigureDefinition(
        filename="06_220_kv_buses.png",
        title="220 kV Buses",
        plots=(
            GeneralPlotDefinition(
                "RMS Voltage", "Voltage (pu)",
                _matches(r"^(?!PMGD_)[^/]*220_MTR/u_rms$"),
                show_voltage_references=True,
                y_limits=(0.0, 1.5),
            ),
            GeneralPlotDefinition(
                "Frequency", "Frequency (Hz)",
                _matches(r"^(?!PMGD_)[^/]*220_MTR/Frequency_Meter/f$"),
                y_limits=(47.5, 52.5),
            ),
        ),
    ),
    GeneralFigureDefinition(
        filename="07_pv_power.png",
        title="PV Power",
        plots=(
            GeneralPlotDefinition("Active Power", "Active power (MW)", _power_selector("PFV", "P"), value_scale=1e-6),
            GeneralPlotDefinition("Reactive Power", "Reactive power (MVAr)", _power_selector("PFV", "Q"), value_scale=1e-6),
        ),
    ),
    GeneralFigureDefinition(
        filename="08_wf_power.png",
        title="WF Power",
        plots=(
            GeneralPlotDefinition("Active Power", "Active power (MW)", _power_selector("PE", "P"), value_scale=1e-6),
            GeneralPlotDefinition("Reactive Power", "Reactive power (MVAr)", _power_selector("PE", "Q"), value_scale=1e-6),
        ),
    ),
    GeneralFigureDefinition(
        filename="09_bess_power.png",
        title="BESS Power",
        plots=(
            GeneralPlotDefinition("Active Power", "Active power (MW)", _power_selector("BESS", "P"), value_scale=1e-6),
            GeneralPlotDefinition("Reactive Power", "Reactive power (MVAr)", _power_selector("BESS", "Q"), value_scale=1e-6),
        ),
    ),
    GeneralFigureDefinition(
        filename="10_der_power.png",
        title="DER Power",
        plots=(
            GeneralPlotDefinition("Active Power", "Active power (MW)", _power_selector("PMGD", "P"), value_scale=1e-6),
            GeneralPlotDefinition("Reactive Power", "Reactive power (MVAr)", _power_selector("PMGD", "Q"), value_scale=1e-6),
        ),
    ),
    GeneralFigureDefinition(
        filename="11_statcom_svc_power.png",
        title="STATCOM / SVC Power",
        plots=(
            GeneralPlotDefinition("Active Power", "Active power (MW)", _power_selector("(?:STATCOM|SVC)", "P"), value_scale=1e-6),
            GeneralPlotDefinition("Reactive Power", "Reactive power (MVAr)", _power_selector("(?:STATCOM|SVC)", "Q"), value_scale=1e-6),
        ),
    ),
    GeneralFigureDefinition(
        filename="12_edac_alto_jahuel.png",
        title="EDAC — Alto Jahuel",
        plots=(
            GeneralPlotDefinition(
                "Load Shedding", "State",
                _matches(r"^AJAHUEL_110_EDAC/LS_"),
                show_legend=True,
            ),
        ),
        columns=1,
    ),
    GeneralFigureDefinition(
        filename="13_500_kv_power_transfers.png",
        title="500 kV Power Transfers",
        plots=(
            GeneralPlotDefinition(
                "North-Center", "Active power (MW)", _matches(r"^PQ_NC_.*/P$"),
                show_legend=True,
            ),
            GeneralPlotDefinition(
                "General", "Active power (MW)", _matches(r"^PQ_(?!NC_|SC_).*/P$"),
                show_legend=True,
            ),
            GeneralPlotDefinition(
                "South-Center", "Active power (MW)", _matches(r"^PQ_SC_.*/P$"),
                show_legend=True,
            ),
        ),
        columns=3,
    ),
)

OBSOLETE_GENERAL_FILENAMES = (
    "01_synchronous_machine_power.png",
    "02_synchronous_machine_mechanical_power.png",
    "03_synchronous_machine_terminal_voltage.png",
    "04_synchronous_machine_dynamics.png",
    "05_bus_500_kv.png",
    "06_bus_220_kv.png",
    "07_solar_pv_power.png",
    "08_wind_power.png",
    "10_pmgd_power.png",
    "11_static_compensator_power.png",
    "12_alto_jahuel_load_shedding.png",
)


class GeneralOverview:
    """Render the fixed, system-wide signal overview for one simulation run."""

    def __init__(
        self,
        run: MonteCarloRun,
        scenario: ContingencyScenario,
        angular_calculation: AngularCalculation | None = None,
    ) -> None:
        if run.dataset is None:
            raise ValueError(f"Run {run.run_id} does not contain a Parquet result.")
        self.run = run
        self.scenario = scenario
        self.angular_calculation = angular_calculation

    def create(self, output_directory: str | Path) -> GeneralOverviewResult:
        """Save every applicable general-overview figure through the run maximum time."""
        import matplotlib.pyplot as plt

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        for filename in OBSOLETE_GENERAL_FILENAMES:
            obsolete_path = output_path / filename
            if obsolete_path.is_file():
                obsolete_path.unlink()
        dataset = self.run.dataset
        dataset.inspect()
        available_names = tuple(series.name for series in dataset.series)
        maximum_time_s = dataset.maximum_time_s()
        figure_paths = []

        for definition in GENERAL_FIGURES:
            matched_plots = tuple(
                (plot, tuple(name for name in available_names if plot.selector(name)))
                for plot in definition.plots
            )
            matched_plots = tuple(
                item
                for item in matched_plots
                if item[1]
                and (
                    not item[0].use_coi_angular_excursions
                    or self.angular_calculation is not None
                )
            )
            if not matched_plots:
                continue

            figure = self._create_figure(definition, matched_plots, maximum_time_s)
            path = output_path / definition.filename
            figure.savefig(path, dpi=150)
            plt.close(figure)
            figure_paths.append(path)

        return GeneralOverviewResult(output_path, tuple(figure_paths))

    def _create_figure(self, definition, matched_plots, maximum_time_s: float):
        import matplotlib.pyplot as plt
        import numpy as np

        columns = min(definition.columns, len(matched_plots))
        rows = ceil(len(matched_plots) / columns)
        has_figure_legend = any(plot.show_legend for plot, _ in matched_plots)
        figure_width = {1: 10.5, 2: 13.6}.get(columns, 16.8)
        figure_height = 5.6 * rows
        with plt.rc_context(
            {
                "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
                "font.size": 10,
                "axes.titleweight": "semibold",
                "axes.labelcolor": "#202020",
                "xtick.color": "#404040",
                "ytick.color": "#404040",
            }
        ):
            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(figure_width, figure_height),
                squeeze=False,
            )
        axes = np.asarray(axes).reshape(-1)
        names = tuple(name for _, plot_names in matched_plots for name in plot_names)
        data = self.run.dataset.read_series_group(names)
        time_values = data["time"]
        legend_handles = []
        legend_labels = []

        for axis, (plot, plot_names) in zip(axes, matched_plots):
            self._draw_event_window(axis)
            if plot.use_coi_angular_excursions:
                for excursion in self.angular_calculation.excursions:
                    axis.plot(
                        self.angular_calculation.time_values,
                        excursion,
                        linewidth=0.7,
                        alpha=0.8,
                        label="_nolegend_",
                    )
                self._draw_angular_references(axis)
            else:
                for name in plot_names:
                    axis.plot(
                        time_values,
                        data[name] * plot.value_scale,
                        linewidth=0.7,
                        alpha=0.8,
                        label=self._curve_label(name) if plot.show_legend else "_nolegend_",
                    )
            if plot.show_voltage_references:
                self._draw_voltage_references(axis)
            axis.set_title(plot.title, fontsize=11, pad=5)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel(plot.y_label)
            axis.set_xlim(0.0, maximum_time_s)
            axis.set_facecolor("white")
            axis.grid(visible=True, color="#D6D9DC", linewidth=0.55, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if plot.use_coi_angular_excursions:
                limit = angular_excursion_limit(self.angular_calculation.excursions)
                axis.set_ylim(-limit, limit)
            elif plot.y_limits is not None:
                axis.set_ylim(*plot.y_limits)
            else:
                axis.set_ylim(*self._nice_y_limits(*axis.get_ylim()))
            self._set_readable_y_ticks(axis, plot.show_voltage_references)
            if plot.show_voltage_references:
                axis.legend(
                    loc="lower right",
                    fontsize=8,
                    frameon=True,
                    fancybox=False,
                    framealpha=0.94,
                    edgecolor="#AEB4BA",
                    facecolor="white",
                )
            if plot.show_angular_references:
                axis.legend(
                    loc="lower right",
                    fontsize=8,
                    frameon=True,
                    fancybox=False,
                    framealpha=0.94,
                    edgecolor="#AEB4BA",
                    facecolor="white",
                )
            if plot.show_legend:
                handles, labels = axis.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label not in legend_labels:
                        legend_handles.append(handle)
                        legend_labels.append(label)

        for axis in axes[len(matched_plots):]:
            axis.set_visible(False)
        figure.suptitle(
            definition.title,
            fontsize=13,
            fontweight="normal",
            y=0.955,
        )
        if has_figure_legend and legend_handles:
            legend_columns = min(5, len(legend_labels))
            legend_rows = ceil(len(legend_labels) / legend_columns)
            figure.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.018),
                ncol=legend_columns,
                fontsize=8,
                frameon=True,
                fancybox=False,
                framealpha=0.96,
                edgecolor="#AEB4BA",
                facecolor="white",
                columnspacing=1.3,
                handlelength=1.8,
            )
            legend_margin = 0.075 + 0.045 * legend_rows
        else:
            legend_margin = 0.055
        figure.subplots_adjust(
            left=0.085,
            right=0.95,
            bottom=max(legend_margin, 0.085),
            top=0.85,
            wspace=0.18,
            hspace=0.36,
        )
        return figure

    def _draw_event_window(self, axis) -> None:
        axis.axvspan(
            self.scenario.fault_time_s,
            self.scenario.clearance_time_s,
            color="#F0B323",
            alpha=0.14,
            zorder=0,
        )

    @staticmethod
    def _draw_voltage_references(axis) -> None:
        for value, color, label in (
            (0.70, "#D73027", "0.70 pu"),
            (0.80, "#E67E22", "0.80 pu"),
            (1.10, "#4C78A8", "1.10 pu"),
            (1.20, "#6A3D9A", "1.20 pu"),
        ):
            axis.axhline(
                value,
                color=color,
                linestyle="--",
                linewidth=0.9,
                alpha=0.9,
                label=label,
            )

    @staticmethod
    def _draw_angular_references(axis) -> None:
        for value, label in ((120.0, "+120°"), (-120.0, "−120°")):
            axis.axhline(
                value,
                color="#D73027",
                linestyle="--",
                linewidth=0.9,
                alpha=0.9,
                label=label,
            )

    @staticmethod
    def _curve_label(name: str) -> str:
        """Return the asset or transfer name without its measured quantity suffix."""
        source_name = name.split("/", 1)[0]
        return source_name.removeprefix("PQ_NC_").removeprefix("PQ_SC_").removeprefix("PQ_")

    @staticmethod
    def _nice_y_limits(lower: float, upper: float) -> tuple[float, float]:
        """Close automatic limits at a readable 1-2-5 scale without clipping data."""
        if lower == upper:
            padding = abs(lower) * 0.1 if lower else 1.0
            lower -= padding
            upper += padding
        target_step = (upper - lower) / 6.0
        magnitude = 10.0 ** floor(log10(target_step))
        candidates = tuple(value * magnitude for value in (1.0, 2.0, 5.0, 10.0))
        step = min(candidates, key=lambda value: abs(value - target_step))
        return floor(lower / step) * step, ceil(upper / step) * step

    @staticmethod
    def _set_readable_y_ticks(axis, voltage_references: bool) -> None:
        """Use a rounded scale that explicitly labels both vertical limits."""
        import numpy as np
        from matplotlib.ticker import ScalarFormatter

        lower, upper = axis.get_ylim()
        span = upper - lower
        if voltage_references:
            # The 0.70, 0.80, 1.10 and 1.20 pu reference values must be readable.
            step = 0.1
        else:
            target_step = span / 6.0
            magnitude = 10.0 ** floor(log10(target_step))
            candidates = tuple(value * magnitude for value in (1.0, 2.0, 5.0, 10.0))
            step = min(candidates, key=lambda value: abs(value - target_step))

        tick_count = round(span / step)
        ticks = lower + np.arange(tick_count + 1) * step
        ticks[-1] = upper
        axis.set_yticks(ticks)
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.yaxis.set_major_formatter(formatter)
