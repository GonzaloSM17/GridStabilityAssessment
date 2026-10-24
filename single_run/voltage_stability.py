"""Voltage-stability assessment and figures for one published simulation run."""

from dataclasses import dataclass
from pathlib import Path

from data.run import MonteCarloRun
from data.signals import Bus, SignalCatalog
from stability.scenario import EvaluationProfile
from stability.voltage import StabilityAssessment, assess_stability


@dataclass(frozen=True)
class BusVoltageAssessment:
    """One RMS-voltage signal evaluated against the active recovery criteria."""

    bus: Bus
    voltage_level_kv: int
    assessment: StabilityAssessment


@dataclass(frozen=True)
class VoltageStabilityAnalysis:
    """Voltage criteria evaluated for all available 220 kV and 500 kV buses."""

    assessments: tuple[BusVoltageAssessment, ...]
    profile: EvaluationProfile

    def for_voltage_level(self, voltage_level_kv: int) -> tuple[BusVoltageAssessment, ...]:
        return tuple(
            item
            for item in self.assessments
            if item.voltage_level_kv == voltage_level_kv
        )


@dataclass(frozen=True)
class VoltageStabilityResult:
    """Files created by the focused single-run voltage assessment."""

    output_directory: Path
    analysis: VoltageStabilityAnalysis
    figure_paths: tuple[Path, ...]


class SingleRunVoltageStability:
    """Evaluate and present every catalogued 220 kV and 500 kV RMS voltage."""

    def __init__(self, run: MonteCarloRun, profile: EvaluationProfile) -> None:
        if run.dataset is None:
            raise ValueError(f"Run {run.run_id} does not contain a Parquet result.")
        self.run = run
        self.profile = profile

    def analyze(self) -> VoltageStabilityAnalysis:
        """Apply the shared voltage-recovery criteria to every eligible bus."""
        catalog = SignalCatalog.from_dataset(self.run.dataset)
        selected_buses = tuple(
            (bus, level)
            for level in (220, 500)
            for bus in catalog.buses
            if self._belongs_to_voltage_level(bus, level)
        )
        if not selected_buses:
            raise ValueError("No 220 kV or 500 kV RMS-voltage signals were found.")

        columns = tuple(bus.voltage_rms().column_name for bus, _ in selected_buses)
        data = self.run.dataset.read_series_group(columns)
        time_values = data["time"].to_list()
        assessments = []
        for bus, level in selected_buses:
            signal_name = bus.voltage_rms().column_name
            try:
                assessment = assess_stability(
                    time_values,
                    data[signal_name].to_list(),
                    self.profile.voltage_criteria,
                )
            except ValueError as error:
                raise ValueError(
                    f"Voltage stability could not be evaluated for {bus.name}: {error}"
                ) from error
            assessments.append(BusVoltageAssessment(bus, level, assessment))
        return VoltageStabilityAnalysis(tuple(assessments), self.profile)

    def create(self, output_directory: str | Path) -> VoltageStabilityResult:
        """Evaluate the run and save overview plus the two contingency close-ups."""
        import matplotlib.pyplot as plt

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        for filename in (
            "04_220_kv_criteria.png",
            "05_500_kv_voltage.png",
            "06_500_kv_contingency_closeup.png",
            "07_500_kv_fault_closeup.png",
            "08_500_kv_criteria.png",
        ):
            obsolete_path = output_path / filename
            if obsolete_path.is_file():
                obsolete_path.unlink()
        analysis = self.analyze()
        maximum_time_s = self.run.dataset.maximum_time_s()
        figure_paths = []

        for level, prefix in ((220, 1), (500, 4)):
            assessments = analysis.for_voltage_level(level)
            if not assessments:
                continue
            figures = (
                (
                    f"{prefix:02d}_{level}_kv_voltage.png",
                    self._plot_voltage_signals(
                        assessments,
                        start_time_s=0.0,
                        end_time_s=min(maximum_time_s, 3.0),
                        title=f"{level} kV Voltage Stability",
                    ),
                ),
                (
                    f"{prefix + 1:02d}_{level}_kv_contingency_closeup.png",
                    self._plot_voltage_signals(
                        assessments,
                        start_time_s=0.9,
                        end_time_s=min(maximum_time_s, 2.2),
                        title=f"{level} kV Voltage Stability — Contingency Close-up",
                    ),
                ),
                (
                    f"{prefix + 2:02d}_{level}_kv_fault_closeup.png",
                    self._plot_voltage_signals(
                        assessments,
                        start_time_s=0.9,
                        end_time_s=min(maximum_time_s, 1.2),
                        title=f"{level} kV Voltage Stability — Fault and Clearance",
                    ),
                ),
            )
            for filename, figure in figures:
                path = output_path / filename
                figure.savefig(path, dpi=150)
                plt.close(figure)
                figure_paths.append(path)

        return VoltageStabilityResult(output_path, analysis, tuple(figure_paths))

    def _plot_voltage_signals(
        self,
        assessments: tuple[BusVoltageAssessment, ...],
        *,
        start_time_s: float,
        end_time_s: float,
        title: str,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.patches import Patch

        if end_time_s <= start_time_s:
            raise ValueError("The selected voltage plot window is empty.")
        columns = tuple(item.bus.voltage_rms().column_name for item in assessments)
        data = self.run.dataset.read_series_window(columns, start_time_s, end_time_s)
        criteria = self.profile.voltage_criteria
        areas = tuple(item.assessment.unified_response_area_pu_s for item in assessments)
        lower_area = min(areas)
        upper_area = max(areas)
        if lower_area == upper_area:
            upper_area = lower_area + max(0.01, lower_area * 0.1)
        color_map = LinearSegmentedColormap.from_list(
            "response_area", ("#009E73", "#F0B323", "#D73027")
        )
        normalization = Normalize(lower_area, upper_area)

        figure, axis = plt.subplots(figsize=(13.5, 6.8))
        figure.patch.set_facecolor("white")
        self._draw_voltage_context(axis)
        for item in assessments:
            name = item.bus.voltage_rms().column_name
            axis.plot(
                data["time"],
                data[name],
                color=color_map(normalization(item.assessment.unified_response_area_pu_s)),
                linewidth=0.8,
                alpha=0.78,
            )
        axis.set(
            xlim=(start_time_s, end_time_s),
            ylim=(0.0, 1.5),
            xlabel="Time (s)",
            ylabel="Voltage (pu)",
        )
        axis.set_yticks(tuple(index / 10 for index in range(16)))
        self._style_axis(axis)
        axis.set_title(title, fontsize=13, fontweight="normal", pad=8)
        reference_legend = axis.legend(
            handles=(
                Patch(facecolor="#F0B323", alpha=0.16, label="Contingency"),
                *self._voltage_reference_handles(),
            ),
            loc="lower right",
            ncol=1,
            fontsize=8,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="#AEB4BA",
            facecolor="white",
        )
        color_bar = figure.colorbar(
            ScalarMappable(norm=normalization, cmap=color_map),
            ax=axis,
            pad=0.018,
            fraction=0.035,
        )
        color_bar.set_label("Unified voltage-response area (pu·s)", fontsize=9)
        color_bar.ax.tick_params(labelsize=8)
        figure.subplots_adjust(left=0.085, right=0.91, bottom=0.12, top=0.91)
        return figure

    def _plot_criteria_matrix(
        self,
        assessments: tuple[BusVoltageAssessment, ...],
        voltage_level_kv: int,
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        criteria = self.profile.voltage_criteria
        ordered = tuple(sorted(assessments, key=lambda item: item.bus.name))
        areas = tuple(item.assessment.unified_response_area_pu_s for item in ordered)
        lower_area, upper_area = min(areas), max(areas)
        if lower_area == upper_area:
            upper_area = lower_area + max(0.01, lower_area * 0.1)
        area_map = LinearSegmentedColormap.from_list(
            "response_area", ("#009E73", "#F0B323", "#D73027")
        )
        area_normalization = Normalize(lower_area, upper_area)
        values = np.zeros((len(ordered), 4, 4))
        labels = []
        for row, item in enumerate(ordered):
            assessment = item.assessment
            values[row, 0] = self._compliance_color(
                assessment.is_crisis_voltage_compliant
            )
            values[row, 1] = self._compliance_color(
                assessment.is_low_voltage_duration_compliant
            )
            values[row, 2] = self._compliance_color(assessment.is_settled)
            values[row, 3] = area_map(
                area_normalization(assessment.unified_response_area_pu_s)
            )
            labels.append(
                (
                    f"{assessment.crisis_decision_duration_s * 1_000:.0f} ms",
                    f"{assessment.cumulative_low_voltage_duration_s:.3f} s",
                    (
                        f"{assessment.final_settlement_peak_absolute_relative_error * 100:.1f}% / "
                        f"{assessment.final_settlement_peak_to_peak_relative_range * 100:.1f}%"
                    ),
                    f"{assessment.unified_response_area_pu_s:.3f}",
                )
            )

        figure_height = max(6.0, 0.34 * len(ordered) + 2.5)
        figure, axis = plt.subplots(figsize=(12.5, figure_height))
        axis.imshow(values, aspect="auto", interpolation="nearest")
        headers = (
            f"0.70 pu recovery\n(limit: {criteria.post_clearance_grace_period_s * 1_000:.0f} ms)",
            f"Below 0.80 pu\n(limit: {criteria.maximum_cumulative_low_voltage_duration_s:.1f} s)",
            "Final settlement\n(peak / range)",
            "Response area\n(pu·s)",
        )
        for row, row_labels in enumerate(labels):
            for column, label in enumerate(row_labels):
                axis.text(column, row, label, ha="center", va="center", fontsize=7.5)
        axis.set_xticks(range(4), headers)
        axis.xaxis.tick_top()
        axis.tick_params(axis="x", length=0, pad=10, labelsize=9)
        axis.set_yticks(range(len(ordered)), tuple(item.bus.name for item in ordered))
        axis.tick_params(axis="y", length=0, labelsize=8)
        axis.set_title(
            f"{voltage_level_kv} kV Voltage Criteria Assessment",
            fontsize=13,
            fontweight="normal",
            pad=35,
        )
        for row in range(len(ordered) + 1):
            axis.axhline(row - 0.5, color="white", linewidth=1.0)
        for column in range(5):
            axis.axvline(column - 0.5, color="white", linewidth=1.0)
        axis.set_xlim(-0.5, 3.5)
        axis.set_ylim(len(ordered) - 0.5, -0.5)
        figure.text(
            0.5,
            0.02,
            "Green = compliant; red = non-compliant. Response area is a relative severity indicator, not a limit.",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#404040",
        )
        figure.subplots_adjust(left=0.28, right=0.97, bottom=0.06, top=0.91)
        return figure

    def _draw_voltage_context(self, axis) -> None:
        criteria = self.profile.voltage_criteria
        axis.axvspan(
            criteria.fault_time_s,
            criteria.clearance_time_s,
            color="#F0B323",
            alpha=0.16,
            zorder=0,
        )
        axis.axvspan(
            criteria.clearance_time_s,
            criteria.response_end_time_s,
            color="#F0B323",
            alpha=0.07,
            zorder=0,
        )
        for value, color in ((0.70, "#D73027"), (0.80, "#E67E22"), (1.10, "#4C78A8"), (1.20, "#6A3D9A")):
            axis.axhline(value, color=color, linestyle="--", linewidth=0.9, zorder=1)
        axis.axvline(
            criteria.clearance_time_s + criteria.post_clearance_grace_period_s,
            color="#D73027",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )

    @staticmethod
    def _voltage_reference_handles():
        from matplotlib.lines import Line2D

        return (
            Line2D((), (), color="#D73027", linestyle="--", label="0.70 pu"),
            Line2D((), (), color="#E67E22", linestyle="--", label="0.80 pu"),
            Line2D((), (), color="#4C78A8", linestyle="--", label="1.10 pu"),
            Line2D((), (), color="#6A3D9A", linestyle="--", label="1.20 pu"),
            Line2D((), (), color="#D73027", linestyle=":", label="50 ms after clearance"),
        )

    @staticmethod
    def _compliance_color(is_compliant: bool) -> tuple[float, float, float, float]:
        return (0.78, 0.90, 0.80, 1.0) if is_compliant else (0.95, 0.72, 0.70, 1.0)

    @staticmethod
    def _belongs_to_voltage_level(bus: Bus, voltage_level_kv: int) -> bool:
        return bus.name.endswith(f"_{voltage_level_kv}") or bus.name.endswith(
            str(voltage_level_kv)
        )

    @staticmethod
    def _style_axis(axis) -> None:
        axis.set_facecolor("white")
        axis.grid(visible=True, color="#D6D9DC", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

