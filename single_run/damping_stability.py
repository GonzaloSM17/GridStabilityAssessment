"""Active-power transfer modal-damping assessment for one simulation run."""

from dataclasses import dataclass
from math import ceil
from pathlib import Path

from data.run import MonteCarloRun
from data.signals import (
    PowerTransfer,
    SignalCatalog,
    order_power_transfers_north_to_south,
)
from stability.plotting import close_y_axis, label_damping_reference_limits
from stability.power_transfer import TransferRunAssessment, assess_transfer_damping, isolate_mode
from stability.scenario import EvaluationProfile


@dataclass(frozen=True)
class TransferDampingRecord:
    """One transfer signal and its shared electromechanical-mode assessment."""

    transfer: PowerTransfer
    assessment: TransferRunAssessment
    time_values: tuple[float, ...]
    power_values: tuple[float, ...]


@dataclass(frozen=True)
class TransferDampingAnalysis:
    """All explicit active-power transfer assessments for a single run."""

    records: tuple[TransferDampingRecord, ...]
    profile: EvaluationProfile


@dataclass(frozen=True)
class DampingStabilityResult:
    """Files created by the focused single-run damping assessment."""

    output_directory: Path
    analysis: TransferDampingAnalysis
    figure_paths: tuple[Path, ...]


class SingleRunDampingStability:
    """Measure and present the 0.1–2.5 Hz transfer damping of one run."""

    def __init__(self, run: MonteCarloRun, profile: EvaluationProfile) -> None:
        if run.dataset is None:
            raise ValueError(f"Run {run.run_id} does not contain a Parquet result.")
        self.run = run
        self.profile = profile

    def analyze(self) -> TransferDampingAnalysis:
        """Apply the common modal-damping procedure to each PQ transfer monitor."""
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
            try:
                assessment = assess_transfer_damping(
                    time_values,
                    power_values,
                    run_id=self.run.run_id,
                    transfer_name=transfer.name,
                    criteria=self.profile.transfer_damping_criteria,
                )
            except ValueError as error:
                raise ValueError(
                    f"Modal damping could not be evaluated for {transfer.name}: {error}"
                ) from error
            records.append(
                TransferDampingRecord(
                    transfer, assessment, time_values, power_values
                ))
        return TransferDampingAnalysis(tuple(records), self.profile)

    def create(self, output_directory: str | Path) -> DampingStabilityResult:
        """Evaluate and save transfer, EM, modal-decomposition, and xi figures."""
        import matplotlib.pyplot as plt

        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        obsolete_path = output_path / "04_em_modal_damping.png"
        if obsolete_path.is_file():
            obsolete_path.unlink()
        analysis = self.analyze()
        figures = (
            ("01_power_transfers.png", self._plot_power_transfers(analysis)),
            ("02_electromechanical_response.png", self._plot_em_response(analysis)),
            ("03_em_mode_decomposition.png", self._plot_mode_decomposition(analysis)),
            (
                "04_dominant_em_mode_damping.png",
                self._plot_modal_damping(
                    analysis,
                    dominant_only=True,
                    title="Dominant Electromechanical-Mode Damping",
                ),
            ),
            (
                "05_all_em_mode_damping.png",
                self._plot_modal_damping(
                    analysis,
                    dominant_only=False,
                    title="All Electromechanical-Mode Damping",
                ),
            ),
        )
        figure_paths = []
        for filename, figure in figures:
            path = output_path / filename
            figure.savefig(path, dpi=150)
            plt.close(figure)
            figure_paths.append(path)
        return DampingStabilityResult(output_path, analysis, tuple(figure_paths))

    def _plot_power_transfers(self, analysis: TransferDampingAnalysis):
        """Show the original active-power signals used by the modal assessment."""
        criteria = self.profile.transfer_damping_criteria
        end_time_s = min(max(record.time_values[-1] for record in analysis.records), 10.0)
        figure, axes = self._new_transfer_grid(len(analysis.records), "Active-Power Transfers")
        for axis, record in zip(axes, analysis.records):
            time, values = self._window(
                record.time_values, record.power_values, criteria.pre_fault_start_time_s, end_time_s
            )
            self._draw_event_context(axis)
            axis.plot(time, values, color="#4C78A8", linewidth=0.9)
            axis.axhline(
                record.assessment.final_power,
                color="#5E6472",
                linewidth=0.8,
                linestyle="--",
            )
            self._format_transfer_axis(
                axis,
                record,
                start_time_s=criteria.pre_fault_start_time_s,
                end_time_s=end_time_s,
                y_label="Active power (MW)",
            )
        self._finalize_transfer_grid(figure, axes, len(analysis.records))
        return figure

    def _plot_em_response(self, analysis: TransferDampingAnalysis):
        """Show the full isolated EM-band response, P(t) − P_final, per transfer."""
        criteria = self.profile.transfer_damping_criteria
        figure, axes = self._new_transfer_grid(
            len(analysis.records), "Electromechanical Transfer Response"
        )
        for axis, record in zip(axes, analysis.records):
            samples = self._deviation_samples(record)
            response = isolate_mode(
                samples,
                criteria.mode_minimum_frequency_hz,
                criteria.mode_maximum_frequency_hz,
            )
            if response is None or record.assessment.status in {"opened", "inactive_pre_fault"}:
                self._not_available(axis, "No evaluable EM response")
            else:
                time, values = zip(*response)
                axis.plot(time, values, color="#4C78A8", linewidth=1.0)
                axis.axhline(0.0, color="#5E6472", linewidth=0.75)
            self._format_transfer_axis(
                axis,
                record,
                start_time_s=criteria.analysis_start_time_s,
                end_time_s=record.time_values[-1],
                y_label="P(t) − P_final (MW)",
            )
        self._finalize_transfer_grid(figure, axes, len(analysis.records))
        return figure

    def _plot_mode_decomposition(self, analysis: TransferDampingAnalysis):
        """Overlay each detected EM mode on the dotted aggregate EM response."""
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        criteria = self.profile.transfer_damping_criteria
        figure, axes = self._new_transfer_grid(
            len(analysis.records), "Electromechanical Mode Decomposition"
        )
        color_map = plt.get_cmap("viridis")
        normalization = Normalize(
            vmin=criteria.mode_minimum_frequency_hz,
            vmax=criteria.mode_maximum_frequency_hz,
        )
        for axis, record in zip(axes, analysis.records):
            samples = self._deviation_samples(record)
            complete_em = isolate_mode(
                samples,
                criteria.mode_minimum_frequency_hz,
                criteria.mode_maximum_frequency_hz,
            )
            if complete_em is None or record.assessment.status in {"opened", "inactive_pre_fault"}:
                self._not_available(axis, "No evaluable EM response")
            else:
                time, values = zip(*complete_em)
                axis.plot(
                    time,
                    values,
                    color="#4A4E69",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.85,
                    zorder=1,
                )
                for mode in record.assessment.modes:
                    response = isolate_mode(samples, mode.lower_band_hz, mode.upper_band_hz)
                    if response is None:
                        continue
                    mode_time, mode_values = zip(*response)
                    axis.plot(
                        mode_time,
                        mode_values,
                        color=color_map(normalization(mode.frequency_hz)),
                        linewidth=1.15 if mode is record.assessment.critical_mode else 0.8,
                        alpha=0.95 if mode is record.assessment.critical_mode else 0.72,
                        zorder=2,
                    )
                axis.axhline(0.0, color="#5E6472", linewidth=0.75)
            self._format_transfer_axis(
                axis,
                record,
                start_time_s=criteria.analysis_start_time_s,
                end_time_s=record.time_values[-1],
                y_label="P(t) − P_final (MW)",
            )
        self._finalize_transfer_grid(figure, axes, len(analysis.records))
        color_bar = figure.colorbar(
            ScalarMappable(norm=normalization, cmap=color_map),
            ax=[axis for axis in axes if axis.get_visible()],
            pad=0.012,
            fraction=0.018,
        )
        color_bar.set_label("EM mode frequency (Hz)", fontsize=9)
        color_bar.ax.tick_params(labelsize=8)
        figure.subplots_adjust(right=0.91)
        return figure

    def _plot_modal_damping(
        self,
        analysis: TransferDampingAnalysis,
        *,
        dominant_only: bool,
        title: str,
    ):
        """Show selected modal xi using the Monte Carlo frequency-coloured cloud."""
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        records = (
            [
                (record, record.assessment.dominant_mode)
                for record in analysis.records
                if record.assessment.dominant_damping_ratio is not None
            ]
            if dominant_only
            else [
                (record, mode)
                for record in analysis.records
                for mode in record.assessment.modes
                if mode.damping_ratio is not None
            ]
        )
        measurable_records = [
            record
            for record in analysis.records
            if (
                record.assessment.dominant_damping_ratio is not None
                if dominant_only
                else any(mode.damping_ratio is not None for mode in record.assessment.modes)
            )
        ]
        # Match the compact Monte Carlo damping summaries: these categorical
        # plots do not need the previous generous vertical spacing.
        figure, axis = plt.subplots(
            figsize=(12, max(4.2, 0.60 * len(measurable_records) + 2.0))
        )
        if not records:
            self._not_available(axis, "No measurable electromechanical modes")
            axis.set_axis_off()
            return figure
        criteria = self.profile.transfer_damping_criteria
        color_map = plt.get_cmap("viridis")
        normalization = Normalize(
            vmin=criteria.mode_minimum_frequency_hz,
            vmax=criteria.mode_maximum_frequency_hz,
        )
        labels = []
        for row, record in enumerate(measurable_records):
            modes = (
                [record.assessment.dominant_mode]
                if dominant_only
                else [
                    mode
                    for mode in record.assessment.modes
                    if mode.damping_ratio is not None
                ]
            )
            offsets = self._vertical_offsets(len(modes))
            axis.scatter(
                [mode.damping_ratio * 100.0 for mode in modes],
                [row + offset for offset in offsets],
                c=[color_map(normalization(mode.frequency_hz)) for mode in modes],
                s=32,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
            labels.append(self._display_transfer_name(record.transfer.name))
        damping_percentages = tuple(mode.damping_ratio * 100.0 for _, mode in records)
        lower_bound = min(-5.0, min(damping_percentages))
        upper_bound = max(5.0, max(damping_percentages))
        padding = max(1.0, (upper_bound - lower_bound) * 0.08)
        axis.set_xlim(lower_bound - padding, upper_bound + padding)
        axis.axvline(0.0, color="#5E6472", linewidth=1.0)
        axis.axvline(5.0, color="#D73027", linestyle="--", linewidth=1.0)
        label_damping_reference_limits(axis)
        axis.set_yticks(range(len(measurable_records)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Damping ratio, ξ (%)", fontsize=10, labelpad=7)
        axis.set_ylabel("500 kV Transmission System", fontsize=10, labelpad=10)
        axis.tick_params(axis="x", labelsize=9)
        axis.tick_params(axis="y", labelsize=9)
        axis.set_title(title, fontsize=13, fontweight="normal", pad=10)
        self._style_axis(axis)
        axis.grid(axis="x", visible=True, color="#D6D9DC", linewidth=0.6, alpha=0.85)
        color_bar = figure.colorbar(
            ScalarMappable(norm=normalization, cmap=color_map), ax=axis, pad=0.018
        )
        color_bar.set_label("EM mode frequency (Hz)", fontsize=9)
        color_bar.ax.tick_params(labelsize=8)
        figure.subplots_adjust(left=0.24, right=0.89, bottom=0.11, top=0.91)
        return figure

    @staticmethod
    def _window(time_values, values, start_time_s: float, end_time_s: float):
        return (
            tuple(time for time in time_values if start_time_s <= time <= end_time_s),
            tuple(value for time, value in zip(time_values, values) if start_time_s <= time <= end_time_s),
        )

    def _deviation_samples(self, record: TransferDampingRecord):
        criteria = self.profile.transfer_damping_criteria
        return tuple(
            (time, value - record.assessment.final_power)
            for time, value in zip(record.time_values, record.power_values)
            if time >= criteria.analysis_start_time_s
        )

    @staticmethod
    def _new_transfer_grid(count: int, title: str):
        import matplotlib.pyplot as plt
        import numpy as np

        columns = 3
        rows = ceil(count / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(16.5, 3.65 * rows), squeeze=False)
        figure.suptitle(title, fontsize=14, fontweight="normal", y=0.975)
        return figure, np.asarray(axes).reshape(-1)

    @staticmethod
    def _finalize_transfer_grid(figure, axes, record_count: int) -> None:
        for axis in axes[record_count:]:
            axis.set_visible(False)
        figure.subplots_adjust(
            left=0.07,
            right=0.96,
            bottom=0.075,
            top=0.92,
            hspace=0.42,
            wspace=0.24,
        )

    def _format_transfer_axis(
        self,
        axis,
        record: TransferDampingRecord,
        *,
        start_time_s: float,
        end_time_s: float,
        y_label: str,
    ) -> None:
        axis.set_title(
            self._display_transfer_name(record.transfer.name),
            fontsize=8.8,
            fontweight="normal",
            pad=5,
        )
        axis.set_xlim(start_time_s, end_time_s)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(y_label)
        self._style_axis(axis)
        close_y_axis(axis)

    def _draw_event_context(self, axis) -> None:
        criteria = self.profile.transfer_damping_criteria
        axis.axvspan(criteria.fault_time_s, criteria.clearance_time_s, color="#F0B323", alpha=0.16, zorder=0)

    @staticmethod
    def _not_available(axis, message: str) -> None:
        axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, color="#5E6472", fontsize=9)

    @staticmethod
    def _vertical_offsets(count: int) -> tuple[float, ...]:
        if count <= 1:
            return (0.0,)
        step = 0.36 / (count - 1)
        return tuple(-0.18 + index * step for index in range(count))

    @staticmethod
    def _display_transfer_name(name: str) -> str:
        return name.removeprefix("NC_").removeprefix("SC_")

    @staticmethod
    def _style_axis(axis) -> None:
        axis.set_facecolor("white")
        axis.grid(visible=True, color="#D6D9DC", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

