"""Study-level signal analysis built on a Monte Carlo context."""

from dataclasses import dataclass
from pathlib import Path

from assets import Assets
from context import MonteCarloContext
from signal_analysis import (
    InitializationAssessment,
    InitializationCriteria,
    StabilityAssessment,
    StabilityCriteria,
    assess_initialization,
    assess_stability,
)


@dataclass(frozen=True)
class RunSignalAnalysis:
    """Analysis results for one signal in one Monte Carlo run."""

    run_id: int
    initialization: InitializationAssessment
    stability: StabilityAssessment | None


@dataclass(frozen=True)
class SignalStudyAnalysis:
    """Signal analysis results across all readable runs in one study."""

    signal_name: str
    run_results: tuple[RunSignalAnalysis, ...]
    skipped_run_ids: tuple[int, ...]

    @property
    def well_initialized_run_ids(self) -> tuple[int, ...]:
        """Return runs that passed the initialization assessment."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.initialization.is_well_initialized
        )

    @property
    def unsettled_initialization_run_ids(self) -> tuple[int, ...]:
        """Return runs rejected before the stability assessment."""
        return tuple(
            result.run_id
            for result in self.run_results
            if not result.initialization.is_well_initialized
        )

    @property
    def stable_run_ids(self) -> tuple[int, ...]:
        """Return initialized runs that settled by the end of simulation."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None and result.stability.is_stable
        )

    @property
    def unsettled_post_contingency_run_ids(self) -> tuple[int, ...]:
        """Return initialized runs that did not settle by simulation end."""
        return tuple(
            result.run_id
            for result in self.run_results
            if result.stability is not None and not result.stability.is_stable
        )


class MonteCarloAnalyzer:
    """Analyze a selected signal across one Monte Carlo study."""

    def __init__(
        self,
        context: MonteCarloContext,
        assets: Assets | None = None,
    ) -> None:
        self.context = context
        self.assets = assets

    @classmethod
    def from_paths(
        cls,
        context_path: str | Path | None,
        assets_path: str | Path | None = None,
    ) -> "MonteCarloAnalyzer":
        """Create an analyzer from the study and optional Assets workbook paths."""
        context = MonteCarloContext(context_path)
        assets = Assets(assets_path) if assets_path is not None else None
        return cls(context, assets)

    @classmethod
    def select_inputs(cls) -> "MonteCarloAnalyzer | None":
        """Request both input paths before loading either study input."""
        context_path = MonteCarloContext.choose_directory()
        if context_path is None:
            return None

        assets_path = Assets.choose_workbook()
        if assets_path is None:
            return None

        return cls.from_paths(context_path, assets_path)

    def analyze_signal(
        self,
        signal_name: str,
        initialization_criteria: InitializationCriteria | None = None,
        stability_criteria: StabilityCriteria | None = None,
        evaluate_stability: bool = True,
    ) -> SignalStudyAnalysis:
        """Run initialization and post-clearance analysis for one signal.

        Stability is evaluated only when the run passes initialization. Each run's
        Parquet signal is read once for this analysis.
        """
        initialization_criteria = initialization_criteria or InitializationCriteria(
            evaluation_start_time_s=0.9,
            evaluation_end_time_s=1.0,
            reference_start_time_s=0.9,
            reference_end_time_s=1.0,
            relative_error_limit=0.01,
            maximum_out_of_tolerance_fraction=0.01,
            minimum_settled_duration_s=0.05,
        )
        stability_criteria = stability_criteria or StabilityCriteria()

        run_results: list[RunSignalAnalysis] = []
        skipped_run_ids: list[int] = []
        for run in self.context.runs:
            if run.dataset is None:
                skipped_run_ids.append(run.run_id)
                continue

            try:
                data = run.dataset.read_series(signal_name)
                time_values = data["time"].to_list()
                signal_values = data[signal_name].to_list()
                initialization = assess_initialization(
                    time_values,
                    signal_values,
                    initialization_criteria,
                )
                stability = (
                    assess_stability(time_values, signal_values, stability_criteria)
                    if evaluate_stability and initialization.is_well_initialized
                    else None
                )
            except (KeyError, ValueError):
                skipped_run_ids.append(run.run_id)
                continue

            run_results.append(
                RunSignalAnalysis(
                    run_id=run.run_id,
                    initialization=initialization,
                    stability=stability,
                )
            )

        return SignalStudyAnalysis(
            signal_name=signal_name,
            run_results=tuple(run_results),
            skipped_run_ids=tuple(skipped_run_ids),
        )

    def analyze_initialization(
        self,
        signal_name: str,
        criteria: InitializationCriteria | None = None,
    ) -> SignalStudyAnalysis:
        """Run only the initialization assessment for one signal."""
        return self.analyze_signal(
            signal_name,
            initialization_criteria=criteria,
            evaluate_stability=False,
        )

    def plot_initialization(self, analysis: SignalStudyAnalysis):
        """Plot the signal from zero to one second by initialization result."""
        import matplotlib.pyplot as plt
        import polars as pl

        groups = (
            ("Well-initialized runs", set(analysis.well_initialized_run_ids), "tab:blue"),
            (
                "Unsettled initialization runs",
                set(analysis.unsettled_initialization_run_ids),
                "tab:red",
            ),
        )
        figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        for axis, (title, run_ids, color) in zip(axes, groups):
            for run in self.context.runs:
                if run.run_id not in run_ids or run.dataset is None:
                    continue

                data = run.dataset.read_series(analysis.signal_name).filter(
                    (pl.col("time") >= 0.0) & (pl.col("time") < 1.0)
                )
                axis.plot(
                    data["time"],
                    data[analysis.signal_name],
                    color=color,
                    alpha=0.25,
                    linewidth=0.65,
                )
            axis.set_title(f"{title} ({len(run_ids)} runs)")
            axis.set_ylabel(analysis.signal_name)
            axis.grid(visible=True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        figure.suptitle(f"{analysis.signal_name}: initialization assessment")
        figure.tight_layout()
        return figure

    def plot_stability(self, analysis: SignalStudyAnalysis):
        """Plot post-clearance signals by their final-tail stability result."""
        import matplotlib.pyplot as plt
        import polars as pl

        groups = (
            ("Stable post-contingency runs", set(analysis.stable_run_ids), "tab:blue"),
            (
                "Unsettled post-contingency runs",
                set(analysis.unsettled_post_contingency_run_ids),
                "tab:red",
            ),
        )
        figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        for axis, (title, run_ids, color) in zip(axes, groups):
            for run in self.context.runs:
                if run.run_id not in run_ids or run.dataset is None:
                    continue

                data = run.dataset.read_series(analysis.signal_name).filter(
                    pl.col("time") >= 1.12
                )
                axis.plot(
                    data["time"],
                    data[analysis.signal_name],
                    color=color,
                    alpha=0.25,
                    linewidth=0.65,
                )
            axis.set_title(f"{title} ({len(run_ids)} runs)")
            axis.set_ylabel(analysis.signal_name)
            axis.grid(visible=True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        figure.suptitle(f"{analysis.signal_name}: post-contingency assessment")
        figure.tight_layout()
        return figure
