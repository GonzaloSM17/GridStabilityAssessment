"""Convenient entry point for one Monte Carlo study analysis."""

from pathlib import Path

from analyzer import MonteCarloAnalyzer
from assets import Assets
from context import MonteCarloContext

# Explicit inputs for this reproducible study draft.
CONTEXT_PATH = r"F:\Montecarlo 02-10\20261002_1452_f90687"
ASSETS_PATH = (
    r"C:\Users\gonzalo.sanchez\OneDrive - Coordinador Eléctrico Nacional"
    r"\Projects Repositories\EmtpApplications\EMTP Assets_SEN2026_v0.8.xlsx"
)
SIGNAL_NAME = "KIMAL220_MTR/u_rms"


def get_figure_output_directory(context_path: str | Path) -> Path:
    """Return the persistent figure directory within the study."""
    output_directory = Path(context_path) / "analysis_figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def main() -> None:
    """Run the available analyses for the explicit inputs above."""
    context = MonteCarloContext(CONTEXT_PATH)
    assets = Assets(ASSETS_PATH)
    analyzer = MonteCarloAnalyzer(context, assets)
    output_directory = get_figure_output_directory(CONTEXT_PATH)
    results = analyzer.analyze(signal_name=SIGNAL_NAME)
    figures = []

    if results.controller_analysis is not None:
        controller_analysis = results.controller_analysis
        print(f"Controller points: {len(controller_analysis.controller_points)}")
        print(f"Weighted run points: {len(controller_analysis.weighted_run_points)}")
        print(f"Skipped controller runs: {len(controller_analysis.skipped_run_ids)}")
        figures.append(
            ("controller_population.png", analyzer.plot_controller_population(controller_analysis))
        )
        figures.append(
            ("weighted_controllers.png", analyzer.plot_weighted_controllers(controller_analysis))
        )

    if results.signal_analysis is not None:
        signal_analysis = results.signal_analysis
        print(f"Signal: {signal_analysis.signal_name}")
        print(f"Analyzed signal runs: {len(signal_analysis.run_results)}")
        print(
            "Initialization passes: " f"{len(signal_analysis.well_initialized_run_ids)}"
        )
        print(f"Final-settled runs: {len(signal_analysis.stable_run_ids)}")
        idd_compliant_run_count = sum(
            result.stability is not None
            and result.stability.is_dynamic_performance_index_compliant
            for result in signal_analysis.run_results
        )
        print(f"IDD-compliant runs: {idd_compliant_run_count}")
        print(
            "Stability not evaluated (simulation < 6 s): "
            f"{len(signal_analysis.stability_not_evaluated_run_ids)}"
        )
        print(
            "At or above 0.8 pu after the area window: "
            f"{len(signal_analysis.above_voltage_threshold_after_evaluation_run_ids)}"
        )
        print(
            "Below 0.8 pu after the area window: "
            f"{len(signal_analysis.below_voltage_threshold_after_evaluation_run_ids)}"
        )
        print(f"Skipped signal runs: {len(signal_analysis.skipped_run_ids)}")
        for reason in ("fast_oscillation",):
            count = sum(
                reason in result.initialization.rejection_reasons
                for result in signal_analysis.run_results
            )
            print(f"Initialization rejection - {reason}: {count}")
        figures.append(("initialization.png", analyzer.plot_initialization(signal_analysis)))
        if signal_analysis.has_stability_assessments:
            figures.append(
                ("recovery_and_settling.png", analyzer.plot_stability(signal_analysis))
            )
        elif signal_analysis.well_initialized_run_ids:
            print(
                "Post-fault figures not generated: no initialized run reached the "
                "required stability-evaluation duration."
            )
        else:
            print("Post-fault figures not generated: no run passed initialization.")

        if (
            results.controller_analysis is not None
            and signal_analysis.has_stability_assessments
        ):
            figures.append(
                (
                    "fault_on_area_vs_controllers.png",
                    analyzer.plot_fault_on_area_vs_weighted_controllers(
                        results.controller_analysis,
                        signal_analysis,
                    ),
                )
            )
            figures.append(
                (
                    "recovery_area_vs_controllers.png",
                    analyzer.plot_recovery_area_vs_weighted_controllers(
                        results.controller_analysis,
                        signal_analysis,
                    ),
                )
            )
            figures.append(
                (
                    "unified_response_area_vs_controllers.png",
                    analyzer.plot_unified_response_area_vs_weighted_controllers(
                        results.controller_analysis,
                        signal_analysis,
                    ),
                )
            )
            figures.append(
                (
                    "post_fault_persistence_area_vs_controllers.png",
                    analyzer.plot_post_fault_persistence_area_vs_weighted_controllers(
                        results.controller_analysis,
                        signal_analysis,
                    ),
                )
            )

    import matplotlib.pyplot as plt

    for filename, figure in figures:
        figure.savefig(output_directory / filename, dpi=180, bbox_inches="tight")
    print(f"Figures saved to: {output_directory}")
    plt.show()


if __name__ == "__main__":
    main()
