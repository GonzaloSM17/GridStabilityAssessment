"""Convenient entry point for one Monte Carlo study analysis."""

from analyzer import MonteCarloAnalyzer
from assets import Assets
from context import MonteCarloContext

# Explicit inputs for this reproducible study draft.
CONTEXT_PATH = r"Z:\Montecarlo 14-10\20261014_1628_bc4566"
ASSETS_PATH = (
    r"C:\Users\gonzalo.sanchez\OneDrive - Coordinador Eléctrico Nacional"
    r"\Projects Repositories\EmtpApplications\EMTP Assets_SEN2026_v0.8.xlsx"
)
SIGNAL_NAME = "KIMAL220_MTR/u_rms"


def main() -> None:
    """Run the available analyses for the explicit inputs above."""
    context = MonteCarloContext(CONTEXT_PATH)
    assets = Assets(ASSETS_PATH)
    analyzer = MonteCarloAnalyzer(context, assets)
    results = analyzer.analyze(signal_name=SIGNAL_NAME)

    if results.controller_analysis is not None:
        controller_analysis = results.controller_analysis
        print(f"Controller points: {len(controller_analysis.controller_points)}")
        print(f"Weighted run points: {len(controller_analysis.weighted_run_points)}")
        print(f"Skipped controller runs: {len(controller_analysis.skipped_run_ids)}")
        analyzer.plot_controller_population(controller_analysis)
        analyzer.plot_weighted_controllers(controller_analysis)

    if results.signal_analysis is not None:
        signal_analysis = results.signal_analysis
        print(f"Signal: {signal_analysis.signal_name}")
        print(f"Analyzed signal runs: {len(signal_analysis.run_results)}")
        print(
            "Initialization passes: " f"{len(signal_analysis.well_initialized_run_ids)}"
        )
        print(f"Stable runs: {len(signal_analysis.stable_run_ids)}")
        print(f"Skipped signal runs: {len(signal_analysis.skipped_run_ids)}")
        analyzer.plot_initialization(signal_analysis)
        analyzer.plot_stability(signal_analysis)

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
