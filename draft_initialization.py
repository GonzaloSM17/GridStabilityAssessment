"""Run the initialization assessment for one selected Monte Carlo signal."""

from analyzer import MonteCarloAnalyzer

STUDY_PATH = r"Z:\Montecarlo 14-10\20261014_1628_bc4566"
ASSETS_PATH = (
    r"C:\Users\gonzalo.sanchez\OneDrive - Coordinador Eléctrico Nacional"
    r"\Projects Repositories\EmtpApplications\EMTP Assets_SEN2026_v0.8.xlsx"
)
SIGNAL_NAME = "KIMAL220_MTR/u_rms"


def main() -> None:
    """Run initialization with the fixed study inputs and show its plot."""
    analyzer = MonteCarloAnalyzer.from_paths(STUDY_PATH, ASSETS_PATH)
    analysis = analyzer.analyze_initialization(SIGNAL_NAME)

    print(f"Signal: {analysis.signal_name}")
    print(f"Analyzed runs: {len(analysis.run_results)}")
    print(f"Well-initialized runs: {len(analysis.well_initialized_run_ids)}")
    print(
        "Unsettled initialization runs: "
        f"{len(analysis.unsettled_initialization_run_ids)}"
    )
    print(f"Skipped runs: {len(analysis.skipped_run_ids)}")

    analyzer.plot_initialization(analysis)

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
