"""Manual place for selecting and running one Monte Carlo study analysis."""

from workbench import MonteCarloWorkbench

if __name__ == "__main__":
    # Set either path to None to select it with a dialog.
    # context_path = r"Z:\Montecarlo 20-10\20261020_1912_8c8eb7"
    context_path = None
    assets_path = (
        r"C:\Users\gonzalo.sanchez\OneDrive - Coordinador Eléctrico Nacional"
        r"\Projects Repositories\EmtpApplications\EMTP Assets_SEN2026_v0.8.xlsx"
    )
    buses = ("KIMAL_220",)

    workbench = MonteCarloWorkbench(context_path, assets_path)
    workbench.analyze(buses=buses)
