"""Manual place for selecting and running one Monte Carlo study analysis."""

from workbench import MonteCarloWorkbench
from scenario import ContingencyScenario

if __name__ == "__main__":
    # Set either path to None to select it with a dialog.
    # context_path = r"Z:\Montecarlo 20-10\20261020_1912_8c8eb7"
    context_path = None
    assets_path = (
        r"C:\Users\gonzalo.sanchez\OneDrive - Coordinador Eléctrico Nacional"
        r"\Projects Repositories\EmtpApplications\EMTP Assets_SEN2026_v0.8.xlsx"
    )
    buses = ("KIMAL_220",)
    scenario = ContingencyScenario(
        name="KIMAL CC-2PH-G",
        severity=4,
        fault_type="two_phase_to_ground",
        fault_time_s=1.0,
        clearance_time_s=1.12,
        location="KIMAL_220",
    )

    workbench = MonteCarloWorkbench(context_path, assets_path, scenario)
    workbench.analyze(buses=buses)
