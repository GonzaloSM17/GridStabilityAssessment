"""Manual place for selecting and running one Monte Carlo study analysis."""

import matplotlib

matplotlib.use("Agg")

from montecarlo.workbench import MonteCarloWorkbench
from stability.scenario import ContingencyScenario

if __name__ == "__main__":
    # Set input/output paths to None to select them with dialogs.
    context_path = None
    assets_path = (
        r"C:\Users\kelly.allendes\Downloads"
        r"\Assets_Coordenates\EMTP Assets_v2026.xlsx"
    )
    output_path = None
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
    workbench.select_output_directory(output_path)
    # Save outputs without opening interactive figure windows.
    workbench.analyze(buses=buses, show_figures=False)
    spatial_result = workbench.create_spatial_stability()
    print(
        f"Created {len(spatial_result.figure_paths)} spatial maps in "
        f"{spatial_result.output_directory}"
    )
