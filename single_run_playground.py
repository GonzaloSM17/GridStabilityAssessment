"""Manual entry point for focused analysis of one published simulation run."""

import matplotlib

matplotlib.use("Agg")

from single_run.workbench import SingleRunWorkbench
from stability.scenario import ContingencyScenario

if __name__ == "__main__":
    # Set to None to select one run_XXXXXX directory through Tk.
    run_path = None
    scenario = ContingencyScenario(
        name="KIMAL CC-2PH-G",
        severity=4,
        fault_type="two_phase_to_ground",
        fault_time_s=1.0,
        clearance_time_s=1.12,
        location="KIMAL_220",
    )
    # Select an Assets workbook with emtp_name, MVA, inertia, posX and posY.
    assets_path = None
    output_path = None

    workbench = SingleRunWorkbench(run_path, scenario, assets_path)
    if workbench.assets is None:
        workbench.open_assets()
    workbench.select_output_directory(output_path)
    result = workbench.analyze()
    for artefact in result.artefact_results:
        print(
            f"Created {len(artefact.figure_paths)} figures in {artefact.output_directory}"
        )
    print(
        f"Spatial stability: {result.spatial.status}; "
        f"{len(result.spatial.figure_paths)} map(s) in "
        f"{result.spatial.output_directory}"
    )
    print(f"Created report at {result.report_path}")
