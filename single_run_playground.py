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
    general_result = workbench.create_general_overview()
    initialization_result = workbench.create_initialization()
    voltage_result = workbench.create_voltage_stability()
    damping_result = workbench.create_damping_stability()
    high_frequency_result = workbench.create_high_frequency()
    angular_result = workbench.create_angular_stability()
    spatial_result = workbench.create_spatial_stability(
        voltage=voltage_result.analysis, damping=damping_result.analysis, angular=angular_result.analysis,
    )
    results = (
        general_result,
        initialization_result,
        voltage_result,
        damping_result,
        high_frequency_result,
        angular_result,
        spatial_result,
    )
    for result in results:
        print(
            f"Created {len(result.figure_paths)} figures in {result.output_directory}"
        )
    report_path = workbench.create_report(
        initialization_result.analysis,
        voltage_result.analysis,
        damping_result.analysis,
        high_frequency_result.analysis,
        angular_result.analysis,
    )
    print(f"Created report at {report_path}")
