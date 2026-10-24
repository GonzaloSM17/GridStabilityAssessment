"""Manual entry point for focused analysis of one published simulation run."""

from single_run.workbench import SingleRunWorkbench
from stability.scenario import ContingencyScenario

if __name__ == "__main__":
    # Set to None to select one run_XXXXXX directory through Tk.
    run_path = (
        r"C:\Users\gonzalo.sanchez\Desktop\Results Compiled"
        r"\cache_20261020_1912_8c8eb7\runs\run_000180"
    )
    scenario = ContingencyScenario(
        name="KIMAL CC-2PH-G",
        severity=4,
        fault_type="two_phase_to_ground",
        fault_time_s=1.0,
        clearance_time_s=1.12,
        location="KIMAL_220",
    )
    assets_path = (
        r"C:\Users\gonzalo.sanchez\OneDrive - Coordinador Eléctrico Nacional"
        r"\Projects Repositories\EmtpApplications\EMTP Assets_SEN2026_v0.8.xlsx"
    )

    workbench = SingleRunWorkbench(run_path, scenario, assets_path)
    general_result = workbench.create_general_overview()
    voltage_result = workbench.create_voltage_stability()
    damping_result = workbench.create_damping_stability()
    angular_result = workbench.create_angular_stability()
    results = (general_result, voltage_result, damping_result, angular_result)
    for result in results:
        print(
            f"Created {len(result.figure_paths)} figures in {result.output_directory}"
        )
    report_path = workbench.create_report(
        voltage_result.analysis,
        damping_result.analysis,
        angular_result.analysis,
    )
    print(f"Created report at {report_path}")
