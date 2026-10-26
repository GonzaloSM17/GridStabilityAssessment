"""Manual entry point for focused analysis of one published simulation run."""

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
    assets_path = (
        r"C:\Users\gonzalo.sanchez\OneDrive - Coordinador Eléctrico Nacional"
        r"\Projects Repositories\EmtpApplications\EMTP Assets_SEN2026_v0.8.xlsx"
    )

    workbench = SingleRunWorkbench(run_path, scenario, assets_path)
    result = workbench.analyze()
    for artefact in result.artefact_results:
        print(
            f"Created {len(artefact.figure_paths)} figures in {artefact.output_directory}"
        )
    print(f"Created report at {result.report_path}")
