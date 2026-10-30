"""Manual entry point for deterministic extreme-case analysis."""

import matplotlib

matplotlib.use("Agg")

from extreme_cases.workbench import ExtremeCasesWorkbench
from stability.scenario import ContingencyScenario

if __name__ == "__main__":
    # Set to None to select the root folder through Tk.
    root_path = r"N:\Gonzalo\Monte Carlo casos extremos"
    output_path = None
    # Set a workbook path, or keep None to choose it through Tk. Assets are
    # required for centre-of-inertia angular stability.
    assets_path = None
    scenario = ContingencyScenario(
        name="CC-2PH-G severity 4",
        severity=4,
        fault_type="two_phase_to_ground",
        fault_time_s=1.0,
        clearance_time_s=1.12,
    )

    workbench = ExtremeCasesWorkbench(root_path, scenario, output_path, assets_path=assets_path)
    if workbench.assets is None:
        workbench.open_assets()
    result = workbench.analyze()
    print(f"Created {len(result.controller_paths)} controller figures")
    print(f"Created {len(result.kimal_paths)} KIMAL figures")
    print(f"Created {len(result.transfer_paths)} transfer figures")
    print(f"Created {len(result.angular_paths)} angular-stability figures")
    print(f"Created report at {result.report_path}")
