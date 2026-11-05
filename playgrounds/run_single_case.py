"""Manual entry point for focused analysis of one published simulation run."""

import matplotlib
import sys
from pathlib import Path

# REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# if str(REPOSITORY_ROOT) not in sys.path:
#     sys.path.insert(0, str(REPOSITORY_ROOT))

matplotlib.use("Agg")

from single_run.workbench import SingleRunWorkbench
from stability.angular import LEGACY_MIXED_POLE_ANGULAR_NORMALIZATION
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
    # Set this only for the historical study with HP_/HE_/GEO_ four-pole angles.
    angular_signal_normalization = None
    # angular_signal_normalization = LEGACY_MIXED_POLE_ANGULAR_NORMALIZATION

    workbench = SingleRunWorkbench(
        run_path,
        scenario,
        assets_path,
        angular_signal_normalization,
    )
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
