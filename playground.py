"""Manual place for selecting and running one Monte Carlo study analysis."""

import matplotlib

matplotlib.use("Agg")

from montecarlo.workbench import MonteCarloWorkbench
from stability.scenario import ContingencyScenario

if __name__ == "__main__":
    # Set input/output paths to None to select them with dialogs.
    context_path = None
    assets_path = None
    output_path = None
    # Set to a positive integer to analyze only the first ordered runs.
    # Keep None to analyze the whole Monte Carlo study.
    max_runs = None
    run_characterization = True
    characterization_repeats = 5
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
    workbench.analyze(buses=buses, show_figures=False, max_runs=max_runs)
    spatial_result = workbench.spatial_stability_result
    print(
        f"Spatial stability: {spatial_result.status}; "
        f"{len(spatial_result.figure_paths)} map(s) in "
        f"{spatial_result.output_directory}"
    )

    if (
        run_characterization
        and spatial_result.analysis is not None
        and spatial_result.analysis.points
    ):
        from montecarlo.spatial_stability.characterization import (
            export_characterization,
        )

        characterization_figures = export_characterization(
            spatial_result.analysis,
            spatial_result.output_directory,
            repeats=characterization_repeats,
        )
        print(
            f"Created {len(characterization_figures)} characterization figures in "
            f"{spatial_result.output_directory / 'stability_characterization'}"
        )
    elif run_characterization:
        print("Characterization skipped: no usable spatial controller values.")
