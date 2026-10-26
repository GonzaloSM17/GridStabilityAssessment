"""Run spatial maps with the same file-selection flow as the workbenches."""
import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
from stability.scenario import ContingencyScenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "study"), default="single")
    parser.add_argument("--run", type=Path, help="Single published run; omitted opens folder picker")
    parser.add_argument("--study", type=Path, help="Study with manifest.json and runs/; omitted opens picker")
    parser.add_argument("--assets", type=Path, help="Assets workbook; omitted opens Excel picker")
    parser.add_argument("--output", type=Path, help="Spatial output directory; omitted opens folder picker")
    parser.add_argument("--fault-time", type=float, default=1.0)
    parser.add_argument("--clearance-time", type=float, default=1.12)
    args = parser.parse_args()
    if args.run and args.study: parser.error("Choose a single --run or a --study, not both")
    if args.mode == "study" and args.run: parser.error("--run requires --mode single")
    scenario = ContingencyScenario(name="Spatial assessment", severity=4, fault_type="two_phase_to_ground",
                                  fault_time_s=args.fault_time, clearance_time_s=args.clearance_time)
    if args.mode == "study" or args.study:
        from montecarlo.workbench import MonteCarloWorkbench
        workbench = MonteCarloWorkbench(args.study, args.assets, scenario)
        output = workbench.select_output_directory(args.output)
        result = workbench.create_spatial_stability(output_directory=output)
    else:
        from single_run.workbench import SingleRunWorkbench
        workbench = SingleRunWorkbench(args.run, scenario, args.assets)
        if workbench.assets is None:
            workbench.open_assets()
        output = workbench.select_output_directory(args.output)
        result = workbench.create_spatial_stability(output_directory=output)
    print(f"Created {len(result.figure_paths)} PNG maps in {result.output_directory}")


if __name__ == "__main__": main()
