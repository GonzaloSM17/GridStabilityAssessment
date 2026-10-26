"""Interactive spatial-map entry point; choose a run folder and an Assets workbook.

For a complete Monte Carlo study, run:
python -m montecarlo.spatial_stability --mode study
"""
import matplotlib

matplotlib.use("Agg")

from montecarlo.spatial_stability.__main__ import main

if __name__ == "__main__":
    main()
