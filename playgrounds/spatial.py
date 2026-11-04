"""Interactive spatial-map entry point; choose a run folder and an Assets workbook.

For a complete Monte Carlo study, run:
python -m montecarlo.spatial_stability --mode study
"""
import matplotlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

matplotlib.use("Agg")

from montecarlo.spatial_stability.__main__ import main

if __name__ == "__main__":
    main()
