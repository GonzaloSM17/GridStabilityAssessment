"""Run case clustering from saved spatial results, with folder selectors."""
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from montecarlo.spatial_stability.clustering import main

if __name__ == "__main__":
    main()
