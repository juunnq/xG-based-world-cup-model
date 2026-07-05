"""Step 1: download the four World Cups and build data/shots.parquet."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_dataset

if __name__ == "__main__":
    build_dataset(force="--force" in sys.argv)
