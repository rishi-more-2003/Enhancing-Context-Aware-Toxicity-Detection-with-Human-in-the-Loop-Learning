"""Generate publication-quality figures from saved training histories.

Usage::

    python analyze_results.py
"""

import json
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, FIGURES_DIR
from src.visualization.plots import plot_training_curves, plot_f1_comparison


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    bc_hist = pd.read_csv(RESULTS_DIR / "bc_history.csv")
    dpo_hist = pd.read_csv(RESULTS_DIR / "dpo_history.csv")
    plot_training_curves(bc_hist, dpo_hist)

    eval_path = RESULTS_DIR / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            results = json.load(f)
        plot_f1_comparison(results["bc"]["f1"], results["dpo"]["f1"])

    print("All figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
