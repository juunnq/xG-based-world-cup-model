"""Step 5: 2026 World Cup match-level xG calibration audit.

Run with --refresh to pull the latest daily update of the dataset
(tournament runs through 2026-07-19).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.plots import save
from src.wc2026 import (calibration_in_the_large, fetch_wc2026,
                        overdispersion_test, reliability_bins, stage_table,
                        team_table)

if __name__ == "__main__":
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tm = fetch_wc2026(refresh="--refresh" in sys.argv)

    citl = calibration_in_the_large(tm)
    over = overdispersion_test(tm)
    pd.DataFrame([{**citl, **{f"disp_{k}": v for k, v in over.items()}}]).to_csv(
        config.TABLES_DIR / "wc2026_calibration.csv", index=False)
    print(f"2026 WC: {citl['matches']} completed matches, "
          f"{citl['total_goals']} goals vs {citl['total_xg']:.1f} xG "
          f"(z={citl['z']:.2f}, p={citl['p_value']:.3f}); "
          f"dispersion ratio {over['dispersion_ratio']:.2f} (p={over['p_value']:.3f})")

    teams = team_table(tm)
    teams.to_csv(config.TABLES_DIR / "wc2026_team_performance.csv", index=False)
    print(teams.head(5).round(2).to_string(index=False))
    print(teams.tail(5).round(2).to_string(index=False))

    stage_table(tm).to_csv(config.TABLES_DIR / "wc2026_by_stage.csv", index=False)

    rel = reliability_bins(tm)
    rel.to_csv(config.TABLES_DIR / "wc2026_reliability_bins.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = max(rel["xg_mean"].max(), rel["goals_mean"].max()) * 1.15
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="goals = xG")
    ax.errorbar(rel["xg_mean"], rel["goals_mean"], yerr=rel["ci_half"],
                marker="o", ms=5, lw=1.2, capsize=3, label="team-match bins")
    ax.set_xlabel("Mean team-match xG (bin)")
    ax.set_ylabel("Mean goals scored")
    ax.set_title("2026 World Cup: provider xG vs realized goals\n(team-match level)")
    ax.legend(loc="upper left")
    save(fig, "wc2026_reliability")
    print("Saved 2026 calibration, team, stage tables and reliability figure.")
