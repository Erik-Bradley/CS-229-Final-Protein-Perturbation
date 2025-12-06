#!/usr/bin/env python

"""
Analysis for Experiment 2: Do AF2 predictions converge toward the ideal backbone
or toward the perturbed backbone used for design?

Input:
    af2_experiment2_convergence.csv
    (written by experiment2_convergence.py)

Key columns:
    seq_id
    backbone
    clean_backbone
    pert_type          (coord / dihedral / ideal)
    pert_rmsd          (0.0, 0.5, 1.0, 2.0, ...)
    rmsd_to_ideal
    rmsd_to_perturbed
    delta_rmsd_pert_minus_ideal
    ratio_rmsd_pert_over_ideal
    mean_plddt
    mean_pae_offdiag
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def find_convergence_csv() -> Path:
    """
    Look for af2_experiment2_convergence.csv in common locations:
      - repo root
      - af2_outputs/
    """
    candidates = [
        Path("af2_experiment2_convergence.csv"),
        Path("af2_outputs") / "af2_experiment2_convergence.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find af2_experiment2_convergence.csv in any of: "
        f"{[str(c) for c in candidates]}"
    )


def main():
    csv_path = find_convergence_csv()
    print(f"[exp2-analysis] Using convergence CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print("[exp2-analysis] Loaded", len(df), "rows")
    print(df.head())

    # Basic sanity check: how many are closer to ideal vs perturbed?
    closer_to_ideal = (df["rmsd_to_ideal"] < df["rmsd_to_perturbed"]).sum()
    closer_to_pert = (df["rmsd_to_perturbed"] < df["rmsd_to_ideal"]).sum()
    ties = (np.isclose(df["rmsd_to_ideal"], df["rmsd_to_perturbed"])).sum()

    print("\n--- Global convergence summary ---")
    print(f"Closer to ideal:     {closer_to_ideal} / {len(df)}")
    print(f"Closer to perturbed: {closer_to_pert} / {len(df)}")
    print(f"Ties (within tolerance): {ties} / {len(df)}")

    # Delta stats (>0 means closer to ideal)
    delta = df["delta_rmsd_pert_minus_ideal"]
    print("\nΔRMSD = RMSD(pred, perturbed) − RMSD(pred, ideal) "
          "(>0 → closer to ideal)")
    print(delta.describe())

    # Wilcoxon signed-rank test comparing pairs per sequence
    try:
        w_stat, w_p = wilcoxon(
            df["rmsd_to_perturbed"],
            df["rmsd_to_ideal"],
            alternative="greater",  # is perturbed RMSD > ideal RMSD on average?
            zero_method="pratt",
        )
        print(f"\nWilcoxon signed-rank test (RMSD_perturbed > RMSD_ideal?): "
              f"stat={w_stat:.3f}, p={w_p:.3e}")
    except Exception as e:
        print("\nWilcoxon test failed:", e)

    # Summaries by perturbation level
    by_level = (
        df.groupby("pert_rmsd")
          .agg(
              n_seq=("seq_id", "count"),
              mean_delta=("delta_rmsd_pert_minus_ideal", "mean"),
              std_delta=("delta_rmsd_pert_minus_ideal", "std"),
              frac_closer_to_ideal=(
                  "delta_rmsd_pert_minus_ideal",
                  lambda s: (s > 0).mean(),
              ),
          )
          .reset_index()
          .sort_values("pert_rmsd")
    )

    print("\n--- By perturbation level (all types) ---")
    print(by_level)

    # Summaries by perturbation level and type
    by_level_type = (
        df.groupby(["pert_type", "pert_rmsd"])
          .agg(
              n_seq=("seq_id", "count"),
              mean_delta=("delta_rmsd_pert_minus_ideal", "mean"),
              std_delta=("delta_rmsd_pert_minus_ideal", "std"),
              frac_closer_to_ideal=(
                  "delta_rmsd_pert_minus_ideal",
                  lambda s: (s > 0).mean(),
              ),
          )
          .reset_index()
          .sort_values(["pert_type", "pert_rmsd"])
    )

    print("\n--- By perturbation level and type ---")
    print(by_level_type)

    # ------------------------------------------------
    # Plots
    # ------------------------------------------------

    # 1. Scatter: RMSD to ideal vs RMSD to perturbed
    plt.figure(figsize=(5, 5))
    plt.scatter(df["rmsd_to_ideal"], df["rmsd_to_perturbed"])
    max_val = max(df["rmsd_to_ideal"].max(), df["rmsd_to_perturbed"].max()) * 1.1
    plt.plot([0, max_val], [0, max_val], "k--", linewidth=1)
    plt.xlabel("RMSD(pred, ideal) [Å]")
    plt.ylabel("RMSD(pred, perturbed) [Å]")
    plt.title("Convergence of AF2 predictions\n(above line → closer to ideal)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Histogram of ΔRMSD
    plt.figure(figsize=(6, 4))
    plt.hist(delta, bins=10, edgecolor="black")
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("ΔRMSD = RMSD(pred, perturbed) − RMSD(pred, ideal) [Å]")
    plt.ylabel("Count")
    plt.title("Distribution of convergence ΔRMSD\n(>0 → closer to ideal)")
    plt.tight_layout()
    plt.show()

    # 3. ΔRMSD vs perturbation level
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        by_level["pert_rmsd"],
        by_level["mean_delta"],
        yerr=by_level["std_delta"],
        fmt="o-",
    )
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Backbone distortion (Å RMSD)")
    plt.ylabel("Mean ΔRMSD (perturbed − ideal) [Å]")
    plt.title("Convergence toward ideal vs perturbed backbone")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------
    # Text snippets for Results
    # ------------------------------------------------

    print("\n--- TEXTUAL SUMMARY SUGGESTIONS ---")

    total = len(df)
    print(
        f"Across all designs, AF2 predictions were closer to the ideal backbone "
        f"in {closer_to_ideal}/{total} cases and closer to the perturbed backbone "
        f"in {closer_to_pert}/{total} cases "
        f"({100 * closer_to_ideal / total:.1f}% vs "
        f"{100 * closer_to_pert / total:.1f}%)."
    )

    if not np.isnan(delta.mean()):
        print(
            f"The mean ΔRMSD (RMSD(pred, perturbed) − RMSD(pred, ideal)) was "
            f"{delta.mean():.3f} ± {delta.std():.3f} Å "
            f"(>0 indicates convergence toward the ideal backbone)."
        )

    # Per-level lines
    for _, row in by_level.iterrows():
        print(
            f"For backbones with distortion {row['pert_rmsd']:.2f} Å, "
            f"AF2 converged closer to the ideal backbone "
            f"in {100 * row['frac_closer_to_ideal']:.1f}% of designs "
            f"(mean ΔRMSD = {row['mean_delta']:.3f} ± {row['std_delta']:.3f} Å, "
            f"n = {int(row['n_seq'])})."
        )

    print(
        "\nYou can describe convergence as 'denoising' when "
        "ΔRMSD > 0 (predictions are closer to the ideal backbone than to the "
        "perturbed input) and as 'noise-preserving' when ΔRMSD < 0."
    )


if __name__ == "__main__":
    main()

