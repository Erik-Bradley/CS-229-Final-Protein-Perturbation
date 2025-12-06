"""
Analysis script for CS229 protein design project:
Quantifying how backbone distortion affects design success using AF2 metrics.

Assumes input file: af2_metrics_colabfold.csv
Columns: seq_id, backbone, ideal_backbone_pdb, best_pdb, ranking_confidence,
         mean_plddt, mean_pae_offdiag, ca_rmsd_to_ideal, tm_score_to_ideal,
         motif_ca_rmsd
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu
import statsmodels.formula.api as smf

# -----------------------------
# 0. Load data
# -----------------------------

df = pd.read_csv("af2_metrics_colabfold.csv")

print("Loaded df with shape:", df.shape)
print(df.head())


# -----------------------------
# 1. Extract perturbation type and level from backbone name
# -----------------------------

# Example backbone names (adapt as needed):
#   1fna_clean_coord_rmsd0.5,
#   1fna_clean_coord_rmsd1.0,
#   1fna_clean_dihedral_rmsd0.5,
#   1fna_clean_ideal

# Perturbation type: coord / dihedral / ideal
df["pert_type"] = df["backbone"].str.extract(r"(coord|dihedral|ideal)", expand=False)
df["pert_type"] = df["pert_type"].fillna("ideal")

# Perturbation magnitude (RMSD value); ideal gets 0
df["pert_rmsd"] = df["backbone"].str.extract(r"rmsd([0-9.]+)", expand=False)
df["pert_rmsd"] = df["pert_rmsd"].astype(float)
df["pert_rmsd"] = df["pert_rmsd"].fillna(0.0)

print("\nUnique perturbation types:", df["pert_type"].unique())
print("Unique perturbation levels:", sorted(df["pert_rmsd"].unique()))


# -----------------------------
# 2. Define design success
# -----------------------------

# As in your Methods:
#   success if CαRMSD ≤ 2.0 Å AND mean pLDDT ≥ 70
SUCCESS_RMSD_THRESH = 2.0
SUCCESS_PLDDT_THRESH = 70.0

df["success"] = (
    (df["ca_rmsd_to_ideal"] <= SUCCESS_RMSD_THRESH) &
    (df["mean_plddt"] >= SUCCESS_PLDDT_THRESH)
)

print("\nOverall success rate: {:.1f}%".format(100 * df["success"].mean()))


# -----------------------------
# 3. Backbone-level and perturbation-level summaries
# -----------------------------

# Per-backbone summary (useful if multiple sequences per backbone)
by_backbone = (
    df.groupby("backbone")
      .agg(
          n_seq=("seq_id", "count"),
          mean_plddt=("mean_plddt", "mean"),
          std_plddt=("mean_plddt", "std"),
          mean_rmsd=("ca_rmsd_to_ideal", "mean"),
          std_rmsd=("ca_rmsd_to_ideal", "std"),
          mean_pae=("mean_pae_offdiag", "mean"),
          std_pae=("mean_pae_offdiag", "std"),
          success_rate=("success", "mean"),
      )
      .reset_index()
)

print("\nPer-backbone summary:")
print(by_backbone)

# Per perturbation level (what you care about for “backbone distortion”)
by_level = (
    df.groupby("pert_rmsd")
      .agg(
          n_seq=("seq_id", "count"),
          success_rate=("success", "mean"),
          mean_plddt=("mean_plddt", "mean"),
          std_plddt=("mean_plddt", "std"),
          mean_rmsd=("ca_rmsd_to_ideal", "mean"),
          std_rmsd=("ca_rmsd_to_ideal", "std"),
          mean_pae=("mean_pae_offdiag", "mean"),
          std_pae=("mean_pae_offdiag", "std"),
      )
      .reset_index()
      .sort_values("pert_rmsd")
)

print("\nPer perturbation level summary:")
print(by_level)


# -----------------------------
# 4. Plots: success vs distortion, RMSD vs distortion, pLDDT vs distortion
# -----------------------------

plt.figure(figsize=(6, 4))
plt.plot(by_level["pert_rmsd"], 100 * by_level["success_rate"], marker="o")
plt.xlabel("Backbone distortion (Å RMSD)")
plt.ylabel("Success rate (%)")
plt.title("Design success vs backbone distortion")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.errorbar(
    by_level["pert_rmsd"],
    by_level["mean_rmsd"],
    yerr=by_level["std_rmsd"],
    marker="o",
    linestyle="-",
)
plt.xlabel("Backbone distortion (Å RMSD)")
plt.ylabel("Cα RMSD to ideal (Å)")
plt.title("Fold error vs backbone distortion")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.errorbar(
    by_level["pert_rmsd"],
    by_level["mean_plddt"],
    yerr=by_level["std_plddt"],
    marker="o",
    linestyle="-",
)
plt.xlabel("Backbone distortion (Å RMSD)")
plt.ylabel("Mean pLDDT")
plt.title("AF2 confidence vs backbone distortion")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.errorbar(
    by_level["pert_rmsd"],
    by_level["mean_pae"],
    yerr=by_level["std_pae"],
    marker="o",
    linestyle="-",
)
plt.xlabel("Backbone distortion (Å RMSD)")
plt.ylabel("Mean PAE (off-diagonal)")
plt.title("Predicted alignment error vs backbone distortion")
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# 5. Compare distributions across distortion levels (KS & Mann–Whitney)
# -----------------------------

print("\nPairwise KS/Mann–Whitney tests between distortion levels (Cα RMSD):")
levels = sorted(df["pert_rmsd"].unique())
for i in range(len(levels)):
    for j in range(i + 1, len(levels)):
        a = levels[i]
        b = levels[j]
        x = df.loc[df["pert_rmsd"] == a, "ca_rmsd_to_ideal"].dropna()
        y = df.loc[df["pert_rmsd"] == b, "ca_rmsd_to_ideal"].dropna()
        if len(x) > 0 and len(y) > 0:
            ks = ks_2samp(x, y)
            mw = mannwhitneyu(x, y, alternative="two-sided")
            print(f"  {a} Å vs {b} Å: KS p={ks.pvalue:.3e}, MW p={mw.pvalue:.3e}")
        else:
            print(f"  {a} Å vs {b} Å: insufficient data for test")


print("\nPairwise KS/Mann–Whitney tests between distortion levels (mean PAE):")
for i in range(len(levels)):
    for j in range(i + 1, len(levels)):
        a = levels[i]
        b = levels[j]
        x = df.loc[df["pert_rmsd"] == a, "mean_pae_offdiag"].dropna()
        y = df.loc[df["pert_rmsd"] == b, "mean_pae_offdiag"].dropna()
        if len(x) > 0 and len(y) > 0:
            ks = ks_2samp(x, y)
            mw = mannwhitneyu(x, y, alternative="two-sided")
            print(f"  {a} Å vs {b} Å: KS p={ks.pvalue:.3e}, MW p={mw.pvalue:.3e}")
        else:
            print(f"  {a} Å vs {b} Å: insufficient data for test")


# -----------------------------
# 6. Logistic regression: success ~ backbone distortion
# -----------------------------

# Simple logistic regression using statsmodels
# success ~ pert_rmsd (+ optionally perturbation type)
try:
    logit_model = smf.logit("success ~ pert_rmsd", data=df).fit(disp=False)
    print("\nLogistic regression: success ~ pert_rmsd")
    print(logit_model.summary())
    print("Odds ratio per 1 Å RMSD:", np.exp(logit_model.params["pert_rmsd"]))
except Exception as e:
    print("\nLogistic regression failed:", e)

# If you want to include perturbation type as a factor:
try:
    logit_model_type = smf.logit("success ~ pert_rmsd + C(pert_type)", data=df).fit(disp=False)
    print("\nLogistic regression: success ~ pert_rmsd + C(pert_type)")
    print(logit_model_type.summary())
except Exception as e:
    print("\nLogistic regression with type failed:", e)


# -----------------------------
# 7. Quick textual summaries to paste into Results
# -----------------------------

print("\n--- TEXTUAL SUMMARY SUGGESTIONS ---")

# (a) Success rates by distortion level
for _, row in by_level.iterrows():
    print(
        f"Distortion {row['pert_rmsd']:.2f} Å: "
        f"success rate = {100*row['success_rate']:.1f}% "
        f"(n = {int(row['n_seq'])})"
    )

# (b) Correlation between distortion and RMSD/pLDDT/PAE
corr_rmsd = df["pert_rmsd"].corr(df["ca_rmsd_to_ideal"])
corr_plddt = df["pert_rmsd"].corr(df["mean_plddt"])
corr_pae = df["pert_rmsd"].corr(df["mean_pae_offdiag"])

print(
    "\nCorrelation between backbone distortion and Cα RMSD: {:.3f}".format(corr_rmsd)
)
print(
    "Correlation between backbone distortion and mean pLDDT: {:.3f}".format(corr_plddt)
)
print(
    "Correlation between backbone distortion and mean PAE: {:.3f}".format(corr_pae)
)

