#!/usr/bin/env python

"""
Experiment 2: Do AF2 predictions converge toward the ideal backbone
or toward the perturbed backbone used for design?

For each designed sequence, we:

  * load the AF2 predicted structure (best_pdb)
  * load the idealized backbone (pdb_minimized/*_minimized.pdb)
  * load the perturbed backbone used for ProteinMPNN
    (pdb_perturbed/<backbone_name>.pdb)

and compute:

  * rmsd_to_ideal      = CA-RMSD(predicted, ideal)
  * rmsd_to_perturbed  = CA-RMSD(predicted, perturbed)
  * delta_rmsd         = rmsd_to_perturbed - rmsd_to_ideal
                          (>0 means closer to ideal)
  * ratio_rmsd         = rmsd_to_perturbed / rmsd_to_ideal

Outputs a CSV: af2_outputs/af2_experiment2_convergence.csv
"""

from pathlib import Path
from typing import Dict, Tuple
import csv
import re

import pandas as pd

# Reuse paths and PyRosetta helpers from your existing script
from run_colabfold_and_score import (
    AF2_OUTPUT_DIR,
    load_pose,
    ca_rmsd,
)


# Where perturbed backbones live (same as in design_with_protein_mpnn.py)
PERTURBED_PDB_DIR = Path("pdb_perturbed")


def parse_backbone_name(backbone: str) -> Tuple[str, str, float]:
    """
    Normalize a backbone name and extract:
      - clean_name (no trailing commas)
      - perturbation type: coord / dihedral / ideal
      - perturbation RMSD: float value, 0.0 for ideal
    """
    # Some names have a trailing comma from FASTA headers
    name = backbone.rstrip(",")

    m_type = re.search(r"(coord|dihedral|ideal)", name)
    pert_type = m_type.group(1) if m_type else "ideal"

    m_rmsd = re.search(r"rmsd([0-9.]+)", name)
    pert_rmsd = float(m_rmsd.group(1)) if m_rmsd else 0.0

    return name, pert_type, pert_rmsd


def find_perturbed_backbone(backbone: str) -> Path:
    """
    Map a backbone name like '1fna_clean_coord_rmsd0.5,' to the
    perturbed PDB file used for design, e.g.:

        pdb_perturbed/1fna_clean_coord_rmsd0.5.pdb
    """
    clean_name, _, _ = parse_backbone_name(backbone)
    path = PERTURBED_PDB_DIR / f"{clean_name}.pdb"
    if not path.exists():
        raise FileNotFoundError(f"Perturbed backbone not found: {path}")
    return path


def main():
    # Candidate search locations for the metrics CSV
    candidates = [
        Path("af2_metrics_colabfold.csv"),
        AF2_OUTPUT_DIR / "af2_metrics_colabfold.csv",
    ]

    metrics_csv = None
    for p in candidates:
        if p.exists():
            metrics_csv = p
            break

    if metrics_csv is None:
        raise FileNotFoundError(
            f"Metrics CSV not found in any of: {[str(c) for c in candidates]}"
        )

    print(f"[exp2] Using metrics CSV: {metrics_csv}")

    # Load metrics CSV
    df = pd.read_csv(metrics_csv)

    # Cache for pose loading
    pose_cache: Dict[Path, object] = {}

    def get_pose(path: Path):
        if path not in pose_cache:
            pose_cache[path] = load_pose(path)
        return pose_cache[path]

    out_rows = []

    for idx, row in df.iterrows():
        seq_id = row["seq_id"]
        backbone_name = row["backbone"]

        ideal_pdb = Path(row["ideal_backbone_pdb"])
        pred_pdb = Path(row["best_pdb"])
        pert_pdb = find_perturbed_backbone(backbone_name)

        clean_name, pert_type, pert_rmsd = parse_backbone_name(backbone_name)

        # Load structures
        ideal_pose = get_pose(ideal_pdb)
        pert_pose = get_pose(pert_pdb)
        pred_pose = get_pose(pred_pdb)

        # RMSDs
        rmsd_to_ideal = float(ca_rmsd(ideal_pose, pred_pose))
        rmsd_to_perturbed = float(ca_rmsd(pert_pose, pred_pose))
        delta_rmsd = rmsd_to_perturbed - rmsd_to_ideal
        ratio_rmsd = (
            rmsd_to_perturbed / rmsd_to_ideal if rmsd_to_ideal > 1e-8 else float("nan")
        )

        out_rows.append({
            "seq_id": seq_id,
            "backbone": backbone_name,
            "clean_backbone": clean_name,
            "pert_type": pert_type,
            "pert_rmsd": pert_rmsd,
            "ideal_backbone_pdb": str(ideal_pdb),
            "perturbed_backbone_pdb": str(pert_pdb),
            "best_pdb": str(pred_pdb),
            "rmsd_to_ideal": rmsd_to_ideal,
            "rmsd_to_perturbed": rmsd_to_perturbed,
            "delta_rmsd_pert_minus_ideal": delta_rmsd,
            "ratio_rmsd_pert_over_ideal": ratio_rmsd,
            # Include AF2 confidence metrics for downstream analysis
            "mean_plddt": float(row["mean_plddt"]),
            "mean_pae_offdiag": float(row["mean_pae_offdiag"]),
        })

        print(
            f"[exp2] {seq_id}: "
            f"ideal={rmsd_to_ideal:.3f} Å, "
            f"perturbed={rmsd_to_perturbed:.3f} Å, "
            f"Δ={delta_rmsd:.3f}"
        )

    # -------------------------
    # Write output to SAME directory as metrics CSV
    # -------------------------

    out_csv = metrics_csv.parent / "af2_experiment2_convergence.csv"

    fieldnames = list(out_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[exp2] Wrote {len(out_rows)} rows → {out_csv}")


if __name__ == "__main__":
    main()

