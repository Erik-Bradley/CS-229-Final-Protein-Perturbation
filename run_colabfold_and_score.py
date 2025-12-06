#!/usr/bin/env python

"""
ColabFold-based structure prediction and evaluation for ProteinMPNN designs.

Pipeline:
  1. Read per-backbone design FASTAs in mpnn_filtered/*_designs.fa.
  2. Create one FASTA per designed sequence in af2_inputs/.
  3. Run `colabfold_batch` (no templates, 5 AF2 models) on all sequences,
     batching limited via --batch-size 3.
  4. For each sequence, parse ColabFold outputs in af2_outputs/ to collect:
       - mean pLDDT
       - mean PAE
       - ranking confidence
     then align the best model to the idealized backbone and compute:
       - Cα RMSD
       - optional TM-score (if TMALIGN_BIN is set)
       - optional motif-level RMSD for user-defined residue ranges.
  5. Write metrics to af2_metrics_colabfold.csv.
"""

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

import pyrosetta
from pyrosetta import rosetta

import re
from glob import glob

# ---------- CONFIG ----------

# Where your per-backbone design FASTAs live (24 *_designs.fa files)
DESIGN_FASTA_DIR = Path("mpnn_filtered")

# Where we put one FASTA per sequence for ColabFold
AF2_INPUT_DIR = Path("af2_inputs")

# Where ColabFold writes outputs
AF2_OUTPUT_DIR = Path("af2_outputs")

# How many models ColabFold runs (5 = standard AF2 ensemble)
NUM_MODELS = 1

# Batch size for ColabFold (controls GPU memory usage)
COLABFOLD_BATCH_SIZE = 3

# Idealized backbones directory (minimized PDBs)
IDEAL_BACKBONE_DIR = Path("pdb_minimized")

# Optional TM-align binary for TM-score (set to None to skip)
TMALIGN_BIN: Optional[Path] = None
# TMALIGN_BIN = Path("/usr/local/bin/TMalign")

# Optional motif residue ranges (1-based, inclusive; in backbone numbering)
# Fill if you want motif-level RMSD. Example:
# MOTIF_RANGES = {
#     "1ubq_clean": [(1, 20), (35, 45)],
# }
MOTIF_RANGES: Dict[str, List[Tuple[int, int]]] = {}

# Initialize PyRosetta quietly
pyrosetta.init("-mute all")


# ---------- FASTA HELPERS ----------

def read_fasta(path: Path):
    """Yield (header, seq) from a FASTA file."""
    header = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        yield header, "".join(seq_lines)


def make_af2_input_fastas() -> List[Tuple[Path, str, str]]:
    """
    From mpnn_filtered/*_designs.fa, create one FASTA per sequence.

    Returns list of (fasta_path, backbone_name, seq_id), where seq_id will
    be the basename used by ColabFold (e.g., 1ubq_clean_coord_rmsd0.5_seq001).
    """
    AF2_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    design_fastas = sorted(DESIGN_FASTA_DIR.glob("*_designs.fa"))
    if not design_fastas:
        raise FileNotFoundError(f"No *_designs.fa files found in {DESIGN_FASTA_DIR}")

    per_seq_fastas: List[Tuple[Path, str, str]] = []

    for fa in design_fastas:
        backbone_name = fa.stem.replace("_designs", "")
        print(f"[inputs] Processing backbone {backbone_name} from {fa.name}")

        seq_idx = 0
        for header, seq in read_fasta(fa):
            seq_idx += 1
            seq_id = f"{backbone_name}_seq{seq_idx:03d}"
            out_path = AF2_INPUT_DIR / f"{seq_id}.fasta"
            with open(out_path, "w") as out_f:
                out_f.write(f">{seq_id}\n{seq}\n")
            per_seq_fastas.append((out_path, backbone_name, seq_id))

    print(f"[inputs] Wrote {len(per_seq_fastas)} per-sequence FASTAs in {AF2_INPUT_DIR}")
    return per_seq_fastas


# ---------- RUN COLABFOLD ----------

def run_colabfold_batch():
    """
    Call `colabfold_batch` on af2_inputs/ → af2_outputs/.

    We rely on colabfold_batch being on $PATH. Basic command is:

      colabfold_batch --num-models 5 --batch-size 3 --use-templates false \
          af2_inputs/ af2_outputs/

    You can run `colabfold_batch --help` to see all options on your install.
    """
    AF2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colabfold_batch",
        "--num-models", str(NUM_MODELS),
        # "--batch-size", str(COLABFOLD_BATCH_SIZE),
        # "--use-templates", "false",
        "--num-recycle", "1",
        "--rank", "plddt",
        str(AF2_INPUT_DIR),
        str(AF2_OUTPUT_DIR),
    ]

    print("[cf] Running ColabFold with command:")
    print("     " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[cf] ColabFold run finished.")


# ---------- METRIC HELPERS ----------

def load_pose(pdb_path: Path) -> rosetta.core.pose.Pose:
    return pyrosetta.pose_from_pdb(str(pdb_path))


def ca_rmsd(pose1, pose2) -> float:
    return rosetta.core.scoring.CA_rmsd(pose1, pose2)


def tm_score(ref: Path, pred: Path) -> Optional[float]:
    if TMALIGN_BIN is None:
        return None
    cmd = [str(TMALIGN_BIN), str(pred), str(ref)]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("TM-score"):
            parts = line.split()
            for p in parts:
                try:
                    val = float(p)
                    if 0.0 <= val <= 1.0:
                        return val
                except ValueError:
                    continue
    return None


def rmsd_for_ranges(ref_pose, pred_pose, ranges: List[Tuple[int, int]]) -> float:
    ref_coords = []
    pred_coords = []
    for start, end in ranges:
        for i in range(start, end + 1):
            if i > ref_pose.total_residue() or i > pred_pose.total_residue():
                continue
            if not (ref_pose.residue(i).has("CA") and pred_pose.residue(i).has("CA")):
                continue
            ref_xyz = ref_pose.residue(i).xyz("CA")
            pred_xyz = pred_pose.residue(i).xyz("CA")
            ref_coords.append(np.array([ref_xyz.x, ref_xyz.y, ref_xyz.z]))
            pred_coords.append(np.array([pred_xyz.x, pred_xyz.y, pred_xyz.z]))
    if len(ref_coords) == 0:
        return float("nan")
    ref_coords = np.stack(ref_coords)
    pred_coords = np.stack(pred_coords)
    diffs = ref_coords - pred_coords
    return math.sqrt((diffs * diffs).sum() / len(ref_coords))


def find_ideal_backbone(backbone_name: str) -> Path:
    """
    Map a backbone name like '1ubq_clean_coord_rmsd0.5' to its idealized
    backbone in pdb_minimized/.

    Adjust if your naming scheme differs.
    """
    base = backbone_name
    for tag in [
        "_coord_rmsd0.5", "_coord_rmsd1.0", "_coord_rmsd2.0",
        "_dihedral_rmsd0.5", "_dihedral_rmsd1.0", "_dihedral_rmsd2.0",
    ]:
        if base.endswith(tag):
            base = base[: -len(tag)]
            break

    candidate = IDEAL_BACKBONE_DIR / f"{base}_minimized.pdb"
    if candidate.exists():
        return candidate

    base4 = base[:4]
    candidate = IDEAL_BACKBONE_DIR / f"{base4}_clean_minimized.pdb"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not find ideal backbone for {backbone_name}")


# ---------- PARSE COLABFOLD OUTPUTS & SCORE ----------

def sanitize_name_for_colabfold(name: str) -> str:
    """
    Approximate the same sanitization ColabFold uses for filenames:
    replace any non [0-9A-Za-z_.-] character by '_'.
    """
    return re.sub(r"[^0-9A-Za-z_.-]", "_", name)

def parse_single_result(seq_id: str, backbone_name: str) -> Dict[str, object]:
    """
    Parse ColabFold outputs for a sequence with given seq_id.

    Compatible with ColabFold 1.5+ filenames like:
      <seq_id>_scores_rank_001_alphafold2_ptm_model_1_seed_000.json
      <seq_id>_predicted_aligned_error_v1.json
      <seq_id>_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
    """

    # ColabFold sanitizes names: anything not alnum/._- becomes "_"
    def sanitize(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z_.-]", "_", name)

    base_ids = [seq_id]
    san = sanitize(seq_id)
    if san not in base_ids:
        base_ids.append(san)

    def first_match(suffix_patterns) -> Optional[Path]:
        """Try base_ids + each suffix glob; return first existing Path or None."""
        for base in base_ids:
            for suff in suffix_patterns:
                pattern = str(AF2_OUTPUT_DIR / f"{base}{suff}")
                hits = sorted(glob(pattern))
                if hits:
                    return Path(hits[0])
        return None

    # ---------- scores JSON ----------
    scores_json = first_match([
        "_scores_rank_001*.json",   # new ColabFold style
        "_scores*.json",            # fallback
    ])
    if scores_json is None:
        raise FileNotFoundError(f"No scores JSON found for {seq_id} in {AF2_OUTPUT_DIR}")

    with open(scores_json) as f:
        scores = json.load(f)

    # pLDDT (vector) -> mean
    plddt = np.array(scores.get("plddt", []), dtype=float)
    mean_plddt = float(plddt.mean()) if plddt.size else float("nan")

    # ranking confidence (if present)
    ranking_conf = scores.get("ranking_confidence", None)
    ranking_conf = float(ranking_conf) if ranking_conf is not None else float("nan")

    # ---------- PAE ----------
    pae_json = first_match([
        "_predicted_aligned_error*.json",
        "_pae*.json",
    ])
    mean_pae = float("nan")
    if pae_json is not None and pae_json.exists():
        with open(pae_json) as f:
            pae_dict = json.load(f)

        # ColabFold sometimes stores under "pae", sometimes under "predicted_aligned_error"
        if "pae" in pae_dict:
            pae = np.array(pae_dict["pae"], dtype=float)
        elif "predicted_aligned_error" in pae_dict:
            pae = np.array(pae_dict["predicted_aligned_error"], dtype=float)
        else:
            pae = np.array([])

        if pae.size:
            L = pae.shape[0]
            mask = ~np.eye(L, dtype=bool)  # off-diagonal only
            mean_pae = float(pae[mask].mean())

    # ---------- rank-1 PDB ----------
    best_pdb = first_match([
        "_unrelaxed_rank_001*.pdb",
        "_unrelaxed_rank_1*.pdb",
        "_relaxed_rank_001*.pdb",
        "_relaxed_rank_1*.pdb",
    ])
    if best_pdb is None:
        raise FileNotFoundError(f"No rank_1 PDB found for {seq_id} in {AF2_OUTPUT_DIR}")

    # Ideal backbone
    ideal_pdb = find_ideal_backbone(backbone_name)
    ideal_pose = load_pose(ideal_pdb)
    pred_pose = load_pose(best_pdb)

    # Cα RMSD
    ca = float(ca_rmsd(ideal_pose, pred_pose))

    # TM-score: may be None if TM-align not configured → write NaN instead
    tm_val = tm_score(ideal_pdb, best_pdb)
    tm = float(tm_val) if tm_val is not None else float("nan")

    # Motif RMSD if motif ranges defined for this backbone
    motif_ranges = MOTIF_RANGES.get(backbone_name) or MOTIF_RANGES.get(ideal_pdb.stem)
    if motif_ranges:
        motif_r = rmsd_for_ranges(ideal_pose, pred_pose, motif_ranges)
    else:
        motif_r = float("nan")

    return {
        "seq_id": seq_id,
        "backbone": backbone_name,
        "ideal_backbone_pdb": str(ideal_pdb),
        "best_pdb": str(best_pdb),
        "ranking_confidence": ranking_conf,
        "mean_plddt": mean_plddt,
        "mean_pae_offdiag": mean_pae,
        "ca_rmsd_to_ideal": ca,
        "tm_score_to_ideal": tm,
        "motif_ca_rmsd": motif_r,
    }

def collect_all_metrics(per_seq_fastas: List[Tuple[Path, str, str]]):
    rows = []
    for _, backbone_name, seq_id in per_seq_fastas:
        print(f"[metrics] Parsing ColabFold outputs for {seq_id}")
        row = parse_single_result(seq_id, backbone_name)
        rows.append(row)

    out_csv = Path("af2_metrics_colabfold.csv")
    fieldnames = [
        "seq_id",
        "backbone",
        "ideal_backbone_pdb",
        "best_pdb",
        "ranking_confidence",
        "mean_plddt",
        "mean_pae_offdiag",
        "ca_rmsd_to_ideal",
        "tm_score_to_ideal",
        "motif_ca_rmsd",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[metrics] Wrote metrics for {len(rows)} sequences to {out_csv}")


# ---------- MAIN ----------

def main():
    per_seq_fastas = make_af2_input_fastas()
    # run_colabfold_batch()
    collect_all_metrics(per_seq_fastas)


if __name__ == "__main__":
    main()
