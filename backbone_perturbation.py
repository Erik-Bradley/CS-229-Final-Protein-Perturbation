#!/usr/bin/env python

"""
Backbone perturbation for robustness experiments.

Given idealized backbones in pdb_minimized/*_minimized.pdb,
generate perturbed versions at ~0.5, 1.0, and 2.0 Å CA RMSD
using:

  1. Gaussian coordinate noise
  2. Backbone phi/psi dihedral noise

Outputs are written to pdb_perturbed/.
"""

import math
import random
from pathlib import Path

import numpy as np

import pyrosetta
from pyrosetta import rosetta

# ---------- INITIALIZE PYROSETTA (simple, one-liner) ----------

# If this ever complains, you can even reduce to: pyrosetta.init("")
pyrosetta.init(
    "-relax:fast -constrain_relax_to_start_coords -relax:ramp_constraints false"
)

# ---------- CONFIG ----------

MINIMIZED_PDB_DIR = Path("pdb_minimized")
PERTURBED_PDB_DIR = Path("pdb_perturbed")

PERTURB_METHODS = ["coord", "dihedral"]
TARGET_RMSDS = [0.5, 1.0, 2.0]

# Heuristic noise magnitudes (tune if needed)
COORD_SIGMA_MAP = {
    0.5: 0.15,   # Å std dev per x/y/z
    1.0: 0.30,
    2.0: 0.60,
}

TORSION_SIGMA_MAP = {
    0.5: 5.0,    # degrees
    1.0: 10.0,
    2.0: 20.0,
}

# Be much more forgiving about how close we are to the target RMSD.
# We'll just record the actual RMSD later in analysis.
RMSD_TOL_FRACTION = 2.0       # accept if rmsd in [0, 3*target] basically
MAX_ATTEMPTS_PER_LEVEL = 30

# Geometry sanity: CA–CA distances between neighbors
CA_CA_MIN = 2.5   # Å
CA_CA_MAX = 6.0   # Å


# ---------- UTILS ----------

def xyz_to_np(xyz):
    """Convert a Rosetta xyzVector to a NumPy array."""
    return np.array([xyz.x, xyz.y, xyz.z], dtype=float)


def calpha_rmsd(pose1, pose2) -> float:
    """Cα RMSD between two poses."""
    return rosetta.core.scoring.CA_rmsd(pose1, pose2)


def geometry_ok(pose) -> bool:
    """
    Very permissive geometry check: just make sure consecutive CA–CA
    distances are not completely insane. We don't try to enforce
    perfect bond geometry here.
    """
    n = pose.total_residue()
    for i in range(1, n):
        res_i = pose.residue(i)
        res_j = pose.residue(i + 1)
        if not (res_i.has("CA") and res_j.has("CA")):
            continue
        ca_i = xyz_to_np(res_i.xyz("CA"))
        ca_j = xyz_to_np(res_j.xyz("CA"))
        dist = np.linalg.norm(ca_j - ca_i)
        if dist < CA_CA_MIN or dist > CA_CA_MAX:
            return False
    return True


# ---------- PERTURBATION METHODS ----------

def perturb_coordinates(ref_pose, target_rmsd: float):
    """Apply Gaussian coordinate noise to all atoms."""
    sigma = COORD_SIGMA_MAP[target_rmsd]
    pose = rosetta.core.pose.Pose()
    pose.assign(ref_pose)

    for i_res in range(1, pose.total_residue() + 1):
        res = pose.residue(i_res)
        for i_atom in range(1, res.natoms() + 1):
            xyz = res.xyz(i_atom)
            dx = random.gauss(0.0, sigma)
            dy = random.gauss(0.0, sigma)
            dz = random.gauss(0.0, sigma)
            new_xyz = rosetta.numeric.xyzVector_double_t(
                xyz.x + dx, xyz.y + dy, xyz.z + dz
            )
            pose.set_xyz(rosetta.core.id.AtomID(i_atom, i_res), new_xyz)
    return pose


def perturb_dihedrals(ref_pose, target_rmsd: float):
    """Perturb backbone phi/psi with Gaussian noise in degrees."""
    sigma_deg = TORSION_SIGMA_MAP[target_rmsd]
    pose = rosetta.core.pose.Pose()
    pose.assign(ref_pose)

    n = pose.total_residue()
    for i in range(1, n + 1):
        # phi (skip N-term)
        if i > 1:
            pose.set_phi(i, pose.phi(i) + random.gauss(0.0, sigma_deg))
        # psi (skip C-term)
        if i < n:
            pose.set_psi(i, pose.psi(i) + random.gauss(0.0, sigma_deg))

    return pose


def generate_perturbed_pose(ref_pose, method: str, target_rmsd: float):
    """
    Try to generate a perturbed pose with roughly the desired CA RMSD
    and reasonable geometry. If we can't hit the window, save the
    closest we got.
    """
    lower = target_rmsd * (1.0 - RMSD_TOL_FRACTION)
    upper = target_rmsd * (1.0 + RMSD_TOL_FRACTION)

    best_pose = None
    best_diff = float("inf")
    best_rmsd = None

    for attempt in range(1, MAX_ATTEMPTS_PER_LEVEL + 1):
        if method == "coord":
            pert = perturb_coordinates(ref_pose, target_rmsd)
        elif method == "dihedral":
            pert = perturb_dihedrals(ref_pose, target_rmsd)
        else:
            raise ValueError(f"Unknown method: {method}")

        rmsd = calpha_rmsd(ref_pose, pert)
        diff = abs(rmsd - target_rmsd)

        # track best-so-far
        if diff < best_diff and geometry_ok(pert):
            best_diff = diff
            best_pose = pert
            best_rmsd = rmsd

        # if we're within the (now wide) window and geometry is OK, accept early
        if lower <= rmsd <= upper and geometry_ok(pert):
            print(
                f"  [ok] {method}, target {target_rmsd:.1f} Å -> "
                f"{rmsd:.2f} Å (attempt {attempt})"
            )
            return pert

        print(
            f"  [retry] {method}, target {target_rmsd:.1f} Å -> "
            f"{rmsd:.2f} Å (attempt {attempt})"
        )

    # If we get here, we never hit the window; use best_pose if any
    if best_pose is not None:
        print(
            f"  [fallback] {method}, target {target_rmsd:.1f} Å -> "
            f"{best_rmsd:.2f} Å (best of {MAX_ATTEMPTS_PER_LEVEL})"
        )
        return best_pose

    print(
        f"  [warn] No geometry-OK pose for method={method}, "
        f"target={target_rmsd:.1f} Å"
    )
    return None


# ---------- MAIN PIPELINE ----------

def main():
    PERTURBED_PDB_DIR.mkdir(parents=True, exist_ok=True)

    minimized_paths = sorted(MINIMIZED_PDB_DIR.glob("*_minimized.pdb"))
    if not minimized_paths:
        print(f"No minimized PDBs found in {MINIMIZED_PDB_DIR}")
        return

    for pdb_path in minimized_paths:
        pdb_id = pdb_path.stem.replace("_minimized", "")
        print(f"\n=== Processing {pdb_id} ===")

        ref_pose = rosetta.core.import_pose.pose_from_file(str(pdb_path))

        # Save the idealized backbone itself into the perturbed dir once
        ideal_out = PERTURBED_PDB_DIR / f"{pdb_id}_ideal.pdb"
        if not ideal_out.exists():
            ref_pose.dump_pdb(str(ideal_out))
            print(f"  [save] {ideal_out} (idealized)")

        for method in PERTURB_METHODS:
            for target in TARGET_RMSDS:
                pert_pose = generate_perturbed_pose(ref_pose, method, target)
                if pert_pose is None:
                    continue

                out_name = f"{pdb_id}_{method}_rmsd{target:.1f}.pdb"
                out_path = PERTURBED_PDB_DIR / out_name
                pert_pose.dump_pdb(str(out_path))
                print(f"  [save] {out_path}")

    print("\nDone generating perturbed backbones.")


if __name__ == "__main__":
    main()
