#!/usr/bin/env python

"""
Backbone Selection and Preprocessing for Protein Design Robustness Project

Steps:
1. Select a small, structurally diverse set of PDB IDs (manually curated here).
2. Download PDB files from the RCSB.
3. Clean structures: remove ligands, water, and non-standard residues.
4. Run a short, restrained relaxation in PyRosetta to remove clashes
   without large-scale backbone deformation.
5. Save the minimized "idealized" backbones for later perturbation.
"""

import os
import urllib.request
from pathlib import Path

from Bio.PDB import PDBParser, PDBIO, Select

# ---------- CONFIG ----------

# Output directories
RAW_PDB_DIR = Path("pdb_raw")
CLEAN_PDB_DIR = Path("pdb_clean")
MINIMIZED_PDB_DIR = Path("pdb_minimized")

for d in [RAW_PDB_DIR, CLEAN_PDB_DIR, MINIMIZED_PDB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hard-coded list of diverse, small(ish) single-chain proteins
# (You can modify this list to better match your needs)
PDB_IDS = [
    "1L2Y",  # alpha-helical bundle
    "1UBQ",  # ubiquitin-like
    "1ENH",  # small helix bundle
    "1TEN",  # beta-sandwich
    "1CSP",  # beta-sheet protein
    "1R69",  # alpha/beta
    "1MJC",  # small repeat-like
    "1RIS",  # alpha/beta
    "1PGX",  # alpha/beta enzyme-like
    "1FNA",  # fibronectin type III (beta-sandwich)
    "1AIL",  # helix-turn-helix
    "1YCC",  # small heme protein, mostly alpha
]

MIN_LEN = 80
MAX_LEN = 160

# Standard 20 amino acids (3-letter codes) for filtering residues
STANDARD_AA3 = {
    "ALA", "CYS", "ASP", "GLU", "PHE",
    "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG",
    "SER", "THR", "VAL", "TRP", "TYR"
}


# ---------- STEP 1: DOWNLOAD PDBs ----------

def download_pdb(pdb_id: str, out_dir: Path = RAW_PDB_DIR) -> Path:
    """
    Download a PDB file from RCSB for a given PDB ID.
    """
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    out_path = out_dir / f"{pdb_id}.pdb"

    if out_path.exists():
        print(f"[download] {pdb_id} already exists, skipping")
        return out_path

    print(f"[download] Downloading {pdb_id} from {url}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


# ---------- STEP 2: CLEAN STRUCTURES ----------

class CleanProteinSelect(Select):
    """
    Biopython Select subclass to:
    - keep only standard amino acids
    - drop HETATM (ligands, ions, etc.)
    - drop waters
    """

    def accept_residue(self, residue):
        # residue.id[0] is ' ' for standard residues, 'H_' for HETATM, etc.
        hetfield = residue.id[0]
        resname = residue.get_resname().upper()

        # Keep only standard residues from main protein chain
        if hetfield.strip() != "":
            return 0  # discard HETATM, waters, etc.
        if resname not in STANDARD_AA3:
            return 0  # discard non-standard residues
        return 1


def clean_pdb(raw_path: Path, out_dir: Path = CLEAN_PDB_DIR) -> Path:
    """
    Load a raw PDB with Biopython and write out a cleaned version.
    - Removes ligands, waters, and non-standard residues.
    - Keeps all chains (you can customize if you want only one).
    """
    pdb_id = raw_path.stem
    out_path = out_dir / f"{pdb_id}_clean.pdb"

    if out_path.exists():
        print(f"[clean] {pdb_id} cleaned PDB already exists, skipping")
        return out_path

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(raw_path))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), select=CleanProteinSelect())
    print(f"[clean] Saved cleaned PDB to {out_path}")

    return out_path


def get_sequence_length(clean_path: Path) -> int:
    """
    Estimate sequence length as number of standard residues in the first model.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(clean_path.stem, str(clean_path))
    model = structure[0]
    count = 0
    for chain in model:
        for residue in chain:
            if residue.get_resname().upper() in STANDARD_AA3:
                count += 1
    return count


# ---------- STEP 3: MINIMIZE / RELAX WITH PYROSETTA ----------

# We import and initialize PyRosetta only after other stuff is done,
# so the script can still be imported without needing PyRosetta instantly.
def init_pyrosetta():
    import pyrosetta
    from pyrosetta import rosetta

    # Flags to keep backbone close to input structure.
    # You can tune these depending on how "restrained" you want the relax to be.
    pyrosetta.init(
        "-relax:fast "
        "-constrain_relax_to_start_coords "
        "-relax:ramp_constraints false"
    )
    return pyrosetta, rosetta


def minimize_with_pyrosetta(clean_path: Path,
                            out_dir: Path = MINIMIZED_PDB_DIR,
                            n_relax_cycles: int = 3) -> Path:
    """
    Run a short, restrained FastRelax on a cleaned structure using PyRosetta.
    Saves the minimized PDB in out_dir.
    """
    pdb_id = clean_path.stem
    out_path = out_dir / f"{pdb_id}_minimized.pdb"

    if out_path.exists():
        print(f"[relax] {pdb_id} already minimized, skipping")
        return out_path

    print(f"[relax] Minimizing {pdb_id} with PyRosetta FastRelax")

    pyrosetta, rosetta = init_pyrosetta()
    pose = pyrosetta.pose_from_pdb(str(clean_path))

    # Score function with coordinate constraints
    scorefxn = rosetta.core.scoring.get_score_function()
    rosetta.core.scoring.constraints.add_fa_constraints_from_cmdline_to_scorefxn(scorefxn)

    # FastRelax with a few cycles, using default MoveMap (can be customized)
    relax = rosetta.protocols.relax.FastRelax(scorefxn, n_relax_cycles)
    relax.apply(pose)

    pose.dump_pdb(str(out_path))
    print(f"[relax] Saved minimized PDB to {out_path}")
    return out_path


# ---------- MAIN PIPELINE ----------

def main():
    selected_clean_paths = []

    # Step 1 & 2: Download and clean
    for pdb_id in PDB_IDS:
        raw_path = download_pdb(pdb_id)
        clean_path = clean_pdb(raw_path)
        length = get_sequence_length(clean_path)

        if length < MIN_LEN or length > MAX_LEN:
            print(f"[filter] {pdb_id} length {length} not in [{MIN_LEN}, {MAX_LEN}], skipping")
            continue

        print(f"[select] {pdb_id} accepted with length {length}")
        selected_clean_paths.append(clean_path)

    # Optionally limit to ~12 proteins if more pass the filter
    selected_clean_paths = selected_clean_paths[:12]

    if not selected_clean_paths:
        print("No structures passed the length filter. Adjust PDB_IDS or length bounds.")
        return

    # Step 3: Minimize / relax each selected structure
    for clean_path in selected_clean_paths:
        minimize_with_pyrosetta(clean_path)


if __name__ == "__main__":
    main()
