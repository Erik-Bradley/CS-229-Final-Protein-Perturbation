#!/usr/bin/env python

"""
Run ProteinMPNN on all backbones in pdb_perturbed/ and collect sequences.

- 20 sequences per PDB
- sampling_temp = 0.1
- batch_size = 1
- no fixed residues

This version:
  1) runs protein_mpnn_run.py per PDB
  2) looks for .fa files under mpnn_raw/
  3) splits sequences into separate FASTA files per backbone (using
     the pdb=... tag in the FASTA headers) and writes them to
     mpnn_filtered/.
"""

import subprocess
from pathlib import Path
from collections import defaultdict

# ---------- CONFIG ----------

PDB_DIR = Path("pdb_perturbed")
PROTEIN_MPNN_DIR = Path("ProteinMPNN")  # adjust if your clone is elsewhere

RAW_OUTPUT_DIR = Path("mpnn_raw")
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_OUTPUT_DIR = Path("mpnn_filtered")
FILTERED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SEQ_PER_TARGET = 20
TEMPERATURE = 0.1
BATCH_SIZE = 1


# ---------- STEP 1: RUN PROTEINMPNN PER PDB ----------

def run_protein_mpnn():
    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in {PDB_DIR}")

    print(f"[mpnn] Found {len(pdb_files)} PDB files in {PDB_DIR}")

    for pdb_path in pdb_files:
        pdb_name = pdb_path.name
        out_subdir = RAW_OUTPUT_DIR / pdb_path.stem
        out_subdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(PROTEIN_MPNN_DIR / "protein_mpnn_run.py"),
            "--pdb_path", str(pdb_path),          # single PDB file
            "--out_folder", str(out_subdir),      # one folder per PDB
            "--num_seq_per_target", str(NUM_SEQ_PER_TARGET),
            "--batch_size", str(BATCH_SIZE),
            "--sampling_temp", str(TEMPERATURE),
        ]

        print("\n[mpnn] Running ProteinMPNN on", pdb_name)
        print("       " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[mpnn] Done: {pdb_name} -> {out_subdir}")


# ---------- FASTA HELPERS ----------

def read_fasta(path: Path):
    """Yield (header, sequence) tuples from a FASTA file."""
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


def extract_pdb_from_header(header: str) -> str:
    """
    Try to extract a pdb name from a ProteinMPNN-style FASTA header.

    Many versions include a token like 'pdb=1ubq_clean_coord_rmsd0.5'.
    If not found, we fall back to the first whitespace-separated token.
    """
    tokens = header.split()
    for tok in tokens:
        if tok.startswith("pdb="):
            return tok.split("=", 1)[1]
    return tokens[0]


# ---------- STEP 2: SPLIT SEQUENCES INTO PER-PDB FASTA FILES ----------

def collect_fasta():
    """
    Search mpnn_raw/** for *.fa, read all sequences, group them by
    backbone (using pdb=... in the header), and write one FASTA file
    per backbone into mpnn_filtered/.
    """

    fa_files = list(RAW_OUTPUT_DIR.rglob("*.fa"))
    if not fa_files:
        print(f"[collect] No .fa files found under {RAW_OUTPUT_DIR}.")
        print("          Open mpnn_raw/ in Finder or run `ls -R mpnn_raw`")
        print("          to see what filenames ProteinMPNN is producing.")
        return

    print(f"[collect] Found {len(fa_files)} FASTA files under {RAW_OUTPUT_DIR}")

    # bucket: pdb_name -> list of (header, seq)
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for fa_path in fa_files:
        print(f"[collect] Reading {fa_path}")
        for header, seq in read_fasta(fa_path):
            pdb_name = extract_pdb_from_header(header)
            buckets[pdb_name].append((header, seq))

    print(f"[collect] Grouped sequences into {len(buckets)} backbones")

    for pdb_name, entries in buckets.items():
        out_path = FILTERED_OUTPUT_DIR / f"{pdb_name}_designs.fa"
        print(f"[collect] {pdb_name}: writing {len(entries)} sequences -> {out_path.name}")
        with open(out_path, "w") as out_f:
            for h, s in entries:
                out_f.write(f">{h}\n{s}\n")

    print("[collect] Done splitting sequences into separate FASTA files in mpnn_filtered/.")


# ---------- MAIN ----------

def main():
    run_protein_mpnn()
    collect_fasta()

    # ---- Move global ProteinMPNN FASTA into mpnn_raw/global/ ----
    fname = f"T={TEMPERATURE},_designs.fa"

    # Look for the global file in a few plausible locations
    candidates = [
        Path(".") / fname,
        RAW_OUTPUT_DIR / fname,
        FILTERED_OUTPUT_DIR / fname,
    ]

    found = None
    for c in candidates:
        if c.exists():
            found = c
            break

    if found is not None:
        global_dir = RAW_OUTPUT_DIR / "global"
        global_dir.mkdir(parents=True, exist_ok=True)
        dest = global_dir / fname
        found.rename(dest)
        print(f"[clean] Moved global FASTA from {found} -> {dest}")
    else:
        print("[clean] No global FASTA file found to move.")

if __name__ == "__main__":
    main()
