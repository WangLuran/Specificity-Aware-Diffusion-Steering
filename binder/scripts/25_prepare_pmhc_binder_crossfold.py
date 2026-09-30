#!/usr/bin/env python
"""Extract BoltzGen binder sequences and write matched Boltz-2 cross-refolds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import gemmi


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    experiment = root / "experiments" / "pmhc_binder_negative_guidance"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=experiment / "targets")
    parser.add_argument(
        "--wanted_target",
        type=Path,
        help="Wanted pMHC PDB; defaults to targets/wanted_nlvpmvatv.pdb",
    )
    parser.add_argument(
        "--unwanted_target",
        type=Path,
        help=(
            "Unwanted pMHC PDB; defaults to "
            "targets/unwanted_nlvfmvatv_aligned.pdb"
        ),
    )
    parser.add_argument("--binder_chain", default="X")
    parser.add_argument("--diffusion_samples", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def pdb_sequence(path: Path, chain_id: str) -> str:
    residues: dict[tuple[int, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line[:6] not in {"ATOM  ", "HETATM"} or len(line) < 27:
            continue
        if line[21] != chain_id:
            continue
        key = (int(line[22:26]), line[26])
        residues.setdefault(key, line[17:20].strip())
    return "".join(THREE_TO_ONE.get(residues[key], "X") for key in sorted(residues))


def gemmi_chain_sequence(path: Path, preferred_chain: str, target_lengths: set[int]) -> tuple[str, str]:
    structure = gemmi.read_structure(str(path))
    model = structure[0]
    candidates: list[tuple[str, str]] = []
    for chain in model:
        letters: list[str] = []
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            letter = info.one_letter_code
            if letter and letter != " ":
                letters.append(letter)
        if letters:
            candidates.append((chain.name, "".join(letters)))

    for chain_name, sequence in candidates:
        if chain_name == preferred_chain:
            return chain_name, sequence
    plausible = [
        item
        for item in candidates
        if len(item[1]) not in target_lengths and 30 <= len(item[1]) <= 250
    ]
    if len(plausible) == 1:
        return plausible[0]
    description = ", ".join(f"{name}:{len(seq)}" for name, seq in candidates)
    raise ValueError(
        f"Could not uniquely identify binder chain {preferred_chain!r} in "
        f"{path}; chains are [{description}]."
    )


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_boltz_yaml(
    path: Path,
    binder_sequence: str,
    hla_sequence: str,
    b2m_sequence: str,
    peptide_sequence: str,
) -> None:
    text = f"""version: 1
sequences:
  - protein:
      id: X
      sequence: {binder_sequence}
      msa: empty
  - protein:
      id: A
      sequence: {hla_sequence}
  - protein:
      id: B
      sequence: {b2m_sequence}
  - protein:
      id: P
      sequence: {peptide_sequence}
      msa: empty
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_root = args.run_root.resolve()
    targets = args.targets.resolve()
    wanted_target = (
        args.wanted_target.resolve()
        if args.wanted_target is not None
        else targets / "wanted_nlvpmvatv.pdb"
    )
    unwanted_target = (
        args.unwanted_target.resolve()
        if args.unwanted_target is not None
        else targets / "unwanted_nlvfmvatv_aligned.pdb"
    )
    hla_sequence = pdb_sequence(wanted_target, "A")
    b2m_sequence = pdb_sequence(wanted_target, "B")
    wanted_peptide = pdb_sequence(wanted_target, "P")
    unwanted_peptide = pdb_sequence(unwanted_target, "P")
    if len(wanted_peptide) != len(unwanted_peptide) or not wanted_peptide:
        raise ValueError(
            "Wanted and unwanted peptides must be nonempty and have equal "
            f"lengths, got {wanted_peptide!r}, {unwanted_peptide!r}."
        )

    input_dir = run_root / "crossfold_inputs"
    output_dir = run_root / "crossfold_outputs"
    log_dir = run_root / "crossfold_logs"
    manifest_path = run_root / "binder_manifest.csv"
    config_path = run_root / "crossfold_config.yaml"
    if (manifest_path.exists() or input_dir.exists()) and not args.force:
        raise FileExistsError(
            "Crossfold inputs already exist; pass --force to regenerate."
        )
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for old_yaml in input_dir.glob("*.yaml"):
            old_yaml.unlink()

    inverse_folded = sorted(
        path
        for path in (run_root / "generation").glob(
            "lambda_*/trial_*/intermediate_designs_inverse_folded/*.cif"
        )
        if "_native" not in path.stem
    )
    if not inverse_folded:
        raise FileNotFoundError(
            f"No inverse-folded CIF files found below {run_root / 'generation'}."
        )

    rows: list[dict[str, str]] = []
    target_lengths = {len(hla_sequence), len(b2m_sequence), len(wanted_peptide)}
    for cif_path in inverse_folded:
        chain_name, binder_sequence = gemmi_chain_sequence(
            cif_path,
            args.binder_chain,
            target_lengths,
        )
        lambda_tag = next(
            part.removeprefix("lambda_")
            for part in cif_path.parts
            if part.startswith("lambda_")
        )
        trial_tag = next(
            part.removeprefix("trial_")
            for part in cif_path.parts
            if part.startswith("trial_")
        )
        sequence_match = re.search(r"_(\d+)$", cif_path.stem)
        if sequence_match is None:
            raise ValueError(
                f"Cannot parse a source-local sequence index from {cif_path.name}."
            )
        sequence_index = int(sequence_match.group(1))
        sequence_digest = hashlib.sha256(
            binder_sequence.encode("ascii")
        ).hexdigest()[:10]
        binder_id = safe_tag(
            f"ng_lam{lambda_tag}_trial{trial_tag}_"
            f"{cif_path.stem}_{sequence_digest}"
        )
        for condition, peptide in (
            ("wanted", wanted_peptide),
            ("unwanted", unwanted_peptide),
        ):
            complex_id = f"{binder_id}_{condition}"
            write_boltz_yaml(
                input_dir / f"{complex_id}.yaml",
                binder_sequence=binder_sequence,
                hla_sequence=hla_sequence,
                b2m_sequence=b2m_sequence,
                peptide_sequence=peptide,
            )
        rows.append(
            {
                "binder_id": binder_id,
                "guidance_scale_tag": lambda_tag,
                "trial": trial_tag,
                "sequence_index": str(sequence_index),
                "binder_chain_in_source": chain_name,
                "binder_length": str(len(binder_sequence)),
                "binder_sequence": binder_sequence,
                "source_cif": str(cif_path),
                "wanted_complex_id": f"{binder_id}_wanted",
                "unwanted_complex_id": f"{binder_id}_unwanted",
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    config_text = f"""project:
  name: pmhc_binder_negative_guidance_crossfold
paths:
  boltz_inputs_dir: {input_dir}
  boltz_outputs_dir: {output_dir}
  logs_dir: {log_dir}
  runtime_log: {run_root / 'crossfold_runtime.csv'}
boltz:
  diffusion_samples: {args.diffusion_samples}
  command:
    executable: {root / '.conda_boltz' / 'bin' / 'boltz'}
    args:
      - predict
      - "{{yaml_path}}"
      - --out_dir
      - "{{outputs_dir}}"
      - --cache
      - {root / 'cache_boltz'}
      - --devices
      - "1"
      - --accelerator
      - gpu
      - --model
      - boltz2
      - --diffusion_samples
      - "{args.diffusion_samples}"
      - --max_parallel_samples
      - "1"
      - --recycling_steps
      - "3"
      - --sampling_steps
      - "{args.sampling_steps}"
      - --use_msa_server
      - --output_format
      - mmcif
      - --override
"""
    config_path.write_text(config_text, encoding="utf-8")
    print(f"Wrote {len(rows)} binders to {manifest_path}")
    print(f"Wrote {2 * len(rows)} matched crossfold inputs to {input_dir}")
    print(f"Wrote scheduler config to {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
