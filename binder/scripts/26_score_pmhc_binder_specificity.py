#!/usr/bin/env python
"""Score matched Boltz-2 wanted/unwanted pMHC cross-refolds."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import gemmi
import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--contact_cutoff", type=float, default=5.0)
    parser.add_argument("--clash_cutoff", type=float, default=1.5)
    parser.add_argument(
        "--mutation_position",
        type=int,
        default=4,
        help="One-based peptide position used for mutation-contact reporting",
    )
    parser.add_argument("--expected_models", type=int, default=None)
    parser.add_argument(
        "--trial",
        action="append",
        help="Optional trial tag to score (repeatable, e.g. 003).",
    )
    parser.add_argument(
        "--output_prefix",
        default="",
        help="Prefix for output CSV names, used for subset reports.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chain_atoms(
    structure: gemmi.Structure,
    chain_name: str,
    residue_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model = structure[0]
    chain = next((chain for chain in model if chain.name == chain_name), None)
    if chain is None:
        available = ", ".join(item.name for item in model)
        raise KeyError(
            f"Chain {chain_name!r} is absent; available chains: {available}."
        )
    coordinates: list[list[float]] = []
    residue_ids: list[int] = []
    for ordinal, residue in enumerate(chain):
        if residue_index is not None and ordinal != residue_index:
            continue
        for atom in residue:
            if atom.element.name == "H":
                continue
            coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
            residue_ids.append(ordinal)
    if not coordinates:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=int)
    return np.asarray(coordinates, dtype=float), np.asarray(residue_ids, dtype=int)


def contact_statistics(
    left_xyz: np.ndarray,
    left_residue: np.ndarray,
    right_xyz: np.ndarray,
    right_residue: np.ndarray,
    cutoff: float,
) -> tuple[int, int, float]:
    if len(left_xyz) == 0 or len(right_xyz) == 0:
        return 0, 0, float("nan")
    tree = cKDTree(right_xyz)
    neighborhoods = tree.query_ball_point(left_xyz, cutoff)
    atom_pairs = 0
    residue_pairs: set[tuple[int, int]] = set()
    for left_atom, right_atoms in enumerate(neighborhoods):
        atom_pairs += len(right_atoms)
        for right_atom in right_atoms:
            residue_pairs.add(
                (
                    int(left_residue[left_atom]),
                    int(right_residue[right_atom]),
                )
            )
    min_distance = float(
        np.min(tree.query(left_xyz, k=1, workers=1)[0])
    )
    return atom_pairs, len(residue_pairs), min_distance


def symmetric_pair_iptm(confidence: dict, left: int, right: int) -> float:
    matrix = confidence["pair_chains_iptm"]
    return 0.5 * (
        float(matrix[str(left)][str(right)])
        + float(matrix[str(right)][str(left)])
    )


def model_index(path: Path) -> int:
    match = re.search(r"_model_(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"Cannot parse model index from {path}.")
    return int(match.group(1))


def score_one(
    complex_id: str,
    condition: str,
    prediction_dir: Path,
    contact_cutoff: float,
    clash_cutoff: float,
    mutation_position: int,
    expected_models: int | None,
) -> list[dict]:
    rows: list[dict] = []
    confidence_paths = sorted(
        prediction_dir.glob(f"confidence_{complex_id}_model_*.json"),
        key=model_index,
    )
    if not confidence_paths:
        raise FileNotFoundError(
            f"No confidence JSON files found in {prediction_dir}."
        )
    indices = [model_index(path) for path in confidence_paths]
    if expected_models is not None and indices != list(range(expected_models)):
        raise RuntimeError(
            f"Incomplete model set for {complex_id}: found indices {indices}, "
            f"expected {list(range(expected_models))}."
        )
    for confidence_path in confidence_paths:
        index = model_index(confidence_path)
        cif_path = prediction_dir / f"{complex_id}_model_{index}.cif"
        if not cif_path.exists():
            raise FileNotFoundError(cif_path)
        confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
        structure = gemmi.read_structure(str(cif_path))
        binder_xyz, binder_residue = chain_atoms(structure, "X")
        peptide_xyz, peptide_residue = chain_atoms(structure, "P")
        mhc_xyz, mhc_residue = chain_atoms(structure, "A")
        b2m_xyz, b2m_residue = chain_atoms(structure, "B")
        mutation_xyz, mutation_residue = chain_atoms(
            structure,
            "P",
            residue_index=mutation_position - 1,
        )

        bp_atoms, bp_residues, bp_min = contact_statistics(
            binder_xyz,
            binder_residue,
            peptide_xyz,
            peptide_residue,
            contact_cutoff,
        )
        bm_atoms, bm_residues, bm_min = contact_statistics(
            binder_xyz,
            binder_residue,
            mhc_xyz,
            mhc_residue,
            contact_cutoff,
        )
        _, mutation_contacts, mutation_min = contact_statistics(
            binder_xyz,
            binder_residue,
            mutation_xyz,
            mutation_residue,
            contact_cutoff,
        )
        target_xyz = np.concatenate([mhc_xyz, b2m_xyz, peptide_xyz], axis=0)
        target_residue = np.concatenate(
            [
                mhc_residue,
                b2m_residue + 10_000,
                peptide_residue + 20_000,
            ]
        )
        clash_atoms, clash_residues, target_min = contact_statistics(
            binder_xyz,
            binder_residue,
            target_xyz,
            target_residue,
            clash_cutoff,
        )

        binder_peptide_iptm = symmetric_pair_iptm(confidence, 0, 3)
        binder_mhc_iptm = symmetric_pair_iptm(confidence, 0, 1)
        # Transparent heuristic for paired ranking, not an affinity estimate.
        interface_contact_score = (
            0.65 * binder_peptide_iptm
            + 0.35 * binder_mhc_iptm
            + 0.01 * min(bp_residues, 10)
            - 0.10 * float(clash_atoms > 0)
        )
        rows.append(
            {
                "complex_id": complex_id,
                "condition": condition,
                "model_index": index,
                "cif_path": str(cif_path),
                "confidence_score": float(confidence["confidence_score"]),
                "global_iptm_not_used_for_ranking": float(confidence["iptm"]),
                "binder_peptide_pair_iptm": binder_peptide_iptm,
                "binder_mhc_pair_iptm": binder_mhc_iptm,
                "binder_peptide_atom_contacts": bp_atoms,
                "binder_peptide_residue_contacts": bp_residues,
                "binder_mhc_atom_contacts": bm_atoms,
                "binder_mhc_residue_contacts": bm_residues,
                "mutation_position": mutation_position,
                "mutation_binder_residue_contacts": mutation_contacts,
                "mutation_min_distance_angstrom": mutation_min,
                "binder_peptide_min_distance_angstrom": bp_min,
                "binder_mhc_min_distance_angstrom": bm_min,
                "target_min_distance_angstrom": target_min,
                "clash_atom_pairs_lt_cutoff": clash_atoms,
                "clash_residue_pairs_lt_cutoff": clash_residues,
                "interface_contact_score_not_affinity": interface_contact_score,
            }
        )
    return rows


def quantile(rows: list[dict], key: str, q: float) -> float:
    return float(np.quantile([float(row[key]) for row in rows], q))


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def maximum(rows: list[dict], key: str) -> float:
    return float(np.max([float(row[key]) for row in rows]))


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    manifest = read_csv(run_root / "binder_manifest.csv")
    if args.trial:
        selected_trials = {str(value).zfill(3) for value in args.trial}
        manifest = [
            row for row in manifest if row["trial"] in selected_trials
        ]
        if not manifest:
            raise ValueError(
                f"No binders found for trials {sorted(selected_trials)}."
            )
    outputs = run_root / "crossfold_outputs"
    model_rows: list[dict] = []
    summaries: list[dict] = []

    for binder in manifest:
        condition_rows: dict[str, list[dict]] = {}
        for condition in ("wanted", "unwanted"):
            complex_id = binder[f"{condition}_complex_id"]
            prediction_dir = (
                outputs
                / f"boltz_results_{complex_id}"
                / "predictions"
                / complex_id
            )
            scored = score_one(
                complex_id=complex_id,
                condition=condition,
                prediction_dir=prediction_dir,
                contact_cutoff=args.contact_cutoff,
                clash_cutoff=args.clash_cutoff,
                mutation_position=args.mutation_position,
                expected_models=args.expected_models,
            )
            for row in scored:
                row["binder_id"] = binder["binder_id"]
                row["guidance_scale_tag"] = binder["guidance_scale_tag"]
                row["trial"] = binder["trial"]
            model_rows.extend(scored)
            condition_rows[condition] = scored

        wanted = condition_rows["wanted"]
        unwanted = condition_rows["unwanted"]
        metric = "interface_contact_score_not_affinity"
        wanted_q25 = quantile(wanted, metric, 0.25)
        unwanted_q75 = quantile(unwanted, metric, 0.75)
        wanted_mean = mean(wanted, metric)
        unwanted_mean = mean(unwanted, metric)
        wanted_max = maximum(wanted, metric)
        unwanted_max = maximum(unwanted, metric)
        summaries.append(
            {
                "binder_id": binder["binder_id"],
                "guidance_scale_tag": binder["guidance_scale_tag"],
                "trial": binder["trial"],
                "binder_length": binder["binder_length"],
                "wanted_score_q25": wanted_q25,
                "unwanted_score_q75": unwanted_q75,
                "robust_wanted_minus_unwanted_gap": wanted_q25 - unwanted_q75,
                "wanted_score_mean": wanted_mean,
                "unwanted_score_mean": unwanted_mean,
                "mean_wanted_minus_unwanted_gap": (
                    wanted_mean - unwanted_mean
                ),
                "wanted_score_max": wanted_max,
                "unwanted_score_max": unwanted_max,
                "best_sample_wanted_minus_worst_offtarget_gap": (
                    wanted_max - unwanted_max
                ),
                "wanted_binder_peptide_iptm_mean": mean(
                    wanted,
                    "binder_peptide_pair_iptm",
                ),
                "unwanted_binder_peptide_iptm_mean": mean(
                    unwanted,
                    "binder_peptide_pair_iptm",
                ),
                "wanted_binder_peptide_iptm_max": maximum(
                    wanted,
                    "binder_peptide_pair_iptm",
                ),
                "unwanted_binder_peptide_iptm_max": maximum(
                    unwanted,
                    "binder_peptide_pair_iptm",
                ),
                "mutation_position": args.mutation_position,
                "wanted_mutation_contact_model_fraction": mean(
                    [
                        {
                            "contact": float(
                                row["mutation_binder_residue_contacts"] > 0
                            )
                        }
                        for row in wanted
                    ],
                    "contact",
                ),
                "unwanted_mutation_contact_model_fraction": mean(
                    [
                        {
                            "contact": float(
                                row["mutation_binder_residue_contacts"] > 0
                            )
                        }
                        for row in unwanted
                    ],
                    "contact",
                ),
                "wanted_clash_model_fraction": mean(
                    [
                        {
                            "clash": float(row["clash_atom_pairs_lt_cutoff"] > 0)
                        }
                        for row in wanted
                    ],
                    "clash",
                ),
                "unwanted_clash_model_fraction": mean(
                    [
                        {
                            "clash": float(row["clash_atom_pairs_lt_cutoff"] > 0)
                        }
                        for row in unwanted
                    ],
                    "clash",
                ),
                "binder_sequence": binder["binder_sequence"],
                "source_cif": binder["source_cif"],
            }
        )

    summaries.sort(
        key=lambda row: float(row["robust_wanted_minus_unwanted_gap"]),
        reverse=True,
    )
    model_scores_path = run_root / f"{args.output_prefix}model_scores.csv"
    summary_path = run_root / f"{args.output_prefix}specificity_summary.csv"
    write_csv(model_scores_path, model_rows)
    write_csv(summary_path, summaries)
    print(f"Wrote {len(model_rows)} model rows to {model_scores_path}")
    print(
        f"Wrote {len(summaries)} binder summaries to "
        f"{summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
