#!/usr/bin/env python3

import argparse
from pathlib import Path

from .utils import read_csv, write_csv


TIE_COLUMNS = [
    "majority_overall_pref",
    "majority_guideline_pref",
    "comet_pref",
    "cometkiwi_pref",
]


def is_tie(value: str) -> bool:
    return (value or "").strip().lower() == "tie"


def normalise_text(text: str) -> str:
    return (text or "").strip()


def merge_rows(annotation_rows, majority_rows):
    annotation_by_id = {
        row["item_id"]: row
        for row in annotation_rows
    }

    merged = []

    for row in majority_rows:
        item_id = row["item_id"]

        if item_id not in annotation_by_id:
            continue

        merged_row = dict(row)
        merged_row.update(annotation_by_id[item_id])

        merged.append(merged_row)

    return merged


def add_exact_match_flag(rows):
    for row in rows:
        row["exact_match"] = (
            normalise_text(row["system_a"])
            == normalise_text(row["system_b"])
        )

    return rows


def tie_breakdown(rows, column, level):
    ties = [r for r in rows if is_tie(r[column])]

    n_ties = len(ties)
    n_exact = sum(r["exact_match"] for r in ties)

    return {
        "level": level,
        "measure": column,
        "n_ties": n_ties,
        "n_exact_matches": n_exact,
        "n_non_exact_matches": n_ties - n_exact,
        "proportion_exact_matches_among_ties":
            n_exact / n_ties if n_ties else 0.0,
    }


def exact_match_breakdown(rows, column, level):
    exact = [r for r in rows if r["exact_match"]]

    n_exact = len(exact)

    n_ties = sum(
        is_tie(r[column])
        for r in exact
    )

    return {
        "level": level,
        "measure": column,
        "n_exact_matches": n_exact,
        "n_ties": n_ties,
        "proportion_ties_among_exact_matches":
            n_ties / n_exact if n_exact else 0.0,
    }


def suspicious_exact_matches(rows):
    out = []

    for row in rows:
        if not row["exact_match"]:
            continue

        overall = row["majority_overall_pref"]
        guideline = row["majority_guideline_pref"]

        if overall != "tie" or guideline != "tie":
            out.append({
                "item_id": row["item_id"],
                "sample_id": row["sample_id"],
                "model": row["model"],
                "overall_pref": overall,
                "guideline_pref": guideline,
                "system_a": row["system_a"],
                "system_b": row["system_b"],
            })

    return out


def group_rows(rows, group_col=None):
    if group_col is None:
        return {"all": rows}

    groups = {}

    for row in rows:
        key = row.get(group_col, "")

        groups.setdefault(key, []).append(row)

    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Analyse ties and exact output matches."
    )
    parser.add_argument("--human-eval-dir", type=str, required=True)
    parser.add_argument(
        "--annotation-sheet",
        type=str,
    )

    parser.add_argument(
        "--majority-csv",
        type=str,
    )

    parser.add_argument(
        "--group-by",
        default=None,
        choices=[None, "model"],
    )

    args = parser.parse_args()

    annotation_sheet = (
        Path(args.human_eval_dir) / "outputs" / "annotation_sheet.csv" 
        if args.annotation_sheet is None 
        else Path(args.annotation_sheet)
    )
    majority_csv = (
        Path(args.human_eval_dir) / "analysis" / "human_majority_with_metrics.csv"
        if args.majority_csv is None
        else Path(args.majority_csv)
    )

    output_dir = majority_csv.parent

    annotation_rows = read_csv(annotation_sheet)
    majority_rows = read_csv(majority_csv)

    rows = merge_rows(
        annotation_rows,
        majority_rows,
    )

    rows = add_exact_match_flag(rows)

    groups = group_rows(
        rows,
        args.group_by,
    )

    tie_rows = []
    exact_rows = []

    for level, group in groups.items():

        for column in TIE_COLUMNS:

            tie_rows.append(
                tie_breakdown(
                    group,
                    column,
                    level,
                )
            )

            exact_rows.append(
                exact_match_breakdown(
                    group,
                    column,
                    level,
                )
            )

    write_csv(
        output_dir / "tie_analysis_summary.csv",
        tie_rows,
        [
            "level",
            "measure",
            "n_ties",
            "n_exact_matches",
            "n_non_exact_matches",
            "proportion_exact_matches_among_ties",
        ],
    )

    write_csv(
        output_dir / "exact_match_analysis.csv",
        exact_rows,
        [
            "level",
            "measure",
            "n_exact_matches",
            "n_ties",
            "proportion_ties_among_exact_matches",
        ],
    )

    write_csv(
        output_dir / "exact_match_non_ties.csv",
        suspicious_exact_matches(rows),
        [
            "item_id",
            "sample_id",
            "model",
            "overall_pref",
            "guideline_pref",
            "system_a",
            "system_b",
        ],
    )

    total = len(rows)
    exact = sum(r["exact_match"] for r in rows)

    print()
    print(f"Total samples: {total}")
    print(
        f"Exact output matches: "
        f"{exact}/{total} "
        f"({100 * exact / total:.1f}%)"
    )
    print()

    print(f"Saved analyses to: {output_dir}")


if __name__ == "__main__":
    main()