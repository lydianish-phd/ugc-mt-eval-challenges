#!/usr/bin/env python3
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from .constants import CORPUS_LABELS
from .utils import read_csv, write_csv

VALID_LABELS = {"default", "guided", "tie"}
PREFERENCE_ORDER = ["guided", "tie", "default"]
BINARY_ORDER = ["guided", "default"]


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def normalise_pref(label: str) -> str:
    label = (label or "").strip().lower()

    if label == "default":
        return "default"
    if label == "tie":
        return "tie"
    if label == "guided":
        return "guided"
    if label == "cannot_judge":
        return "cannot_judge"
    if label == "":
        return ""

    # Corpus-specific guided labels such as rocsmt / pfsmb.
    return "guided"


def preference_distribution(rows: list[dict], pref_col: str, level_name: str) -> list[dict]:
    labels = [normalise_pref(r[pref_col]) for r in rows]
    valid = [x for x in labels if x in VALID_LABELS]
    counts = Counter(valid)
    total = sum(counts.values())

    out = []
    for label in PREFERENCE_ORDER:
        count = counts[label]
        out.append({
            "level": level_name,
            "question": pref_col,
            "preference": label,
            "count": count,
            "proportion": count / total if total else 0.0,
        })

    cannot_judge = sum(1 for x in labels if x == "cannot_judge")
    missing = sum(1 for x in labels if x == "")

    out.append({
        "level": level_name,
        "question": pref_col,
        "preference": "cannot_judge",
        "count": cannot_judge,
        "proportion": cannot_judge / len(labels) if labels else 0.0,
    })
    out.append({
        "level": level_name,
        "question": pref_col,
        "preference": "missing",
        "count": missing,
        "proportion": missing / len(labels) if labels else 0.0,
    })

    return out


def binary_preference_distribution(rows: list[dict], pref_col: str, level_name: str) -> list[dict]:
    labels = [normalise_pref(r[pref_col]) for r in rows]
    binary = [x for x in labels if x in BINARY_ORDER]
    counts = Counter(binary)
    total = sum(counts.values())

    out = []
    for label in BINARY_ORDER:
        count = counts[label]
        out.append({
            "level": level_name,
            "question": pref_col,
            "preference": label,
            "count": count,
            "proportion_excluding_ties": count / total if total else 0.0,
            "n_excluding_ties": total,
        })

    return out


def quality_adherence_table(rows: list[dict], level_name: str) -> list[dict]:
    pairs = []

    for r in rows:
        overall = normalise_pref(r["majority_overall_pref"])
        adherence = normalise_pref(r["majority_guideline_pref"])

        if overall not in VALID_LABELS or adherence not in VALID_LABELS:
            continue

        pairs.append((overall, adherence))

    counts = Counter(pairs)
    total = sum(counts.values())

    out = []
    for overall in PREFERENCE_ORDER:
        for adherence in PREFERENCE_ORDER:
            count = counts[(overall, adherence)]
            out.append({
                "level": level_name,
                "overall_pref": overall,
                "guideline_pref": adherence,
                "count": count,
                "proportion": count / total if total else 0.0,
            })

    return out


def tradeoff_summary(rows: list[dict], level_name: str) -> list[dict]:
    pairs = []

    for r in rows:
        overall = normalise_pref(r["majority_overall_pref"])
        adherence = normalise_pref(r["majority_guideline_pref"])

        if overall not in VALID_LABELS or adherence not in VALID_LABELS:
            continue

        pairs.append((overall, adherence))

    total = len(pairs)

    patterns = {
        "guided_wins_both": lambda o, a: o == "guided" and a == "guided",
        "guided_adherence_quality_tie": lambda o, a: o == "tie" and a == "guided",
        "guided_adherence_default_quality": lambda o, a: o == "default" and a == "guided",
        "default_wins_both": lambda o, a: o == "default" and a == "default",
        "tie_on_both": lambda o, a: o == "tie" and a == "tie",
        "guided_quality_default_adherence": lambda o, a: o == "guided" and a == "default",
    }

    out = []
    for name, fn in patterns.items():
        count = sum(1 for overall, adherence in pairs if fn(overall, adherence))
        out.append({
            "level": level_name,
            "pattern": name,
            "count": count,
            "proportion": count / total if total else 0.0,
        })

    return out


def group_rows(rows: list[dict], group_col: str | None) -> dict[str, list[dict]]:
    if group_col is None:
        return {"all": rows}

    groups = {}
    for r in rows:
        key = r.get(group_col, "")
        groups.setdefault(key, []).append(r)

    return groups


def plot_preference_distribution(rows: list[dict], output_path: Path, title: str) -> None:
    questions = [
        ("majority_overall_pref", "Overall quality"),
        ("majority_guideline_pref", "Guideline adherence"),
    ]

    fig, ax = plt.subplots(figsize=(4.1, 3))

    x_positions = list(range(len(questions)))
    bottoms = [0.0 for _ in questions]

    for pref in PREFERENCE_ORDER:
        values = []
        for col, _ in questions:
            labels = [
                normalise_pref(r[col])
                for r in rows
                if normalise_pref(r[col]) in VALID_LABELS
            ]
            counts = Counter(labels)
            total = sum(counts.values())
            values.append(100 * counts[pref] / total if total else 0.0)

        bars = ax.bar(x_positions, values, bottom=bottoms, label=pref, width=0.5)

        for bar, value, bottom in zip(bars, values, bottoms):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in questions])
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.98, 0.5))
    ax.set_title(title)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_quality_adherence_heatmap(rows: list[dict], output_path: Path, title: str) -> None:
    matrix = []

    valid_pairs = []
    for r in rows:
        overall = normalise_pref(r["majority_overall_pref"])
        adherence = normalise_pref(r["majority_guideline_pref"])
        if overall in VALID_LABELS and adherence in VALID_LABELS:
            valid_pairs.append((overall, adherence))

    counts = Counter(valid_pairs)
    total = sum(counts.values())

    for overall in PREFERENCE_ORDER:
        matrix.append([
            100 * counts[(overall, adherence)] / total if total else 0.0
            for adherence in PREFERENCE_ORDER
        ])

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(matrix, cmap="GnBu", vmin=0, vmax=55)

    ax.set_xticks(range(len(PREFERENCE_ORDER)))
    ax.set_yticks(range(len(PREFERENCE_ORDER)))
    ax.set_xticklabels(PREFERENCE_ORDER)
    ax.set_yticklabels(PREFERENCE_ORDER)

    ax.set_xlabel("Guideline adherence preference")
    ax.set_ylabel("Overall quality preference")
    ax.set_title(title)

    for i, overall in enumerate(PREFERENCE_ORDER):
        for j, adherence in enumerate(PREFERENCE_ORDER):
            val = matrix[i][j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Analyse human preference distributions and quality/adherence tradeoffs."
    )
    parser.add_argument("--human-eval-dir", type=str, required=True)
    parser.add_argument("--majority-csv", type=str, default=None)
    parser.add_argument(
        "--group-by",
        type=str,
        default=None,
        choices=[None, "model"],
        help="Optionally compute summaries by model or corpus.",
    )
    args = parser.parse_args()

    human_eval_dir = Path(args.human_eval_dir)
    majority_csv = (
        Path(args.majority_csv)
        if args.majority_csv
        else human_eval_dir / "human_majority_votes.csv"
    )

    rows = read_csv(majority_csv)

    groups = group_rows(rows, args.group_by)

    all_pref_rows = []
    all_binary_rows = []
    all_tradeoff_rows = []
    all_contingency_rows = []

    plots_dir = human_eval_dir / "plots"

    # The dir is name is like ab_rocsmt_... or ab_pfsmb-dev_..., so we can extract the corpus name from it.
    corpus = human_eval_dir.name.split("_")[1].split("-")[0] if "_" in human_eval_dir.name else None

    for level, group in groups.items():
        for pref_col in ["majority_overall_pref", "majority_guideline_pref"]:
            all_pref_rows.extend(preference_distribution(group, pref_col, level))
            all_binary_rows.extend(binary_preference_distribution(group, pref_col, level))

        all_tradeoff_rows.extend(tradeoff_summary(group, level))
        all_contingency_rows.extend(quality_adherence_table(group, level))

        safe_level = level.replace("/", "_").replace(" ", "_")

        plot_preference_distribution(
            group,
            plots_dir / f"preference_distribution_{safe_level}.pdf",
            title=CORPUS_LABELS.get(corpus, corpus) if corpus else safe_level
        )
        plot_quality_adherence_heatmap(
            group,
            plots_dir / f"quality_vs_adherence_{safe_level}.pdf",
            title=CORPUS_LABELS.get(corpus, corpus) if corpus else safe_level
        )

    write_csv(
        human_eval_dir / "human_preference_distribution_3class.csv",
        all_pref_rows,
        ["level", "question", "preference", "count", "proportion"],
    )

    write_csv(
        human_eval_dir / "human_preference_distribution_binary.csv",
        all_binary_rows,
        [
            "level",
            "question",
            "preference",
            "count",
            "proportion_excluding_ties",
            "n_excluding_ties",
        ],
    )

    write_csv(
        human_eval_dir / "human_quality_adherence_contingency.csv",
        all_contingency_rows,
        ["level", "overall_pref", "guideline_pref", "count", "proportion"],
    )

    write_csv(
        human_eval_dir / "human_quality_adherence_tradeoff_summary.csv",
        all_tradeoff_rows,
        ["level", "pattern", "count", "proportion"],
    )

    print(f"Saved analyses to: {human_eval_dir}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()