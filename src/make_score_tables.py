#!/usr/bin/env python3
import argparse
import os
import re

import numpy as np
import pandas as pd

from .constants import (
    GRANITE,
    MISTRAL,
    NLLB,
    LLAMA,
    GEMMA,
    QWEN,
    TOWER,
    MODEL_LABELS,
    ROCSMT,
    FOOTWEETS,
    MMTC,
    PFSMB,
    CORPUS_LABELS,
    DEFAULT,
    GUIDELINE_LABELS,
    BLEU,
    COMET,
    COMETKIWI,
    VS_NLLB,
    VS_DEFAULT,
    HIGHER_SCORE_MARKER,
    LOWER_SCORE_MARKER,
)

from .utils import (
    extract_guideline,
)

GUIDELINE_ORDER = [DEFAULT, ROCSMT, FOOTWEETS, MMTC, PFSMB]
CORPUS_ORDER = GUIDELINE_ORDER[1:]
MODEL_ORDER_ALL = [NLLB, GEMMA, GRANITE, LLAMA, MISTRAL, QWEN, TOWER]
MODEL_ORDER_NO_NLLB = MODEL_ORDER_ALL[1:]
METRICS = [BLEU, COMET, COMETKIWI]


def significance_marker(is_sig) -> str:
    return "*" if bool(is_sig) else ""


def arrow_latex(delta) -> str:
    if pd.isna(delta):
        return ""
    if delta > 0:
        return HIGHER_SCORE_MARKER
    if delta < 0:
        return LOWER_SCORE_MARKER
    return ""


def get_csv_path(score_dir: str, corpus: str, comparison_mode: str) -> str:
    if comparison_mode == VS_NLLB:
        return os.path.join(score_dir, f"scores_ci_{corpus}.csv")
    if comparison_mode == VS_DEFAULT:
        return os.path.join(score_dir, f"scores_ci_default_{corpus}.csv")
    raise ValueError(f"Unsupported comparison mode: {comparison_mode}")


def default_output_filename(metric: str, comparison_mode: str) -> str:
    if comparison_mode == VS_NLLB:
        return f"{metric}_compact_table.tex"
    if comparison_mode == VS_DEFAULT:
        return f"{metric}_compact_table_vs_default.tex"
    raise ValueError(f"Unsupported comparison mode: {comparison_mode}")


def load_all_metric_rows(score_dir: str, comparison_mode: str) -> pd.DataFrame:
    frames = []
    for corpus in CORPUS_ORDER:
        csv_path = get_csv_path(score_dir, corpus, comparison_mode)
        df = pd.read_csv(csv_path)
        df["guideline"] = df["file"].apply(extract_guideline)
        df["corpus"] = corpus
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _find_best_values(df: pd.DataFrame, metric: str, comparison_mode: str) -> dict:
    """
    Best raw score per corpus column over all rows that appear in the table.
    Used to bold the best scores overall.
    """
    best_values = {}

    if comparison_mode == VS_NLLB:
        candidate_models = MODEL_ORDER_ALL
        candidate_guidelines = GUIDELINE_ORDER + ["baseline"]
    elif comparison_mode == VS_DEFAULT:
        candidate_models = MODEL_ORDER_NO_NLLB + [NLLB]
        candidate_guidelines = GUIDELINE_ORDER + ["baseline"]
    else:
        raise ValueError(f"Unsupported comparison mode: {comparison_mode}")

    filtered = df[
        df["model"].isin(candidate_models) &
        df["guideline"].isin(candidate_guidelines)
    ].copy()

    for corpus in CORPUS_ORDER:
        corpus_df = filtered[filtered["corpus"] == corpus]
        values = corpus_df[metric].dropna()
        best_values[corpus] = values.max() if not values.empty else np.nan

    return best_values


def _find_family_best_values(df: pd.DataFrame, metric: str, comparison_mode: str) -> dict:
    """
    Best raw score per (model family, corpus) among guideline rows.
    Used to underline the best guideline within each model family.
    For VS_DEFAULT, this includes DEFAULT so that 'None' can be underlined if it is best.
    For VS_NLLB, this excludes the NLLB baseline row and considers guideline rows only.
    """
    family_best = {}

    if comparison_mode == VS_NLLB:
        candidate_models = MODEL_ORDER_NO_NLLB
        candidate_guidelines = GUIDELINE_ORDER
    elif comparison_mode == VS_DEFAULT:
        candidate_models = MODEL_ORDER_NO_NLLB
        candidate_guidelines = GUIDELINE_ORDER
    else:
        raise ValueError(f"Unsupported comparison mode: {comparison_mode}")

    filtered = df[
        df["model"].isin(candidate_models) &
        df["guideline"].isin(candidate_guidelines)
    ].copy()

    for model in candidate_models:
        family_best[model] = {}
        model_df = filtered[filtered["model"] == model]
        for corpus in CORPUS_ORDER:
            corpus_df = model_df[model_df["corpus"] == corpus]
            values = corpus_df[metric].dropna()
            family_best[model][corpus] = values.max() if not values.empty else np.nan

    return family_best


def _format_score_cell(
    score,
    delta,
    is_sig,
    best_value,
    family_best_value=None,
    show_markers: bool = True,
) -> str:
    if pd.isna(score):
        return ""

    score_str = f"{score:.2f}"

    if pd.notna(family_best_value) and np.isclose(score, family_best_value):
        score_str = rf"\underline{{{score_str}}}"

    if pd.notna(best_value) and np.isclose(score, best_value):
        score_str = rf"\textbf{{{score_str}}}"

    if not show_markers:
        return score_str

    arrow = arrow_latex(delta)
    marker = significance_marker(is_sig)
    return f"{score_str}{arrow}{marker}"


def build_metric_table_compact(score_dir: str, metric: str, comparison_mode: str) -> pd.DataFrame:
    df = load_all_metric_rows(score_dir, comparison_mode)

    score_col = metric
    delta_col = f"{metric}_delta_mean"
    sig_col = f"{metric}_significant_95_ci"

    rows = []
    best_values = _find_best_values(df, metric, comparison_mode)
    family_best_values = _find_family_best_values(df, metric, comparison_mode)

    if comparison_mode == VS_NLLB:
        model_order = MODEL_ORDER_ALL
    elif comparison_mode == VS_DEFAULT:
        model_order = MODEL_ORDER_NO_NLLB
    else:
        raise ValueError(f"Unsupported comparison mode: {comparison_mode}")

    # Add NLLB reference row at top for vs_default
    if comparison_mode == VS_DEFAULT:
        nllb_df = df[df["model"] == NLLB].copy()
        row = {
            "Model": MODEL_LABELS.get(NLLB, NLLB),
            "Guideline": "—",
        }

        for corpus in CORPUS_ORDER:
            corpus_df = nllb_df[
                (nllb_df["corpus"] == corpus) & (nllb_df["guideline"] == "baseline")
            ]
            value = corpus_df.iloc[0][score_col] if not corpus_df.empty else np.nan
            row[CORPUS_LABELS.get(corpus, corpus)] = _format_score_cell(
                score=value,
                delta=np.nan,
                is_sig=False,
                best_value=best_values[corpus],
                family_best_value=None,
                show_markers=False,
            )

        rows.append(row)

    for model in model_order:
        model_df = df[df["model"] == model].copy()
        model_label = MODEL_LABELS.get(model, model)

        if comparison_mode == VS_NLLB and model == NLLB:
            row = {"Model": model_label, "Guideline": "—"}

            for corpus in CORPUS_ORDER:
                corpus_df = model_df[
                    (model_df["corpus"] == corpus) & (model_df["guideline"] == "baseline")
                ]
                value = corpus_df.iloc[0][score_col] if not corpus_df.empty else np.nan
                row[CORPUS_LABELS.get(corpus, corpus)] = _format_score_cell(
                    score=value,
                    delta=np.nan,
                    is_sig=False,
                    best_value=best_values[corpus],
                    family_best_value=None,
                    show_markers=False,
                )

            rows.append(row)
            continue

        for i, guideline in enumerate(GUIDELINE_ORDER):
            row = {
                "Model": model_label if i == 0 else "",
                "Guideline": GUIDELINE_LABELS.get(guideline, guideline),
            }

            for corpus in CORPUS_ORDER:
                sub = model_df[
                    (model_df["corpus"] == corpus) & (model_df["guideline"] == guideline)
                ]

                if sub.empty or pd.isna(sub.iloc[0].get(score_col, np.nan)):
                    row[CORPUS_LABELS.get(corpus, corpus)] = ""
                    continue

                r = sub.iloc[0]
                family_best = family_best_values.get(model, {}).get(corpus, np.nan)

                if comparison_mode == VS_DEFAULT and guideline == DEFAULT:
                    row[CORPUS_LABELS.get(corpus, corpus)] = _format_score_cell(
                        score=r[score_col],
                        delta=np.nan,
                        is_sig=False,
                        best_value=best_values[corpus],
                        family_best_value=family_best,
                        show_markers=False,
                    )
                else:
                    row[CORPUS_LABELS.get(corpus, corpus)] = _format_score_cell(
                        score=r[score_col],
                        delta=r.get(delta_col, np.nan),
                        is_sig=r.get(sig_col, False),
                        best_value=best_values[corpus],
                        family_best_value=family_best,
                        show_markers=True,
                    )

            rows.append(row)

    table_df = pd.DataFrame(rows)
    ordered_cols = ["Model", "Guideline"] + [CORPUS_LABELS.get(c, c) for c in CORPUS_ORDER]
    table_df = table_df[ordered_cols]
    return table_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate compact LaTeX score tables with improvement and significance markers."
    )
    parser.add_argument(
        "--score-dir",
        type=str,
        required=True,
        help="Directory containing aggregated score CSV files.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=METRICS,
        default=METRICS,
        help="One or more metrics to export.",
    )
    parser.add_argument(
        "--comparison-mode",
        type=str,
        choices=[VS_NLLB, VS_DEFAULT],
        default=VS_NLLB,
        help="Whether to build tables against NLLB or against each model's default configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where .tex tables will be written. Defaults to --score-dir.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional explicit output filename. Only valid with a single metric.",
    )
    args = parser.parse_args()

    if args.output_name is not None and len(args.metrics) > 1:
        raise ValueError("--output-name can only be used with a single metric.")

    output_dir = args.output_dir or args.score_dir
    os.makedirs(output_dir, exist_ok=True)

    for metric in args.metrics:
        table_df = build_metric_table_compact(
            score_dir=args.score_dir,
            metric=metric,
            comparison_mode=args.comparison_mode,
        )

        output_name = args.output_name or default_output_filename(
            metric=metric,
            comparison_mode=args.comparison_mode,
        )
        output_path = os.path.join(output_dir, output_name)

        table_df.to_latex(
            output_path,
            index=False,
            escape=False,
            column_format="ll" + "c" * len(CORPUS_ORDER),
        )
        print(f"Saved table to: {output_path}")


if __name__ == "__main__":
    main()