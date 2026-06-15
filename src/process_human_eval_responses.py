#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from .utils import read_csv, write_csv

OVERALL_QUESTION = "Which translation is better overall?"
GUIDELINE_QUESTION = "Which output better follows the UGC guidelines?"
COMMENT_QUESTION = "Optional comment"


def decode_preference(answer: str, a_condition: str, b_condition: str) -> str:
    answer = (answer or "").strip().lower()

    if answer == "a":
        return a_condition
    if answer == "b":
        return b_condition
    if answer == "tie":
        return "tie"
    if answer == "cannot judge":
        return "cannot_judge"
    return ""


def process_responses(
    responses_tsv: Path,
    annotation_key_csv: Path,
    output_csv: Path,
    annotator_prefix: str = "annotator",
) -> None:
    key_rows = read_csv(annotation_key_csv)
    key_by_item = {int(row["item_id"]): row for row in key_rows}

    with open(responses_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows_raw = list(reader)

    headers = rows_raw[0]
    response_rows = rows_raw[1:]

    n_items = len(key_rows)

    expected_cols = 1 + (n_items * 3)  # Timestamp + 3 questions per item
    if len(headers) != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns: 1 timestamp + {n_items}*3 item columns, "
            f"but found {len(headers)} columns."
        )

    rows = []

    for annotator_idx, response in enumerate(response_rows):
        annotator = f"{annotator_prefix}_{annotator_idx + 1}"

        for item_id in range(n_items):
            key = key_by_item[item_id]

            base_col = 1 + item_id * 3

            overall_raw = response[base_col]
            guideline_raw = response[base_col + 1]
            comment = response[base_col + 2]

            overall_pref = decode_preference(
                overall_raw,
                key["system_a_condition"],
                key["system_b_condition"],
            )
            guideline_pref = decode_preference(
                guideline_raw,
                key["system_a_condition"],
                key["system_b_condition"],
            )

            rows.append({
                "annotator": annotator,
                "item_id": item_id,
                "sample_id": key["sample_id"],
                "corpus": key["corpus"],
                "sentence_id_0based": key["sentence_id_0based"],
                "sentence_id_1based": key["sentence_id_1based"],
                "labels": key.get("labels", ""),
                "sampled_for_label": key.get("sampled_for_label", ""),
                "model": key["model"],
                "default_guideline": key["default_guideline"],
                "guided_guideline": key["guided_guideline"],
                "system_a_condition": key["system_a_condition"],
                "system_b_condition": key["system_b_condition"],
                "overall_raw": overall_raw,
                "guideline_raw": guideline_raw,
                "overall_pref": overall_pref,
                "guideline_pref": guideline_pref,
                "comment": comment,
            })

    fieldnames = [
        "annotator",
        "item_id",
        "sample_id",
        "corpus",
        "sentence_id_0based",
        "sentence_id_1based",
        "labels",
        "sampled_for_label",
        "model",
        "default_guideline",
        "guided_guideline",
        "system_a_condition",
        "system_b_condition",
        "overall_raw",
        "guideline_raw",
        "overall_pref",
        "guideline_pref",
        "comment",
    ]

    write_csv(output_csv, rows, fieldnames)
    print(f"Saved processed annotations to: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-eval-dir", type=str, required=True)
    parser.add_argument("--responses-tsv", type=str, default=None)
    parser.add_argument("--annotation-key", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    human_eval_dir = Path(args.human_eval_dir)

    outputs_dir = human_eval_dir / "outputs"
    inputs_dir = human_eval_dir / "inputs"

    responses_tsv = (
        Path(args.responses_tsv)
        if args.responses_tsv
        else outputs_dir / "responses.tsv"
    )
    annotation_key = (
        Path(args.annotation_key)
        if args.annotation_key
        else inputs_dir / "annotation_key.csv"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else outputs_dir / "human_annotations_long.csv"
    )
    
    process_responses(
        responses_tsv=responses_tsv,
        annotation_key_csv=annotation_key,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()