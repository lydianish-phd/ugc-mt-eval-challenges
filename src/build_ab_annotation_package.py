#!/usr/bin/env python3
import argparse
import csv
import os
import random
from pathlib import Path

from .constants import (
    CORPORA_CONFIG,
    DEFAULT,
    GEMMA,
    LLAMA,
    NLLB,
    TOWER,
)
from .utils import get_guideline_from_corpus, read_config, read_file, sanitize_model_name


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def read_metadata_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_system_output_file(
    experiment_dir: Path,
    model: str,
    corpus: str,
    src_file_name: str,
    guideline: str,
) -> Path:
    if model == NLLB:
        return experiment_dir / "outputs" / model / corpus / f"{src_file_name}.out.postproc"

    return (
        experiment_dir
        / "outputs"
        / model
        / corpus
        / f"{src_file_name}.{guideline}.out.postproc"
    )


def load_system_outputs(
    experiment_dir: Path,
    models: list[str],
    corpus: str,
    src_file_name: str,
    default_guideline: str,
    guided_guideline: str,
) -> dict:
    outputs = {}

    for model in models:
        default_file = get_system_output_file(
            experiment_dir=experiment_dir,
            model=model,
            corpus=corpus,
            src_file_name=src_file_name,
            guideline=default_guideline,
        )
        guided_file = get_system_output_file(
            experiment_dir=experiment_dir,
            model=model,
            corpus=corpus,
            src_file_name=src_file_name,
            guideline=guided_guideline,
        )

        if not default_file.exists():
            raise FileNotFoundError(f"Missing default output: {default_file}")
        if not guided_file.exists():
            raise FileNotFoundError(f"Missing guided output: {guided_file}")

        outputs[model] = {
            default_guideline: {
                "file": default_file,
                "lines": read_file(default_file),
            },
            guided_guideline: {
                "file": guided_file,
                "lines": read_file(guided_file),
            },
        }

    return outputs

def build_balanced_model_assignments(
    n_samples: int,
    models: list[str],
    rng: random.Random,
) -> list[str]:
    """
    Build a balanced model assignment list of length n_samples.
    """
    if not models:
        raise ValueError("At least one model must be provided.")

    assignments = []
    while len(assignments) < n_samples:
        block = models.copy()
        rng.shuffle(block)
        assignments.extend(block)

    return assignments[:n_samples]

def build_ab_annotation_package(
    sample_dir: Path,
    experiment_dir: Path,
    output_dir: Path,
    corpus: str,
    src_file_name: str,
    models: list[str],
    default_guideline: str = DEFAULT,
    guided_guideline: str = "rocsmt",
    seed: int = 13,
) -> None:
    rng = random.Random(seed)

    source_lines = read_file(sample_dir / "source.txt")
    normed_source_lines = read_file(sample_dir / "normed_source.txt")
    reference_lines = read_file(sample_dir / "reference.txt")
    metadata_rows = read_metadata_csv(sample_dir / "metadata.csv")

    if not (
        len(source_lines)
        == len(normed_source_lines)
        == len(reference_lines)
        == len(metadata_rows)
    ):
        raise ValueError(
            "Mismatch between source.txt, normed_source.txt, reference.txt, and metadata.csv lengths."
        )

    system_outputs = load_system_outputs(
        experiment_dir=experiment_dir,
        models=models,
        corpus=corpus,
        src_file_name=src_file_name,
        default_guideline=default_guideline,
        guided_guideline=guided_guideline,
    )

    system_a_lines = []
    system_b_lines = []
    annotation_rows = []
    key_rows = []

    corpus_metadata_rows = [
        (sample_idx, meta)
        for sample_idx, meta in enumerate(metadata_rows)
        if meta["corpus"] == corpus
    ]

    model_assignments = build_balanced_model_assignments(
        n_samples=len(corpus_metadata_rows),
        models=models,
        rng=rng,
    )

    item_id = 0

    for (sample_idx, meta), model in zip(corpus_metadata_rows, model_assignments):

        sentence_id = int(meta["sentence_id_0based"])

        default_output = system_outputs[model][default_guideline]["lines"][sentence_id]
        guided_output = system_outputs[model][guided_guideline]["lines"][sentence_id]

        if rng.random() < 0.5:
            system_a = default_output
            system_b = guided_output
            system_a_condition = default_guideline
            system_b_condition = guided_guideline
        else:
            system_a = guided_output
            system_b = default_output
            system_a_condition = guided_guideline
            system_b_condition = default_guideline

        system_a_lines.append(system_a)
        system_b_lines.append(system_b)

        annotation_rows.append({
            "item_id": item_id,
            "sample_id": meta["sample_id"],
            "corpus": corpus,
            "source": source_lines[sample_idx],
            "normed_source": normed_source_lines[sample_idx],
            "reference": reference_lines[sample_idx],
            "system_a": system_a,
            "system_b": system_b,
            "overall_preference": "",
            "guideline_preference": "",
            "comments": "",
        })

        key_rows.append({
            "item_id": item_id,
            "sample_id": meta["sample_id"],
            "corpus": corpus,
            "sentence_id_0based": sentence_id,
            "sentence_id_1based": sentence_id + 1,
            "labels": meta.get("labels", ""),
            "sampled_for_label": meta.get("sampled_for_label", ""),
            "model": model,
            "default_guideline": default_guideline,
            "guided_guideline": guided_guideline,
            "system_a_condition": system_a_condition,
            "system_b_condition": system_b_condition,
            "system_a_file": str(system_outputs[model][system_a_condition]["file"]),
            "system_b_file": str(system_outputs[model][system_b_condition]["file"]),
        })

        item_id += 1

    combined = list(zip(annotation_rows, key_rows, system_a_lines, system_b_lines))
    rng.shuffle(combined)

    annotation_rows = []
    key_rows = []
    system_a_lines = []
    system_b_lines = []

    for new_item_id, (ann_row, key_row, sys_a, sys_b) in enumerate(combined):
        ann_row["item_id"] = new_item_id
        key_row["item_id"] = new_item_id

        annotation_rows.append(ann_row)
        key_rows.append(key_row)
        system_a_lines.append(sys_a)
        system_b_lines.append(sys_b)

    output_dir.mkdir(parents=True, exist_ok=True)

    write_lines(output_dir / "source.txt", [row["source"] for row in annotation_rows])
    write_lines(output_dir / "normed_source.txt", [row["normed_source"] for row in annotation_rows])
    write_lines(output_dir / "reference.txt", [row["reference"] for row in annotation_rows])
    write_lines(output_dir / "system_a.txt", system_a_lines)
    write_lines(output_dir / "system_b.txt", system_b_lines)

    annotation_fieldnames = [
        "item_id",
        "sample_id",
        "corpus",
        "source",
        "normed_source",
        "reference",
        "system_a",
        "system_b",
        "overall_preference",
        "guideline_preference",
        "comments",
    ]

    key_fieldnames = [
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
        "system_a_file",
        "system_b_file",
    ]

    write_csv(output_dir / "annotation_sheet.csv", annotation_rows, annotation_fieldnames)
    write_csv(output_dir / "annotation_key.csv", key_rows, key_fieldnames)

    print(f"Saved annotation package to: {output_dir}")
    print(f"Number of annotation items: {len(annotation_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build randomized A/B human-evaluation files from sampled data and model outputs."
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        required=True,
        help="Directory containing source.txt, normed_source.txt, reference.txt, and metadata.csv.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment directory containing outputs/<model>/<corpus>/...",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <sample-dir parent>/annotations/ab_<corpus>_<models>.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional data directory used with corpora config.",
    )
    parser.add_argument(
        "--corpora-config",
        type=str,
        default=CORPORA_CONFIG,
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="rocsmt",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[GEMMA, LLAMA],
        help="Models to include in the A/B package.",
    )
    parser.add_argument(
        "--default-guideline",
        type=str,
        default=DEFAULT,
    )
    parser.add_argument(
        "--guided-guideline",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
    )
    args = parser.parse_args()

    config = read_config(args.corpora_config, args.data_dir)
    src_file = Path(config[args.corpus]["src_file_path"])
    src_file_name = src_file.name

    sample_dir = Path(args.sample_dir)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        parent_dir = sample_dir.parent

        model_tag = "_".join(sanitize_model_name(m) for m in args.models)
        output_dir_name = f"ab_{args.corpus}_{model_tag}"

        output_dir = parent_dir / "annotations" / output_dir_name
    
    guided_guideline = args.guided_guideline or f"{get_guideline_from_corpus(args.corpus)}"

    build_ab_annotation_package(
        sample_dir=sample_dir,
        experiment_dir=Path(args.experiment_dir),
        output_dir=output_dir,
        corpus=args.corpus,
        src_file_name=src_file_name,
        models=args.models,
        default_guideline=args.default_guideline,
        guided_guideline=guided_guideline,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()