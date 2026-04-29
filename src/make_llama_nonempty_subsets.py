#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Sequence

from .constants import (
    LLAMA,
    NLLB,
    TOWER,
    GEMMA,
    CORPORA_CONFIG,
    DEFAULT,
    ROCSMT,
    FOOTWEETS,
    MMTC,
    PFSMB,
)
from .utils import (
    read_file,
    read_config,
    write_lines,
    write_json,
)
from .prompt_templates import REFUSAL_TO_TRANSLATE

GUIDELINES = [DEFAULT, ROCSMT, FOOTWEETS, MMTC, PFSMB]
ALL_MODELS = [NLLB, LLAMA, GEMMA, TOWER]
ALL_CORPORA = GUIDELINES[1:] + [f"{PFSMB}-dev"] # All except DEFAULT, which is not a corpus, and adding PFSMB-dev which is a separate corpus with the same guideline as PFSMB   

def is_empty_or_refusal(output: str) -> bool:
    normalised_output = output.strip()
    return (
        normalised_output == ""
        or normalised_output == REFUSAL_TO_TRANSLATE
    )

def keep_indices_from_llama_outputs(
    llama_files: List[str],
) -> Dict[str, List[int]]:
    """
    Return indices to keep / skip.
    A line is skipped if ANY LLaMA output file has an empty line or a refusal marker at that index.
    """
    if not llama_files:
        raise ValueError("No LLaMA files were provided.")

    llama_outputs = [read_file(path) for path in llama_files]

    n_lines = len(llama_outputs[0])
    for path, lines in zip(llama_files, llama_outputs):
        if len(lines) != n_lines:
            raise ValueError(
                f"Mismatched number of lines across LLaMA outputs: "
                f"{path} has {len(lines)} lines, expected {n_lines}"
            )

    keep_indices = []
    skipped_indices = []

    for i in range(n_lines):
        has_invalid_output = any(
            is_empty_or_refusal(lines[i])
            for lines in llama_outputs
        )

        if has_invalid_output:
            skipped_indices.append(i)
        else:
            keep_indices.append(i)

    return {
        "keep_indices": keep_indices,
        "skipped_indices": skipped_indices,
    }


def subset_lines(lines: Sequence[str], keep_indices: Sequence[int]) -> List[str]:
    return [lines[i] for i in keep_indices]


def build_output_file(
    experiment_dir: str,
    model: str,
    corpus: str,
    src_file_name: str,
    guideline: str = None,
) -> str:
    if model == NLLB:
        filename = f"{src_file_name}.out.postproc"
    else:
        if guideline is None:
            raise ValueError(f"Guideline must be provided for non-NLLB model {model}")
        filename = f"{src_file_name}.{guideline}.out.postproc"

    return os.path.join(experiment_dir, "outputs", model, corpus, filename)


def get_all_output_files_for_corpus(
    experiment_dir: str,
    corpus: str,
    src_file_name: str,
    models: Sequence[str],
    guidelines: Sequence[str],
) -> List[str]:
    files = []

    for model in models:
        if model == NLLB:
            files.append(build_output_file(experiment_dir, model, corpus, src_file_name))
        else:
            for guideline in guidelines:
                files.append(
                    build_output_file(
                        experiment_dir,
                        model,
                        corpus,
                        src_file_name,
                        guideline=guideline,
                    )
                )

    return files


def copy_subsetted_outputs(
    input_experiment_dir: str,
    output_experiment_dir: str,
    corpus: str,
    src_file_name: str,
    keep_indices: Sequence[int],
    models: Sequence[str],
    guidelines: Sequence[str],
) -> None:
    output_files = get_all_output_files_for_corpus(
        input_experiment_dir,
        corpus,
        src_file_name,
        models=models,
        guidelines=guidelines,
    )

    for src_path in output_files:
        if not os.path.exists(src_path):
            print(f"Skipping missing output file: {src_path}")
            continue

        lines = read_file(src_path)
        subset = subset_lines(lines, keep_indices)

        rel_path = os.path.relpath(src_path, input_experiment_dir)
        dst_path = os.path.join(output_experiment_dir, rel_path)
        write_lines(dst_path, subset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create subsetted corpora and outputs by removing sentence pairs for which "
            "any LLaMA configuration has an empty output."
        )
    )
    parser.add_argument(
        "-i",
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment directory (the one containing outputs/).",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        required=True,
        help="Parent directory containing all corpora files referenced in corpora.yaml.",
    )
    parser.add_argument(
        "-c",
        "--corpora",
        type=str,
        nargs="+",
        default=ALL_CORPORA,
        help="Corpora to process.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=ALL_MODELS,
        help="Models whose output files should be subsetted into the new experiment folder.",
    )
    parser.add_argument(
        "--guidelines",
        type=str,
        nargs="+",
        default=GUIDELINES,
        help="Guidelines to check/subset for non-NLLB models.",
    )
    parser.add_argument(
        "--corpora-config",
        type=str,
        default=CORPORA_CONFIG,
        help="Path to corpora config YAML.",
    )
    args = parser.parse_args()

    config = read_config(args.corpora_config, args.data_dir)
    raw_config = read_config(args.corpora_config, data_dir=None)

    input_experiment_dir = args.experiment_dir.rstrip("/")

    subset_experiment_dir = f"{input_experiment_dir}_subsets"
    subset_data_dir = os.path.join(subset_experiment_dir, "data")
    subset_outputs_dir = os.path.join(subset_experiment_dir, "outputs")

    os.makedirs(subset_data_dir, exist_ok=True)
    os.makedirs(subset_outputs_dir, exist_ok=True)

    print(f"Input experiment dir : {input_experiment_dir}")
    print(f"Output subset dir    : {subset_experiment_dir}")

    for corpus in args.corpora:
        print(f"\nProcessing corpus: {corpus}")

        src_file = config[corpus]["src_file_path"]
        ref_file = config[corpus]["ref_file_path"]
        src_file_name = os.path.basename(src_file)
        ref_file_name = os.path.basename(ref_file)

        src_lines = read_file(src_file)
        ref_lines = read_file(ref_file)

        if len(src_lines) != len(ref_lines):
            raise ValueError(
                f"Mismatched source/reference line counts for {corpus}: "
                f"src={len(src_lines)}, ref={len(ref_lines)}"
            )

        llama_files = [
            build_output_file(
                input_experiment_dir,
                LLAMA,
                corpus,
                src_file_name,
                guideline=guideline,
            )
            for guideline in args.guidelines
        ]

        missing_llama = [path for path in llama_files if not os.path.exists(path)]
        if missing_llama:
            raise FileNotFoundError(
                f"Missing LLaMA outputs for corpus '{corpus}':\n" + "\n".join(missing_llama)
            )

        index_info = keep_indices_from_llama_outputs(llama_files)
        keep_indices = index_info["keep_indices"]
        skipped_indices = index_info["skipped_indices"]

        if len(src_lines) != len(keep_indices) + len(skipped_indices):
            raise AssertionError("Index accounting error while computing subset.")

        print(
            f" - original size: {len(src_lines)} | "
            f"kept: {len(keep_indices)} | "
            f"skipped: {len(skipped_indices)}"
        )

        subset_src = subset_lines(src_lines, keep_indices)
        subset_ref = subset_lines(ref_lines, keep_indices)

        # Preserve corpus-relative structure under data/
        src_rel = raw_config[corpus]["src_file_path"]
        ref_rel = raw_config[corpus]["ref_file_path"]

        subset_src_path = os.path.join(subset_data_dir, src_rel)
        subset_ref_path = os.path.join(subset_data_dir, ref_rel)
        
        write_lines(subset_src_path, subset_src)
        write_lines(subset_ref_path, subset_ref)

        skipped_info = {
            "corpus": corpus,
            "src_file": src_file,
            "ref_file": ref_file,
            "n_original": len(src_lines),
            "n_kept": len(keep_indices),
            "n_skipped": len(skipped_indices),
            "keep_indices": keep_indices,
            "skipped_indices": skipped_indices,
            "llama_files_checked": llama_files,
            "criterion": (
                "skip line if any LLaMA configuration has an empty output after stripping "
                f"whitespace or equals {REFUSAL_TO_TRANSLATE!r}"
            ),
        }

        skipped_info_path = os.path.join(
            subset_data_dir,
            os.path.dirname(src_rel),
            f"{src_file_name}.skipped_indices.json",
        )
        write_json(skipped_info_path, skipped_info)

        copy_subsetted_outputs(
            input_experiment_dir=input_experiment_dir,
            output_experiment_dir=subset_experiment_dir,
            corpus=corpus,
            src_file_name=src_file_name,
            keep_indices=keep_indices,
            models=args.models,
            guidelines=args.guidelines,
        )

    print("\nDone.")
    print(f"Subset experiment written to: {subset_experiment_dir}")


if __name__ == "__main__":
    main()