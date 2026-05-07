#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sacrebleu.metrics import BLEU as bleu, CHRF as chrf

try:
    from scipy.stats import ttest_rel
except Exception:  # pragma: no cover
    ttest_rel = None

from .constants import (
    GRANITE,
    MISTRAL,
    QWEN,
    TOWER,
    LLAMA,
    GEMMA,
    NLLB,
    CORPORA_CONFIG,
    DEFAULT,
    ROCSMT,
    FOOTWEETS,
    MMTC,
    PFSMB,
    BLEU,
    CHRF,
    COMET,
    COMETKIWI,
)

from .utils import (
    read_file,
    read_json,
    write_json,
    read_config,
)

GUIDELINES = [DEFAULT, ROCSMT, FOOTWEETS, MMTC, PFSMB]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    corpus_score_fn: Callable[[Sequence[int]], float]


class BleuMetric:
    def __init__(self, ref_lines: Sequence[str], baseline_lines: Sequence[str], system_lines: Sequence[str]):
        self.ref_lines = ref_lines
        self.baseline_lines = baseline_lines
        self.system_lines = system_lines
        self.metric = bleu(effective_order=True)

    def corpus_score(self, system_lines: Sequence[str], indices: Sequence[int]) -> float:
        sampled_sys = [system_lines[i] for i in indices]
        sampled_ref = [self.ref_lines[i] for i in indices]
        return float(self.metric.corpus_score(sampled_sys, [sampled_ref]).score)

    def baseline_score(self, indices: Sequence[int]) -> float:
        return self.corpus_score(self.baseline_lines, indices)

    def system_score(self, indices: Sequence[int]) -> float:
        return self.corpus_score(self.system_lines, indices)


class MeanArrayMetric:
    def __init__(self, baseline_scores: Sequence[float], system_scores: Sequence[float]):
        self.baseline_scores = np.asarray(baseline_scores, dtype=float)
        self.system_scores = np.asarray(system_scores, dtype=float)

    def baseline_score(self, indices: Sequence[int]) -> float:
        return float(np.mean(self.baseline_scores[list(indices)]))

    def system_score(self, indices: Sequence[int]) -> float:
        return float(np.mean(self.system_scores[list(indices)]))


def paired_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if ttest_rel is not None:
        result = ttest_rel(x, y)
        return float(result.statistic), float(result.pvalue)

    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = diffs.size
    if n < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0
    t_stat = mean / (std / math.sqrt(n))
    pvalue = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return float(t_stat), float(pvalue)


def bootstrap_compare(
    baseline_score_fn: Callable[[Sequence[int]], float],
    system_score_fn: Callable[[Sequence[int]], float],
    n_items: int,
    n_splits: int = 300,
    sample_ratio: float = 0.4,
    seed: int = 13,
) -> Dict[str, float]:
    if not (0 < sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    if n_items < 2:
        raise ValueError("At least 2 examples are required for significance testing.")

    sample_size = max(2, int(round(n_items * sample_ratio)))
    rng = np.random.default_rng(seed)

    baseline_samples = np.zeros(n_splits, dtype=float)
    system_samples = np.zeros(n_splits, dtype=float)
    deltas = np.zeros(n_splits, dtype=float)

    for split_id in range(n_splits):
        indices = rng.choice(n_items, size=sample_size, replace=True)
        baseline_value = baseline_score_fn(indices)
        system_value = system_score_fn(indices)
        baseline_samples[split_id] = baseline_value
        system_samples[split_id] = system_value
        deltas[split_id] = system_value - baseline_value

    t_stat, t_pvalue = paired_ttest(system_samples, baseline_samples)

    p_ge_zero = float(np.mean(deltas >= 0.0))
    p_le_zero = float(np.mean(deltas <= 0.0))
    two_sided_pvalue = float(min(1.0, 2.0 * min(p_ge_zero, p_le_zero)))
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])

    return {
        "n_splits": int(n_splits),
        "sample_ratio": float(sample_ratio),
        "sample_size": int(sample_size),
        "baseline_mean": float(np.mean(baseline_samples)),
        "system_mean": float(np.mean(system_samples)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas, ddof=1)) if n_splits > 1 else 0.0,
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "delta_median": float(np.median(deltas)),
        "wins": int(np.sum(deltas > 0.0)),
        "losses": int(np.sum(deltas < 0.0)),
        "ties": int(np.sum(deltas == 0.0)),
        "paired_ttest": {
            "statistic": float(t_stat),
            "pvalue": float(t_pvalue),
        },
        "bootstrap": {
            "pvalue_two_sided": float(two_sided_pvalue),
            "prob_system_ge_baseline": float(p_ge_zero),
            "prob_system_le_baseline": float(p_le_zero),
        },
        "significant_95_ci": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def get_output_files(
    corpora: Iterable[str],
    models: Iterable[str],
    guidelines: Iterable[str],
    input_dir: str,
    corpora_config: str,
    data_dir: str = None,
    comparison_mode: str = "vs_nllb",
) -> List[Dict[str, str]]:
    config = read_config(corpora_config, data_dir)
    items: List[Dict[str, str]] = []

    for corpus in corpora:
        src_file = config[corpus]["src_file_path"]
        ref_file = config[corpus]["ref_file_path"]
        src_file_name = os.path.basename(src_file)

        for model in models:
            # NLLB is the reference only; do not run within-model comparisons for it
            if comparison_mode == "vs_default" and model == NLLB:
                continue

            for guideline in guidelines:
                # In within-model mode, default is the baseline and should not be compared to itself
                if comparison_mode == "vs_default" and guideline == DEFAULT:
                    continue

                system_file = os.path.join(
                    input_dir,
                    "outputs",
                    model,
                    corpus,
                    f"{src_file_name}.{guideline}.out.postproc",
                )

                if comparison_mode == "vs_nllb":
                    baseline_file = os.path.join(
                        input_dir,
                        "outputs",
                        NLLB,
                        corpus,
                        f"{src_file_name}.out.postproc",
                    )
                elif comparison_mode == "vs_default":
                    baseline_file = os.path.join(
                        input_dir,
                        "outputs",
                        model,
                        corpus,
                        f"{src_file_name}.default.out.postproc",
                    )
                else:
                    raise ValueError(f"Unsupported comparison mode: {comparison_mode}")

                items.append(
                    {
                        "corpus": corpus,
                        "model": model,
                        "guideline": guideline,
                        "ref_file": ref_file,
                        "baseline_file": baseline_file,
                        "system_file": system_file,
                        "comparison_mode": comparison_mode,
                    }
                )

    return items


def build_metric(metric_name: str, ref_file: str, baseline_file: str, system_file: str):
    metric_name = metric_name.lower()

    if metric_name == BLEU:
        ref_lines = read_file(ref_file)
        baseline_lines = read_file(baseline_file)
        system_lines = read_file(system_file)
        if not (len(ref_lines) == len(baseline_lines) == len(system_lines)):
            raise ValueError(
                f"Mismatched line counts for BLEU: ref={len(ref_lines)}, "
                f"baseline={len(baseline_lines)}, system={len(system_lines)}"
            )
        metric = BleuMetric(ref_lines, baseline_lines, system_lines)
        return metric, len(ref_lines)

    if metric_name in {COMET, COMETKIWI}:
        baseline_comet_file = f"{baseline_file}.comet.json"
        system_comet_file = f"{system_file}.comet.json"
        baseline_scores = read_json(baseline_comet_file)[metric_name]
        system_scores = read_json(system_comet_file)[metric_name]
        if len(baseline_scores) != len(system_scores):
            raise ValueError(
                f"Mismatched sentence-level scores for {metric_name}: "
                f"baseline={len(baseline_scores)}, system={len(system_scores)}"
            )
        metric = MeanArrayMetric(baseline_scores, system_scores)
        return metric, len(baseline_scores)

    raise ValueError(f"Unsupported metric: {metric_name}")


def get_scores_ci_path(system_file: str, comparison_mode: str) -> str:
    if comparison_mode == "vs_nllb":
        return f"{system_file}.scores_ci.json"
    if comparison_mode == "vs_default":
        return f"{system_file}.scores_ci_default.json"
    raise ValueError(f"Unsupported comparison mode: {comparison_mode}")


def update_scores_ci_file(output_file: str, metric_results: Dict[str, Dict], comparison_mode: str) -> None:
    scores_ci_file = get_scores_ci_path(output_file, comparison_mode)
    if os.path.exists(scores_ci_file):
        scores_ci = read_json(scores_ci_file)
    else:
        scores_ci = {}
    scores_ci.update(metric_results)
    write_json(scores_ci_file, scores_ci)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired significance testing using bootstrap resampling."
    )
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("-d", "--data-dir", type=str, required=True, help="Parent directory containing all corpora files referenced in corpora.yaml")
    parser.add_argument("-c", "--corpora", type=str, nargs="+", default=GUIDELINES[1:], help="One or more corpora to compare.")
    parser.add_argument("-m", "--models", type=str, nargs="+", default=[NLLB, GEMMA, GRANITE, LLAMA, MISTRAL, QWEN, TOWER])
    parser.add_argument("-g", "--guidelines", type=str, nargs="+", default=GUIDELINES)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=[COMET, COMETKIWI],
        choices=[BLEU, COMET, COMETKIWI],
        help="Metrics to test.",
    )
    parser.add_argument("--n-splits", type=int, default=300)
    parser.add_argument("--sample-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)
    parser.add_argument(
        "--comparison-mode",
        type=str,
        choices=["vs_nllb", "vs_default"],
        default="vs_nllb",
        help="Compare each system either against NLLB or against the default configuration of the same model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing metric-specific entries in the comparison-specific scores_ci file.",
    )
    args = parser.parse_args()

    outputs = get_output_files(
        corpora=args.corpora,
        models=args.models,
        guidelines=args.guidelines,
        input_dir=args.input_dir,
        corpora_config=args.corpora_config,
        data_dir=args.data_dir,
        comparison_mode=args.comparison_mode,
    )

    for item in outputs:
        system_file = item["system_file"]
        baseline_file = item["baseline_file"]
        ref_file = item["ref_file"]
        descriptor = (
            f"{item['model']} | {item['corpus']} | {item['guideline']} | {item['comparison_mode']}"
        )

        if not os.path.exists(system_file):
            print(f"Skipping missing system output: {descriptor}")
            continue
        if not os.path.exists(baseline_file):
            print(f"Skipping missing baseline output: {descriptor}")
            continue

        scores_ci_file = get_scores_ci_path(system_file, args.comparison_mode)
        existing_scores_ci = read_json(scores_ci_file) if os.path.exists(scores_ci_file) else {}

        metric_results: Dict[str, Dict] = {}
        for metric_name in args.metrics:
            metric_key = metric_name.lower()
            metric_already_present = metric_key in existing_scores_ci
            if metric_already_present and not args.overwrite:
                print(f"Skipping existing {metric_name} stats for {descriptor}")
                continue

            try:
                metric, n_items = build_metric(metric_name, ref_file, baseline_file, system_file)
                stats = bootstrap_compare(
                    baseline_score_fn=metric.baseline_score,
                    system_score_fn=metric.system_score,
                    n_items=n_items,
                    n_splits=args.n_splits,
                    sample_ratio=args.sample_ratio,
                    seed=args.seed,
                )
                metric_results[metric_key] = stats
                print(
                    f"Computed {metric_name} stats for {descriptor}: "
                    f"Δ={stats['delta_mean']:.4f}, "
                    f"t-test p={stats['paired_ttest']['pvalue']:.4g}, "
                    f"bootstrap p={stats['bootstrap']['pvalue_two_sided']:.4g}"
                )
            except FileNotFoundError as exc:
                print(f"Skipping {metric_name} for {descriptor}: missing file ({exc})")
            except Exception as exc:
                print(f"Failed on {metric_name} for {descriptor}: {exc}")

        if metric_results:
            update_scores_ci_file(system_file, metric_results, args.comparison_mode)


if __name__ == "__main__":
    main()