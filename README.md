# When the Gold Standard Isn't Necessarily Standard: Challenges of Evaluating the Translation of User-Generated Content

[Accepted at EAMT 2026 — Preprint](https://arxiv.org/abs/2512.17738)

Authors: Lydia Nishimwe, Benoît Sagot, Rachel Bawden

This repository contains the code, data organization, and experiment assets for a study on translating user-generated content (UGC) with both a strong neural machine translation baseline and instruction-tuned large language models.

## Overview

We evaluate translation performance on four parallel UGC datasets and measure the effect of incorporating corpus-specific translation guidelines to control the style of model outputs.

Key components:

- Baseline: `NLLB-200-3.3B`
- Instruction-tuned LLMs: `Gemma-2-9B`, `Granite-4.1`, `LLaMA-3.1-8B`, `Mistral-7B-Instruct-v0.3`, `Qwen-2.5-7B-Instruct`, and `Tower-4.1`
- Controlled generation with 12 corpus-specific translation guidelines
- Automatic evaluation with reference-based `COMET`, reference-less `COMETkiwi`, and surface-level `BLEU`
- Human evaluation for `RoCSMT` and `PFSMB` datasets

## Experimental Setup

### Datasets

The experiments cover four UGC translation datasets with distinct source styles and target strategies. The datasets include:

- `RoCSMT` (English → French)
- `PFSMB` (French → English), as well as the PMUMT subset from Rosales-Nuñez et al. (2021)
- `Footweets`
- `MMTC`

### Translation Models

We compare a strong encoder-decoder baseline with a suite of instruction-tuned decoder-only models.

- Baseline: `NLLB-200-3.3B`
- Instruction-tuned LLMs: `Gemma-2-9B`, `Granite-4.1`, `LLaMA-3.1-8B`, `Mistral-7B-Instruct-v0.3`, `Qwen-2.5-7B-Instruct`, `Tower-4.1`
- Note: `Tower-4.1` is specifically fine-tuned for translation tasks

### Inference Details

- `NLLB-200-3.3B`: beam search with beam size 5
- LLMs: greedy sampling with `vLLM`, BF16 mixed precision, maximum context length 2,048
- One source sentence per prompt
- Post-processing extracts translated sentences from verbose outputs and detects refusals to translate
- Maximum output length capped at 512 tokens for all models

### Controlled Generation

To guide outputs toward the style of each corpus, we use a list of 12 translation guidelines derived from our dataset-specific taxonomy.

Each prompting configuration is defined by:

- a model
- a set of translation guidelines

We compare:

1. Default prompting without any guidelines
2. Matching guidelines: corpus-specific guidelines applied to the same corpus
3. Mismatching guidelines: guidelines from a different corpus applied to the target corpus

### Evaluation Metrics

Automatic evaluation uses:

- `COMET` (reference-based)
- `COMETkiwi` (reference-less)
- `BLEU` via `sacrebleu`

Implementation details:

- Empty outputs are scored as zero
- Scores are reported as percentages
- Statistical significance is assessed with paired bootstrap resampling: 300 samples, sampling ratio 0.4
- For each metric, we compute the mean score difference, 95% confidence intervals, and p-values relative to the default no-guideline baseline

### Human Evaluation

Human evaluation was conducted on two UGC datasets:

- `RoCSMT` (English → French)
- `PFSMB` (French → English)

For the human evaluation methodology and package structure, see [`human_eval/README.md`](human_eval/README.md).

## Results Summary

- `Tower-4.1` is the best-performing default model on most datasets, except `RoCSMT`, where `Gemma-2-9B` is stronger.
- Most instruction-tuned LLMs outperform `NLLB-200-3.3B` on the UGC datasets, with the main exception being `MMTC`.
- `LLaMA-3.1-8B` shows substantial refusal behavior and is often harmed by guideline prompting.
- `Gemma-2-9B`, `Granite-4.1`, `Mistral-7B-Instruct-v0.3`, and `Qwen-2.5-7B-Instruct` adapt more effectively to guideline prompts.
- Matching corpus-specific guidelines usually improve `COMET` scores, while mismatched guidelines can still transfer positively when the source styles are similar.
- `COMETkiwi` tends to favor more standardised outputs and is less robust to highly non-standard UGC translations.
- Human evaluation confirms that guided outputs are preferred more often than default outputs for overall quality and guideline adherence.
- Agreement between humans and metrics is moderate; `COMETkiwi` aligns best with overall quality, while `COMET` better reflects guideline adherence.

## Reproducing the Experiments

Key scripts and paths:

- Data extraction: `slurm/extract_data.sh`
- Model inference: `slurm/generate.slurm`, `src/generate.py`
- Post-processing and evaluation: `slurm/evaluate.slurm`, `src/postprocess.py`, `src/evaluate.py`, `src/significance.py`, `src/aggregate.py`, `src/analyze.py`, `src/plot_delta.py`, `src/make_score_tables.py`
- Prompts and guideline templates: `src/prompt_templates.py`
- Model ID definitions: `src/constants.py`
- Experiment outputs: `experiments/eamt2026/`

## Repository Structure

- `data/`, `data_extracted/`: raw and processed dataset files
- `src/`: experiment and analysis scripts
- `slurm/`: job scripts for inference, evaluation, and data preparation
- `experiments/`: generated outputs, score tables, plots, and analysis artifacts
- `human_eval/`: human evaluation packages, sample data, and process documentation

## Paper and Citation

Accepted at EAMT 2026. Preprint paper link:

https://arxiv.org/abs/2512.17738


```bibtex
@misc{nishimwe2026goldstandardisntnecessarily,
      title={When the Gold Standard Isn't Necessarily Standard: Challenges of Evaluating the Translation of User-Generated Content}, 
      author={Lydia Nishimwe and Benoît Sagot and Rachel Bawden},
      year={2026},
      eprint={2512.17738},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.17738}, 
}
```
