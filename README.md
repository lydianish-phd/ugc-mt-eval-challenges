# Challenges of Evaluating the Translation of User-Generated Content

This repository contains the code, data organization, and experiment assets for a study on translating user-generated content (UGC) with both a strong neural machine translation baseline and instruction-tuned large language models.

This work extends **Experiment VII** from **Chapter 9: Challenges of evaluating UGC Machine Translation: A Case Study on LLMs** of my PhD thesis, and has been published at the EAMT 2026 conference.

📄 Read the article here: [When the Gold Standard Isn't Necessarily Standard: Challenges of Evaluating the Translation of User-Generated Content](https://aclanthology.org/2026.eamt-1.30/)

Authors: Lydia Nishimwe, Benoît Sagot, Rachel Bawden

🎓 Read the full thesis here: [Robust Neural Machine Translation of User-Generated Content](https://theses.hal.science/tel-05448644).

## Overview

We evaluate translation performance on four parallel UGC datasets and measure the effect of incorporating corpus-specific translation guidelines to control the style of model outputs.

Key components:

- Baseline: `NLLB-200-3.3B`
- Instruction-tuned LLMs: `Gemma-2-9B`, `Granite-4.1-8B`, `LLaMA-3.1-8B`, `Mistral-7B-v0.3`, `Qwen-2.5-7B`, and `Tower-7B-v0.2`
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
- Instruction-tuned LLMs: `Gemma-2-9B`, `Granite-4.1-8B`, `LLaMA-3.1-8B`, `Mistral-7B-v0.3`, `Qwen-2.5-7B`, `Tower-7B-v0.2`
- Note: `Tower-7B-v0.2` is specifically fine-tuned for translation tasks

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

- `Tower-7B-v0.2` is the best-performing default model on most datasets, except `RoCSMT`, where `Gemma-2-9B` is stronger.
- Most instruction-tuned LLMs outperform `NLLB-200-3.3B` on the UGC datasets, with the main exception being `MMTC`.
- `LLaMA-3.1-8B` shows substantial refusal behavior and is often harmed by guideline prompting.
- `Gemma-2-9B`, `Granite-4.1-8B`, `Mistral-7B-v0.3`, and `Qwen-2.5-7B` adapt more effectively to guideline prompts.
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

Presented at EAMT 2026. Conference paper link:

https://aclanthology.org/2026.eamt-1.30

```bibtex
@inproceedings{nishimwe-etal-2026-gold,
    title = "When the Gold Standard Isn{'}t Necessarily Standard: Challenges of Evaluating the Translation of User-Generated Content",
    author = "Nishimwe, Lydia  and
      Sagot, Beno{\^i}t  and
      Bawden, Rachel",
    editor = "Shterionov, Dimitar  and
      Vanmassenhove, Eva  and
      De Sisto, Mirella  and
      Blain, Fred  and
      Pourmostafa Roshan Sharami, Javad  and
      Lepp, Lisa  and
      Manna, Chiara  and
      Rescigno, Argentina Anna  and
      Karakanta, Alina  and
      Rigouts Terryn, Ayla  and
      Lardelli, Manuel  and
      Resende, Natalia  and
      Murgolo, Elena  and
      Hackenbuchner, Jani{\c{c}}a  and
      Zaretskaya, Anna  and
      Espl{\`a}-Gomis, Miquel  and
      Etchegoyhen, Thierry  and
      Gromann, Dagmar  and
      Bawden, Rachel  and
      Haddow, Barry  and
      Szoc, Sara  and
      Forcada, Mikel  and
      Moniz, Helena",
    booktitle = "Proceedings of the 26th Annual Conference of the {E}uropean Association for Machine Translation (Volume 1)",
    month = jun,
    year = "2026",
    address = "Tilburg, The Netherlands",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2026.eamt-1.30/",
    pages = "473--495",
    ISBN = "9789403901411",
    abstract = "User-generated content (UGC) is characterised by frequent use of non-standard language, from spelling errors to expressive choices such as slang, character repetitions, and emojis. This makes evaluating UGC translation challenging: what counts as a ``good'' translation depends on the desired standardness level of the output. To explore this, we examine the human translation guidelines of four UGC datasets, and derive a taxonomy of twelve non-standard phenomena and five translation actions (NORMALISE, COPY, TRANSFER, OMIT, CENSOR). Our analysis reveals notable differences in how UGC is treated, resulting in a spectrum of standardness in reference translations. We show that translation scores of large language models are highly sensitive to prompts with explicit UGC translation instructions, and that they improve when they align with the dataset guidelines. We argue that fair evaluation requires both models and metrics to be aware of translation guidelines. Finally, we call for clear guidelines during dataset creation and for the development of controllable, guideline-aware evaluation frameworks for UGC translation."
}
```
