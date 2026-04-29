#!/bin/bash

#SBATCH --job-name=evaluate      # Job name
#SBATCH --account=ncm@v100
#SBATCH --partition=gpu_p2l
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=24	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=4:00:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/ncm/udc54vm/evaluation-challenges/logs/%x/%x_%j.log # Standard output and error log

echo "### Running $SLURM_JOB_NAME @ ###"

module purge
module load pytorch-gpu/py3/2.3.0
source $HOME/.bashrc 
# source $HOME/.bash_profile

set -e

# Get repo root (parent of slurm/)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPERIMENT_DIR=$REPO_DIR/experiments/experiment_049d
DATA_DIR=$REPO_DIR/data_extracted

TOWER=Unbabel/TowerInstruct-7B-v0.2
GEMMA=google/gemma-2-9b-it
LLAMA=meta-llama/Llama-3.1-8B-Instruct
NLLB=facebook/nllb-200-3.3B
QWEN=Qwen/Qwen2.5-7B-Instruct
MISTRAL=mistralai/Mistral-7B-Instruct-v0.3

ALL_MODELS=($NLLB $LLAMA $GEMMA $TOWER $QWEN $MISTRAL)
MODELS=($QWEN $MISTRAL) # MODELS=$ALL_MODELS 

CORPORA=(rocsmt footweets mmtc pfsmb)
GUIDELINES=(default rocsmt footweets mmtc pfsmb)
METRICS=(bleu comet cometkiwi)

echo "Models: ${MODELS[@]}"
echo "Corpora: ${CORPORA[@]}"
echo "Guidelines: ${GUIDELINES[@]}"
echo "Metrics: ${METRICS[@]}"

echo "------------------------------------------------------------"
echo "Postprocessing outputs..."
echo "------------------------------------------------------------"

for MODEL in ${MODELS[@]}; do
    echo "Postprocessing $MODEL outputs"

    python $HOME/evaluation-challenges/src/postprocess.py \
        --output-dir $EXPERIMENT_DIR/outputs/${MODEL}

done

echo "------------------------------------------------------------"
echo "Evaluating translations with multiple metrics..."
echo "------------------------------------------------------------"

python $HOME/evaluation-challenges/src/evaluate.py \
    --output-dir $EXPERIMENT_DIR/outputs \
    --data-dir $DATA_DIR \
    --models ${MODELS[@]} \
    --corpora ${CORPORA[@]} \
    --guidelines ${GUIDELINES[@]}

echo "------------------------------------------------------------"
echo "Performing significance testing and aggregation..."
echo "------------------------------------------------------------"

for MODE in "vs_nllb" "vs_default" ; do
    python $HOME/evaluation-challenges/src/significance.py \
        --input-dir $EXPERIMENT_DIR \
        --data-dir $DATA_DIR \
        --models ${MODELS[@]} \
        --corpora ${CORPORA[@]} \
        --metrics ${METRICS[@]} \
        --comparison-mode $MODE

    python $HOME/evaluation-challenges/src/aggregate.py \
        --input-dir $EXPERIMENT_DIR \
        --models ${ALL_MODELS[@]} \
        --corpora ${CORPORA[@]} \
        --comparison-mode $MODE
done  

echo "------------------------------------------------------------"
echo "Performing qualitative analysis and generating plots..." 
echo "------------------------------------------------------------"

python $HOME/evaluation-challenges/src/analyze.py \
    --input-dir $EXPERIMENT_DIR 

for MODE in "vs_nllb" "vs_default"; do
    python $REPO_DIR/src/plot_delta.py \
        --score-dir $EXPERIMENT_DIR/scores \
        --output-dir $EXPERIMENT_DIR/plots \
        --metrics ${METRICS[@]} \
        --models ${ALL_MODELS[@]} \
        --comparison-mode $MODE 

    python $REPO_DIR/src/make_score_tables.py \
        --score-dir $EXPERIMENT_DIR/scores \
        --output-dir $EXPERIMENT_DIR/tables \
        --metrics ${METRICS[@]} \
        --models ${ALL_MODELS[@]} \
        --comparison-mode $MODE
done

echo "Done..."

