#!/bin/bash

source $HOME/.bashrc 
source $HOME/.bash_profile

set -e

EXPERIMENT_DIR=$EXPERIMENTS/evaluation-challenges/eamt2026
CORPORA_CONFIG=$HOME/evaluation-challenges/src/config/corpora.yaml

TOWER=Unbabel/TowerInstruct-7B-v0.2
GEMMA=google/gemma-2-9b-it
LLAMA=meta-llama/Llama-3.1-8B-Instruct
NLLB=facebook/nllb-200-3.3B

ALL_MODELS=($NLLB $LLAMA $GEMMA $TOWER)

echo "Models: ${ALL_MODELS[@]}"

for MODEL in ${ALL_MODELS[@]}; do
    echo "Processing $MODEL"
    OUTPUT_DIR=$EXPERIMENT_DIR/outputs/$MODEL

    # Postprocess the output
    python3 $HOME/evaluation-challenges/src/postprocess.py \
        -o $OUTPUT_DIR \
        -c $CORPORA_CONFIG 

done

echo "Done..."


