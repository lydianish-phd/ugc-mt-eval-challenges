#!/bin/bash

#SBATCH --job-name=generate      # Job name
#SBATCH --account=ncm@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1		# node count
#SBATCH --ntasks-per-node=1               # number of tasks per node
#SBATCH --gres=gpu:1        # numper of GPUs per node
#SBATCH --cpus-per-task=96	# number of cores per task (8x8 = 64 cores, or all the cores)
#SBATCH --hint=multithread       # we get physical cores not logical
#SBATCH --time=0:30:00                # Time limit hrs:min:sec
#SBATCH --output=/lustre/fsn1/projects/rech/ncm/udc54vm/evaluation-challenges/logs/%x/%x_%j.log # Standard output and error log
#SBATCH --array=0-4

echo "### Running $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID ###"

module purge
# Using pytorch-gpu/py3/2.5.0 which has pytorch 2.4.0 and vllm 0.6.1
module load arch/h100 pytorch-gpu/py3/2.5.0 # works with arch/a100 as well
source $HOME/.bashrc 
# source $HOME/.bash_profile

set -e

# Get repo root (parent of slurm/)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPERIMENT_DIR=$REPO_DIR/experiments/eamt2026
DATASETS=$REPO_DIR/data_extracted
MODELS=$REPO_DIR/models

MODEL_NAME[0]=meta-llama/Llama-3.1-8B-Instruct
MODEL_DIR[0]=$MODELS/${MODEL_NAME[0]}

MODEL_NAME[1]=google/gemma-2-9b-it
MODEL_DIR[1]=$MODELS/${MODEL_NAME[1]}

MODEL_NAME[2]=Unbabel/TowerInstruct-7B-v0.2
MODEL_DIR[2]=$MODELS/${MODEL_NAME[2]}

MODEL_NAME[3]=Qwen/Qwen2.5-7B-Instruct
MODEL_DIR[3]=$MODELS/${MODEL_NAME[3]}

MODEL_NAME[4]=mistralai/Mistral-7B-Instruct-v0.3
MODEL_DIR[4]=$MODELS/${MODEL_NAME[4]}


# Non-standard English

CORPUS[0]=rocsmt
DATA_DIR[0]=$DATASETS/rocsmt/test
INPUT_FILE[0]=${DATA_DIR[0]}/raw.en.test
SRC_LANG[0]=English
TGT_LANG[0]=French

CORPUS[1]=footweets
DATA_DIR[1]=$DATASETS/footweets
INPUT_FILE[1]=${DATA_DIR[1]}/detok.twitter.sent.en.txt
SRC_LANG[1]=English
TGT_LANG[1]=German

# Non-standard French

CORPUS[2]=mmtc
DATA_DIR[2]=$DATASETS/mmtc
INPUT_FILE[2]=${DATA_DIR[2]}/test.fr-en.fr
SRC_LANG[2]=French
TGT_LANG[2]=English

CORPUS[3]=pfsmb
DATA_DIR[3]=$DATASETS/pfsmb
INPUT_FILE[3]=${DATA_DIR[3]}/test.fr
SRC_LANG[3]=French
TGT_LANG[3]=English

CORPUS[4]=pfsmb-dev
DATA_DIR[4]=$DATASETS/pfsmb
INPUT_FILE[4]=${DATA_DIR[4]}/dev.fr
SRC_LANG[4]=French
TGT_LANG[4]=English

GUIDELINES=(default rocsmt footweets mmtc pfsmb)

for i in {3..4}
do
    echo "Model: ${MODEL_NAME[$i]}"

    OUTPUT_DIR=$EXPERIMENT_DIR/outputs/${MODEL_NAME[$i]}/${CORPUS[$SLURM_ARRAY_TASK_ID]}    

    python $HOME/evaluation-challenges/src/generate.py \
        -i ${INPUT_FILE[$SLURM_ARRAY_TASK_ID]} \
        -o $OUTPUT_DIR \
        -m ${MODEL_NAME[$i]} \
        --source-lang ${SRC_LANG[$SLURM_ARRAY_TASK_ID]} \
        --target-lang ${TGT_LANG[$SLURM_ARRAY_TASK_ID]} \
        -g ${GUIDELINES[@]} 

done

echo "Done..."


