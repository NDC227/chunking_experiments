#!/bin/bash
#SBATCH --job-name=scorer
###SBATCH --output=<path-to-log-file>
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --partition=a3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=800G
#SBATCH --time=1-00:00:00

set -eo pipefail
set -x

ulimit -Sn 20480

source /env/bin/start-ctx-user
conda activate andrew-clone

NODE_INDEX=$SLURM_ARRAY_TASK_ID

export HF_TOKEN=<hf-token-here>

srun --ntasks-per-node=1 --gres=gpu:8 --exclusive bash -c '
###for ((i=0; i<8; i++)); do
    NODE_INDEX=$SLURM_ARRAY_TASK_ID
    SHARD_ID=$NODE_INDEX
    echo $SHARD_ID
    ###CUDA_VISIBLE_DEVICES=$i \
    python -u get_llm_parallel.py \
        --shard-id=$SHARD_ID \
        --shards=4 \
        --user={USER} \
        --dataset=combined_train_rechunked \
        --split=train \
        --output_name=combined_train_scored \
        --llm-name=meta-llama/Meta-Llama-3.1-8B-Instruct \
        --downsample=100 \
        --batch-size=2 \
        --max-seq-length=400 \
        --cache_dir=/scratch &
###done

wait
'
