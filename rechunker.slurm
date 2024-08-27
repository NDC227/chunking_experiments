#!/bin/bash
#SBATCH --job-name=rechunk
###SBATCH --output=<path-to-log-file>
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --partition=a3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=800G
#SBATCH --time=10:00:00

set -eo pipefail
set -x

ulimit -Sn 20480

source /env/bin/start-ctx-user
conda activate gbm-clm

NODE_INDEX=$SLURM_ARRAY_TASK_ID

###export HF_TOKEN=<hf-token-here>

srun --ntasks-per-node=1 --gres=gpu:8 --exclusive bash -c '
for ((i=0; i<8; i++)); do
    NODE_INDEX=$SLURM_ARRAY_TASK_ID
    SHARD_ID=$((NODE_INDEX * 8 + i))
    echo $SHARD_ID
    CUDA_VISIBLE_DEVICES=$i \
    python -u indexer.py \
        --shard-id=$SHARD_ID \
        --shards=16 \
        --input=new_chunks_with_retrieve \
        --split=dev \
        --output=toy_dev_rechunked &
done

wait
'