#!/bin/bash
#SBATCH --job-name=rechunk
###SBATCH --output=<path-to-log-file>
#SBATCH --array=0-3
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
conda activate andrew-clone

NODE_INDEX=$SLURM_ARRAY_TASK_ID

export HF_TOKEN=<hf-token-here>

srun --ntasks-per-node=1 --gres=gpu:8 --exclusive bash -c '
for ((i=0; i<8; i++)); do
    NODE_INDEX=$SLURM_ARRAY_TASK_ID
    SHARD_ID=$((NODE_INDEX * 8 + i))
    echo $SHARD_ID
    CUDA_VISIBLE_DEVICES=$i \
    python -u rechunker_parallel.py \
        --shard-id=$SHARD_ID \
        --shards=32 \
        --user={USER} \
        --input=chunks_with_retrieve \
        --split=train \
        --output=split_train_rechunked \
        --cache-dir=/scratch &
done

wait
'

# srun --ntasks-per-node=1 --gres=gpu:8 --exclusive bash -c '
# for ((i=0; i<8; i++)); do
#     NODE_INDEX=$SLURM_ARRAY_TASK_ID
#     SHARD_ID=$((NODE_INDEX * 8 + i))
#     echo $SHARD_ID
#     CUDA_VISIBLE_DEVICES=$i \
#     python -u rechunker_parallel.py \
#         --shard-id=$SHARD_ID \
#         --shards=32 \
#         --user={USER} \
#         --input=new_chunks_with_retrieve \
#         --split=dev \
#         --output=split_dev_rechunked \
#         --cache-dir=/scratch &
# done
# 
# wait
# '
