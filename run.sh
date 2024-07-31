#!/bin/bash
#SBATCH --partition=sphinx
#SBATCH --account=nlp
#SBATCH --nodelist=
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:0

#SBATCH --job-name="sample-1-gpu"
#SBATCH --output=sample-1-gpu-%j.out

## only use the following if you want email notification (uncomment if needed)
###SBATCH --mail-user=jimmyw@cs.stanford.edu
###SBATCH --mail-type=ALL

## list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /sailhome/ayc227/miniconda/bin/activate chunking

python create_datasets_copy.py --dir data --output_name new_chunks_with_retrieve --retrieve_k 100