#!/bin/bash
#SBATCH --job-name=combine
###SBATCH --output=<path-to-log-file>
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --partition=cpumedspot
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=3:00:00

set -x

ulimit -Sn 20480

source /env/bin/start-ctx-user
conda activate andrew-clone

python combine_datasets.py --user {USER} --dataset split_dev_rechunked --config RecursiveCharacterTextSplitter_500_0 --split dev --output-name combined_dev_rechunked --option 1
python combine_datasets.py --user {USER} --dataset split_dev_rechunked --config RecursiveCharacterTextSplitter_250_0 --split dev --output-name combined_dev_rechunked --option 1
python combine_datasets.py --user {USER} --dataset split_dev_rechunked --config ClusterSemanticChunker_500_0 --split dev --output-name combined_dev_rechunked --option 1
python combine_datasets.py --user {USER} --dataset split_dev_rechunked --config KamradtModifiedChunker_500_0 --split dev --output-name combined_dev_rechunked --option 1
python combine_datasets.py --user {USER} --dataset dev_rechunked --output-name combined_dev_rechunked --split dev --option 2

# python combine_datasets.py --user {USER} --dataset split_train_rechunked --config RecursiveCharacterTextSplitter_500_0 --split train --output-name combined_train_rechunked --option 1
# python combine_datasets.py --user {USER} --dataset split_train_rechunked --config RecursiveCharacterTextSplitter_250_0 --split train --output-name combined_train_rechunked --option 1
# python combine_datasets.py --user {USER} --dataset split_train_rechunked --config ClusterSemanticChunker_500_0 --split train --output-name combined_train_rechunked --option 1
# python combine_datasets.py --user {USER} --dataset split_train_rechunked --config KamradtModifiedChunker_500_0 --split train --output-name combined_train_rechunked --option 1
# python combine_datasets.py --user {USER} --dataset train_rechunked --output-name combined_train_rechunked --split train --option 2
