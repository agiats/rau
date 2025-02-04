#!/bin/bash

#SBATCH --ntasks=128
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=BLLIP_XS_shuffle
#SBATCH --output=logs/BLLIP_XS_shuffle.out
#SBATCH --error=logs/BLLIP_XS_shuffle.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."

# Input files
train_file="data/BLLIP/XS/train.txt"
dev_file="data/BLLIP/XS/dev.txt"
test_file="data/BLLIP/XS/test.txt"
exp_dir="data/fairseq_train/BLLIP_XS"

python scripts/perturb_split_sentences.py \
    --train_file "$train_file" \
    --dev_file "$dev_file" \
    --test_file "$test_file" \
    --exp_dir "$exp_dir" \
    --perturb_config_file "config/perturbation_func_local2.json" \
    --n_workers 128
