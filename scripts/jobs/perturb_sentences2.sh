#!/bin/bash

#SBATCH --ntasks=128
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=ptb_local_shuffle2
#SBATCH --output=logs/ptb_local_shuffle2.out
#SBATCH --error=logs/ptb_local_shuffle2.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."


input_file="data/ptb/ptb.all.txt.gz"
exp_dir="results/ptb_local_shuffle"


python scripts/perturb_sentences.py \
    --input_file "$input_file" \
    --exp_dir "$exp_dir" \
    --perturb_config_file "config/perturbation_func_local2.json" \
    --n_workers 128
