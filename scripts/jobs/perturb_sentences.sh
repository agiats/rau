#!/bin/bash

#SBATCH --ntasks=128
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=perturb_sentences
#SBATCH --output=logs/perturb_sentences.out
#SBATCH --error=logs/perturb_sentences.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."


input_file="results/length_sampling/100M_samples_eos_zipf_min1_max20/samples.txt.gz"
exp_dir="results/length_sampling/100M_samples_eos_zipf_min1_max20"

python scripts/perturb_sentences.py \
    --input_file "$input_file" \
    --exp_dir "$exp_dir" \
    --perturb_config_file "config/perturbation_func.json" \
    --n_workers 128
