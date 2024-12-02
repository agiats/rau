#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16g
#SBATCH --tmp=32g
#SBATCH --job-name=logs/calculate_lower_bound_perplexity
#SBATCH --output=logs/calculate_lower_bound_perplexity.out
#SBATCH --error=logs/calculate_lower_bound_perplexity.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."


n_samples=100_000_000
grammar_name="base-grammar_eos_zipf.gr"
min_length=1
max_length=20
exp_dir="results/length_sampling/100M_samples_eos_zipf_min${min_length}_max${max_length}/true_probs_count_balanced_samples100"


export PYTHONPATH="."
python scripts/calculate_lower_bound_perplexity.py \
    --grammar_file data_gen/${grammar_name} \
    --start_symbol "S" \
    --normalize \
    --min_length $min_length \
    --max_length $max_length \
    --exp_dir $exp_dir \
    --input_suffix ".csv.gz"
