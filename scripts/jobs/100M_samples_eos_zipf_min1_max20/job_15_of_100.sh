#!/bin/bash

#SBATCH --ntasks=128
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=100M_samples_eos_zipf_min1_max20_count_balanced_samples100_split_15_of_100
#SBATCH --output=logs/100M_samples_eos_zipf_min1_max20/count_balanced_samples100_split_15_of_100.out
#SBATCH --error=logs/100M_samples_eos_zipf_min1_max20/count_balanced_samples100_split_15_of_100.err

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

grammar_name="base-grammar_eos_zipf.gr"
n_processes=128

python scripts/calculate_true_prob.py \
    --grammar_file data_gen/$grammar_name \
    --start_symbol "S" \
    --normalize \
    --sample_path "results/length_sampling/100M_samples_eos_zipf_min1_max20/balanced_samples100.txt.gz" \
    --output_path "results/length_sampling/100M_samples_eos_zipf_min1_max20/true_probs_count_balanced_samples100/probability_split_15_of_100.json.gz" \
    --num_workers $n_processes \
    --start_index 3710 \
    --end_index 3975 \
    --sample_size 26442
