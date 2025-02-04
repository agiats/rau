#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --output=logs/6switches_3values_min1_max20_10K/020020/split_1_of_1.out
#SBATCH --error=logs/6switches_3values_min1_max20_10K/020020/split_1_of_1.err

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

n_processes=64

if [ ! -f "data/variations_wc/6switches_3values_min1_max20_10K/020020/true_prob/lower_bound_entropy.value" ]; then
    python scripts/calculate_true_prob.py \
        --grammar_file data/grammars/variations/6switches_3values/020020.gr \
        --start_symbol "S" \
        --normalize \
        --sentence_counts_path data/variations_wc/6switches_3values_min1_max20_10K/020020/sample_counts.csv.gz \
        --output_path data/variations_wc/6switches_3values_min1_max20_10K/020020/true_prob/probability_split_1_of_1.csv.gz \
        --num_workers $n_processes \
        --start_index 0 \
        --end_index 10000
fi
