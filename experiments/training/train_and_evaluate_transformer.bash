#!/bin/bash
#SBATCH --job-name=submit_jobs_transformer
#SBATCH --output=/cluster/home/tsomeya/projects/lm_inductive_bias/results/submit_jobs_transformer.out
#SBATCH --error=/cluster/home/tsomeya/projects/lm_inductive_bias/results/submit_jobs_transformer.err
#SBATCH --time=04:00:00  # Adjust time as needed
#SBATCH --mem-per-cpu=4G         # Adjust memory as needed
#SBATCH --tmp=20g
#SBATCH --cpus-per-task=1  # Adjust CPU as needed


set -euo pipefail
. experiments/include.bash

data_name="PFSA"
exp_names=("local_entropy_non_disjoint_larger_m")
# data_name="babylm2024_10M_fixed"
# exp_names=("deterministic_shuffles")
exp_base_dir="$DATA_DIR"/"$data_name"
examples_per_checkpoint=10000
max_tokens_per_batch=2048
time_limit=04:00:00
# gpu_mem=10g

for exp_name in "${exp_names[@]}"; do
    for trial in $(seq 0 $(($NUM_TRIALS - 1))); do
        for data_dir in "$exp_base_dir"/"$exp_name"/*; do
            grammar_name=$(basename "$data_dir")
            if [ -f "$RESULTS_DIR"/"$data_name"/"$exp_name"/transformer/"$grammar_name"_trial"$trial"/evaluation/validation.json ]; then
                continue
            fi
            # echo "$RESULTS_DIR"/"$data_name"/"$exp_name"/transformer/"$grammar_name"_trial"$trial"
            # rm -rf "$RESULTS_DIR"/"$data_name"/"$exp_name"/transformer/"$grammar_name"_trial"$trial"

            submit_job \
            train_transformer+"$data_name"+"$exp_name"+"$grammar_name"+trial"$trial" \
            gpu \
            --gpus=1 \
            --mem-per-cpu=16g \
            --tmp=20g \
            --time="$time_limit" \
            -- \
            bash neural_networks/train_and_evaluate_transformer.sh \
                "$data_dir" \
                "$RESULTS_DIR"/"$data_name"/"$exp_name"/transformer/"$grammar_name"_trial"$trial" \
                "$examples_per_checkpoint" \
                "$max_tokens_per_batch"
        done
    done
done

