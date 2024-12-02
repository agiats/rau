#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16g
#SBATCH --tmp=32g
#SBATCH --job-name=calculate_entropy_of_samples_10M
#SBATCH --output=logs/calculate_entropy_of_samples_10M.out
#SBATCH --error=logs/calculate_entropy_of_samples_10M.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."

input_dirs=(
    "results/length_sampling/10M_samples_eos_zipf_min1_max20"
    "results/length_sampling/10M_samples_eos_zipf_min1_max20_DeterministicShuffle"
    "results/length_sampling/10M_samples_eos_zipf_min1_max20_EvenOddShuffle"
    "results/length_sampling/10M_samples_eos_zipf_min1_max20_NonDeterministicShuffle"
    "results/length_sampling/10M_samples_eos_zipf_min1_max20_LocalShuffle"
    "results/length_sampling/10M_samples_eos_zipf_min1_max20_NoReverse"
    "results/length_sampling/10M_samples_eos_zipf_min1_max20_PartialReverse"
    "results/length_sampling/10M_samples_eos_zipf_min1_max20_FullReverse"
)

# for input_dir in "${input_dirs[@]}"; do
#     input_file="${input_dir}/balanced_sample_counts.csv.gz"
#     python scripts/calculate_entropy_of_samples.py \
#         --input_file "$input_file" \
#         --output_dir "$input_dir" \
#         --sample_size 10_000_000 \
#         --output_suffix "_balanced_samples100"
# done


for input_dir in "${input_dirs[@]}"; do
    input_file="${input_dir}/sample_counts.csv.gz"
    python scripts/calculate_entropy_of_samples.py \
        --input_file "$input_file" \
        --output_dir "$input_dir" \
        --sample_size 10_000_000 \
        --output_suffix "_all_samples"
done




