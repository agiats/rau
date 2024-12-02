#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16g
#SBATCH --tmp=32g
#SBATCH --job-name=subsample_sentences
#SBATCH --output=logs/subsample_sentences.out
#SBATCH --error=logs/subsample_sentences.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."

input_dirs=(
    "results/length_sampling/100M_samples_eos_zipf_min1_max20"
    "results/length_sampling/100M_samples_eos_zipf_min1_max20_DeterministicShuffle"
    "results/length_sampling/100M_samples_eos_zipf_min1_max20_EvenOddShuffle"
    "results/length_sampling/100M_samples_eos_zipf_min1_max20_NonDeterministicShuffle"
    "results/length_sampling/100M_samples_eos_zipf_min1_max20_LocalShuffle"
    "results/length_sampling/100M_samples_eos_zipf_min1_max20_NoReverse"
    "results/length_sampling/100M_samples_eos_zipf_min1_max20_PartialReverse"
    "results/length_sampling/100M_samples_eos_zipf_min1_max20_FullReverse"
)

for input_dir in "${input_dirs[@]}"; do
    input_file="${input_dir}/samples.txt.gz"
    output_dir="${input_dir/100M/100K_dedup}"
    output_file="${output_dir}/samples.txt.gz"
    mkdir -p "$output_dir"
    python scripts/subsample_sentences.py \
        --input_file "$input_file" \
        --output_file "$output_file" \
        --sample_size 100_000
        --deduplicate
done
