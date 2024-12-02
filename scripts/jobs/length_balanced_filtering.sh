#!/bin/bash

#SBATCH --ntasks=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16g
#SBATCH --tmp=32g
#SBATCH --job-name=logs/length_balanced_filtering
#SBATCH --output=logs/length_balanced_filtering.out
#SBATCH --error=logs/length_balanced_filtering.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."


# output_name="100M_samples_eos_zipf_min1_max20"

# python scripts/length_balanced_filtering.py \
#     --input_path "results/length_sampling/${output_name}/sample_counts.json.gz" \
#     --target-samples-per-count 100 \
#     --output-path "results/length_sampling/${output_name}/balanced_sample_counts.csv.gz"


output_names=(
    "100M_samples_eos_zipf_min1_max20_DeterministicShuffle"
    "100M_samples_eos_zipf_min1_max20_EvenOddShuffle"
    "100M_samples_eos_zipf_min1_max20_NonDeterministicShuffle"
    "100M_samples_eos_zipf_min1_max20_LocalShuffle"
    "100M_samples_eos_zipf_min1_max20_NoReverse"
    "100M_samples_eos_zipf_min1_max20_PartialReverse"
    "100M_samples_eos_zipf_min1_max20_FullReverse"
)
for output_name in "${output_names[@]}"; do
    python scripts/length_balanced_filtering.py \
        --input_path "results/length_sampling/${output_name}/sample_counts.csv.gz" \
        --target-samples-per-count 100 \
        --output-path "results/length_sampling/${output_name}/balanced_sample_counts.csv.gz"
done
