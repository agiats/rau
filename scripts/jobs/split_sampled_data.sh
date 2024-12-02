#!/bin/bash

#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4g
#SBATCH --tmp=32g
#SBATCH --job-name=split_sampled_data_100K_dedup
#SBATCH --output=logs/split_sampled_data_100K_dedup.out
#SBATCH --error=logs/split_sampled_data_100K_dedup.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."


grammar_names=(
    "DeterministicShuffle"
    "EvenOddShuffle"
    "NonDeterministicShuffle"
    "LocalShuffle"
    "NoReverse"
    "PartialReverse"
    "FullReverse"
)
for grammar_name in "${grammar_names[@]}"; do
    python scripts/split_sampled_data.py \
        --input_file results/length_sampling/100K_dedup_samples_eos_zipf_min1_max20_${grammar_name}/samples.txt.gz \
        --output_dir data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/${grammar_name}
done

python scripts/split_sampled_data.py \
    --input_file results/length_sampling/100K_dedup_samples_eos_zipf_min1_max20/samples.txt.gz \
    --output_dir data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/Base
