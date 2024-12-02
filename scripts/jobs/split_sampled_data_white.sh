#!/bin/bash

#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4g
#SBATCH --tmp=32g
#SBATCH --job-name=split_sampled_data_white
#SBATCH --output=logs/split_sampled_data_white.out
#SBATCH --error=logs/split_sampled_data_white.err
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
    python scripts/split_sampled_data_white.py \
        --input_file results/length_sampling/100M_samples_eos_zipf_min1_max20_${grammar_name}/samples.txt.gz \
        --output_dir data/fairseq_train/eos_zipf_min1_max_20/${grammar_name} \
        --num_splits 10 \
        --num_samples_per_split 10000
done

python scripts/split_sampled_data_white.py \
    --input_file results/length_sampling/100M_samples_eos_zipf_min1_max20/samples.txt.gz \
    --output_dir data/fairseq_train/eos_zipf_min1_max_20/Base \
    --num_splits 10 \
    --num_samples_per_split 10000
