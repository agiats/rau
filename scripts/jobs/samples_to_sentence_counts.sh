#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16g
#SBATCH --tmp=32g
#SBATCH --job-name=samples_to_sentence_counts_variations_wc_10K
#SBATCH --output=logs/samples_to_sentence_counts_variations_wc_10K.out
#SBATCH --error=logs/samples_to_sentence_counts_variations_wc_10K.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."


for input_dir in data/variations_wc/6switches_3values_min1_max20_10K/*; do
    input_file="${input_dir}/samples.txt.gz"
    output_file="${input_dir}/sample_counts.csv.gz"
    python scripts/samples_to_sentence_counts.py \
        --input_file "$input_file" \
        --output_file "$output_file"
done
