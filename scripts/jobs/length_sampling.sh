#!/bin/bash

#SBATCH --ntasks=128
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=sample
#SBATCH --output=logs/sample.out
#SBATCH --error=logs/sample.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."

n_samples=1_000_000
n_processes=100
min_length=1
max_length=20
grammar_dir="data/grammars/variations/6switches_3values_zipf"
output_dir="data/variations_wc/6switches_3values_zipf_min${min_length}_max${max_length}_1M"
mkdir -p $output_dir

for grammar_file in ${grammar_dir}/*.gr; do
    grammar_name=$(basename ${grammar_file} .gr)
    mkdir -p "${output_dir}/${grammar_name}"
    echo "Sampling from ${grammar_name}"
    python scripts/length_sampling.py \
        --grammar_file $grammar_file \
        --start_symbol "S" \
        --normalize \
        --seed 42 \
        --min_length $min_length \
        --max_length $max_length \
        --num_samples $n_samples \
        --output_path "${output_dir}/${grammar_name}/samples.txt.gz" \
        --num_workers $n_processes \
        --output_sent
done

for grammar_file in ${grammar_dir}/*.gr; do
    grammar_name=$(basename ${grammar_file} .gr)
    input_file="${output_dir}/${grammar_name}/samples.txt.gz"
    output_dir_2="data/fairseq_train/6switches_3values_zipf_min${min_length}_max${max_length}_1M/${grammar_name}"
    echo "Splitting ${input_file} to ${output_dir_2}"
    python scripts/split_sampled_data.py \
        --input_file $input_file \
        --output_dir $output_dir_2
done


# n_samples=10_000_000
# n_processes=100
# min_length=1
# max_length=20
# grammar_file="data/grammars/base-grammar.gr"
# output_dir="results/length_sampling/10M_samples_eos_min${min_length}_max${max_length}"
# mkdir -p $output_dir

# python scripts/length_sampling.py \
#     --grammar_file $grammar_file \
#     --start_symbol "S" \
#     --normalize \
#     --seed 42 \
#     --min_length $min_length \
#     --max_length $max_length \
#     --num_samples $n_samples \
#     --output_path "${output_dir}/samples.txt.gz" \
#     --num_workers $n_processes \
#     --output_sent
