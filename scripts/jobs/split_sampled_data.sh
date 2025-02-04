#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=local_shuffle
#SBATCH --output=logs/local_shuffle.out
#SBATCH --error=logs/local_shuffle.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."


# n_samples=1_000_000
# n_processes=100
# min_length=1
# max_length=20
# grammar_dir="data/grammars/variations/6switches_3values"
# input_dir="data/variations_wc/6switches_3values_min${min_length}_max${max_length}_1M"

# for grammar_file in ${grammar_dir}/*.gr; do
#     grammar_name=$(basename ${grammar_file} .gr)
#     input_file="${input_dir}/${grammar_name}/samples.txt.gz"
#     output_dir="data/fairseq_train/6switches_3values_min${min_length}_max${max_length}_1M/${grammar_name}"
#     echo "Splitting ${input_file} to ${output_dir}"
#     python scripts/split_sampled_data.py \
#         --input_file $input_file \
#         --output_dir $output_dir
# done



# for input_dir in results/length_sampling_local_shuffle/*; do
#     input_file="${input_dir}/samples.txt.gz"
#     output_dir="data/fairseq_train/length_sampling_local_shuffle/$(basename ${input_dir})"
#     echo "Splitting ${input_file} to ${output_dir}"
#     python scripts/split_sampled_data.py \
#         --input_file $input_file \
#         --output_dir $output_dir
# done


for input_dir in results/ptb_local_shuffle/*; do
    seed=$(basename "$input_dir" | grep -oP 'seed\K\d+')

    if [ -n "$seed" ] ; then
        input_file="${input_dir}/samples.txt.gz"
        dir_name=$(basename "$input_dir")
        grammar_name=${dir_name#1M_samples_eos_min1_max20_}
        output_dir="data/fairseq_train/ptb_local_shuffle/$grammar_name"
        echo "Splitting ${input_file} to ${output_dir}"
        python scripts/split_sampled_data.py \
            --input_file "$input_file" \
            --output_dir "$output_dir"
    fi
done


python scripts/split_sampled_data.py \
    --input_file /cluster/home/tsomeya/projects/impossible_inherent_entropy/data/ptb/ptb.all.txt.gz \
    --output_dir data/fairseq_train/ptb_local_shuffle/Base
