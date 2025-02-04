#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=local_shuffle2
#SBATCH --output=logs/local_shuffle2.out
#SBATCH --error=logs/local_shuffle2.err
#SBATCH --mail-type=FAIL

module load stack/2024-05  gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH="."


# for input_dir in results/length_sampling_local_shuffle/*; do
#     input_file="${input_dir}/samples.txt.gz"
#     output_dir="data/fairseq_train/length_sampling_local_shuffle/$(basename ${input_dir})"
#     echo "Splitting ${input_file} to ${output_dir}"
#     python scripts/split_sampled_data.py \
#         --input_file $input_file \
#         --output_dir $output_dir
# done


for input_dir in results/local_shuffle_10M_alpha07/*; do
    seed=$(basename "$input_dir" | grep -oP 'seed\K\d+')

    # seedが3以下の場合のみ処理を実行
    if [ -n "$seed" ] && [ "$seed" -le 5 ]; then
        input_file="${input_dir}/samples.txt.gz"
        dir_name=$(basename "$input_dir")
        grammar_name=${dir_name#1M_samples_eos_min1_max20_}
        output_dir="data/fairseq_train/local_shuffle_10M_alpha07/$grammar_name"
        echo "Splitting ${input_file} to ${output_dir}"
        python scripts/split_sampled_data.py \
            --input_file "$input_file" \
            --output_dir "$output_dir"
    fi
done


python scripts/split_sampled_data.py \
    --input_file /cluster/home/tsomeya/projects/impossible_inherent_entropy/results/local_shuffle_10M_alpha07/local_shuffle_10M_alpha07/samples.txt.gz \
    --output_dir data/fairseq_train/local_shuffle_10M_alpha07/Base
