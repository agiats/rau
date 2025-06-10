#!/bin/bash
set -euo pipefail
. experiments/include.bash

exp_name="6switches_deterministic"
exp_dir="data/PCFG/$exp_name"
start_symbol="S"
normalize=True
seed=42
min_length=1
max_length=40
num_samples=300000
num_workers=8

for grammar_dir in $exp_dir/*; do
    grammar_file="$grammar_dir/grammar.gr"

    # submit_job \
    #     pcfg_generate_strings+"$exp_name"+"$grammar_dir" \
    #     cpu \
    #     --time=48:00:00 \
    #     --tasks $num_workers \
    #     -- \
        python src/length_sampling/sample_strings.py \
            --grammar_file=$grammar_file \
            --start_symbol=$start_symbol \
            $(if [ "$normalize" = True ]; then echo "--normalize"; fi) \
            --seed=$seed \
            --min_length=$min_length \
            --max_length=$max_length \
            --num_samples=$num_samples \
            --num_workers=$num_workers \
            --output_path=$grammar_dir/strings.txt
done
