#!/bin/bash
set -euo pipefail
. experiments/include.bash

data_name="babylm2024_10M"
BASE_DIR="$DATA_DIR"/"$data_name"
exp_name="deterministic_shuffles"

for grammar_dir in "$BASE_DIR"/"$exp_name"/*; do
    grammar_name=$(basename "$grammar_dir")

    train_file="$grammar_dir/main.tok"
    valid_file="$grammar_dir/datasets/validation/main.tok"
    test_file="$grammar_dir/datasets/test/main.tok"
    output_file="$grammar_dir/metadata.json"

    if [[ ! -f "$train_file" || ! -f "$valid_file" || ! -f "$test_file" ]]; then
        echo "Required files not found in $grammar_dir. Skipping..."
        continue
    fi

    submit_job \
        ngram_entropy+"$data_name"+"$exp_name"+"$grammar_name" \
        cpu \
        --time=8:00:00 \
        --mem-per-cpu=64g \
        -- \
        python local_entropy/estimate_local_entropy.py \
            --train_path "$train_file" \
            --valid_path "$valid_file" \
            --test_path "$test_file" \
            --output_path "$output_file" \
            --n 2 3 4 5 \
            --gamma 0
done
