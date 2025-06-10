#!/bin/bash
set -euo pipefail
# . experiments/include.bash

DATA_DIR=/Users/agiats/Projects/lm_inductive_bias/data
data_name="PCFG"
BASE_DIR="$DATA_DIR"/"$data_name"
exp_name="6switches_deterministic"

for grammar_dir in "$BASE_DIR"/"$exp_name"/*; do
    grammar_name=$(basename "$grammar_dir")
    strings_file="$grammar_dir/strings.txt"
    output_file="$grammar_dir/metadata_predicted.json"

    if [[ ! -f "$strings_file" ]]; then
        echo "Required files not found in $grammar_dir. Skipping..."
        continue
    fi

    python src/predictive_information/estimate_predictive_information.py \
        --input_path "$strings_file" \
        --output_path "$output_file" \
        --min_n 2 \
        --max_n 8 \
        --memory 8G \
        --num-processes 1 \
        --convergence-tol 1e-4 \
        --convergence-window 3 \
        --work-dir "work/${data_name}_${exp_name}_${grammar_name}"
done




