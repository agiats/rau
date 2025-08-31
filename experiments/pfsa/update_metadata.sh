#!/bin/bash
set -euo pipefail
. experiments/include.bash

BASE_DIR=data/PFSA/local_entropy_non_disjoint_final

for grammar_dir in "$BASE_DIR"/*; do
    if [[ ! -d "$grammar_dir" ]]; then
        continue
    fi

    grammar_name=$(basename "$grammar_dir")
    echo "Processing $grammar_name..."
    # submit_job \
    #     pfsa_update_metadata+"$grammar_name" \
    #     cpu \
    #     --time=1:00:00 \
    #     --mem-per-cpu=4g \
    #     --cpus-per-task=1 \
    #     -- \
    python src/pfsa/update_metadata.py \
        --model_path "$grammar_dir/model.pickle" \
        --metadata_path "$grammar_dir/metadata.json"
done
