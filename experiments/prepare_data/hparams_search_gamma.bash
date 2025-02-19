#!/bin/bash
set -euo pipefail
. experiments/include.bash

data_name="BLLIP_SM"
BASE_DIR="$DATA_DIR"/"$data_name"
exp_name="deterministic_shuffles"
target_dir="$BASE_DIR"/"$exp_name"/Base
RESULTS_DIR="$target_dir/gamma_analysis"
method="Lidstone"
gammas=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1)

mkdir -p "$RESULTS_DIR"

# for gamma in "${gammas[@]}"; do

#     train_file="$target_dir/main.tok"
#     valid_file="$target_dir/datasets/validation/main.tok"
#     test_file="$target_dir/datasets/test/main.tok"
#     output_file="$RESULTS_DIR/metadata_gamma_$gamma.json"

#     if [[ ! -f "$train_file" || ! -f "$valid_file" || ! -f "$test_file" ]]; then
#         echo "Required files not found in $target_dir. Skipping..."
#         continue
#     fi
#     submit_job \
#         gamma_hparams_search+"$data_name"+"$exp_name"+"$(basename "$target_dir")"+"$gamma" \
#         cpu \
#         --time=24:00:00 \
#         --mem-per-cpu=32g \
#         -- \
#         python local_entropy/estimate_local_entropy.py \
#             --train_path "$train_file" \
#             --valid_path "$valid_file" \
#             --test_path "$test_file" \
#             --output_path "$output_file" \
#             --n 2 3 4 5  \
#             --method "$method" \
#             --gamma "$gamma"
# done



python src/analysis/gamma_hparams_search.py \
    --data_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR"


