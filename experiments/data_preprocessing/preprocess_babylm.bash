set -euo pipefail
. experiments/include.bash

raw_dir="$DATA_DIR"/babylm2024_raw
dst_dir="$DATA_DIR"/babylm2024_100M/deterministic_shuffles/Base
n_jobs=6
mem_per_cpu=64g

declare -a src_dst_pair=(
    "train_100M train.txt"
    # "train_10M train.txt"
    # "dev dev.txt"
    # "test test.txt"
    # "small small.txt"
)

for pair in "${src_dst_pair[@]}"; do
    src_name=$(echo $pair | awk '{print $1}')
    dst_name=$(echo $pair | awk '{print $2}')

    src_dir="$raw_dir"/"$src_name"
    dst_path="$dst_dir"/"$dst_name"

    submit_job \
    preprocess_babylm+"$(basename "$raw_dir")"+"$src_name" \
    cpu \
    --tasks="$n_jobs" \
    --mem-per-cpu="$mem_per_cpu" \
    --time=4:00:00 \
    -- \
    python data_preprocessing/preprocess_babylm.py \
    --input_dir "$src_dir" \
    --output_path "$dst_path" \
    --min-length 2 \
    --spacy-model en_core_web_lg \
    --n-jobs "$n_jobs"
done
