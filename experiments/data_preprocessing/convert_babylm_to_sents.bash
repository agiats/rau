set -euo pipefail
. experiments/include.bash

raw_dir="$DATA_DIR"/babylm_raw
dst_dir="$DATA_DIR"/babylm/deterministic_shuffles/Base

declare -a src_dst_pair=(
    "babylm_10M train.txt"
    "babylm_dev dev.txt"
    "babylm_test test.txt"
)

for pair in "${src_dst_pair[@]}"; do
    src_name=$(echo $pair | awk '{print $1}')
    dst_name=$(echo $pair | awk '{print $2}')

    src_dir="$raw_dir"/"$src_name"
    dst_path="$dst_dir"/"$dst_name"

    submit_job \
    convert_babylm_to_sents+"$src_name" \
    cpu \
    --mem-per-cpu=32g \
    --time=4:00:00 \
    -- \
    python data_preprocessing/convert_babylm_to_sents.py \
    --input_dir "$src_dir" \
    --output_path "$dst_path" \
    --min-length 2
done
