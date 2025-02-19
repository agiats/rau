set -euo pipefail
. experiments/include.bash

src_dir="$DATA_DIR"/babylm2024_100M/deterministic_shuffles/DeterministicShuffle_seed1
dst_dir="$DATA_DIR"/babylm2024_100M_longer_than_10/deterministic_shuffles/DeterministicShuffle_seed1
min_tokens=10

# targets=("train.txt" "dev.txt" "test.txt")
targets=("main.tok" "datasets/validation/main.tok" "datasets/test/main.tok")


for target in "${targets[@]}"; do
    src_path="$src_dir"/"$target"
    dst_path="$dst_dir"/"$target"

    mkdir -p "$(dirname "$dst_path")"
    echo "Processing $target..."
    echo "dst_path: $dst_path"

    awk -v min="$min_tokens" 'NF>=min' "$src_path" > "$dst_path"

    total=$(wc -l < "$src_path")
    filtered=$(wc -l < "$dst_path")
    echo "Total lines: $total"
    echo "Lines after filtering: $filtered"
    echo "Removed lines: $((total - filtered))"
    echo "---"
done
