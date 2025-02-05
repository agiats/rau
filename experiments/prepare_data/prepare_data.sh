set -euo pipefail
. experiments/include.bash

BASE_DIR="$DATA_DIR"/BLLIP_XS
exp_name="deterministic_shuffles"
for grammar_dir in "$BASE_DIR"/"$exp_name"/*; do
    grammar_name=$(basename "$grammar_dir")
    mkdir -p "$grammar_dir"/datasets/validation
    mkdir -p "$grammar_dir"/datasets/test
    if [ ! -f "$grammar_dir"/main.tok ]; then
        mv "$grammar_dir"/train.txt "$grammar_dir"/main.tok
    fi
    if [ ! -f "$grammar_dir"/datasets/validation/main.tok ]; then
        mv "$grammar_dir"/dev.txt "$grammar_dir"/datasets/validation/main.tok
    fi
    if [ ! -f "$grammar_dir"/datasets/test/main.tok ]; then
        mv "$grammar_dir"/test.txt "$grammar_dir"/datasets/test/main.tok
    fi

    submit_job \
    prepare_data+"$(basename "$BASE_DIR")"+"$exp_name"+"$grammar_name" \
    cpu \
    --time=4:00:00 \
    -- \
    python "$RAU_DIR"/src/rau/tasks/language_modeling/prepare_data.py \
        --training-data "$grammar_dir" \
        --more-data validation \
        --more-data test \
        --always-allow-unk
done
