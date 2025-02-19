set -euo pipefail
. experiments/include.bash

data_dir="$DATA_DIR"/babylm2024_raw/test
mem_per_cpu=32g


for corpus_path in "$data_dir"/*; do
    corpus_name=$(basename "$corpus_path")
    dst_path="$data_dir"/"$corpus_name".txt

    submit_job \
        preprocess_babylm+"$corpus_name" \
        cpu \
        --mem-per-cpu="$mem_per_cpu" \
        --time=48:00:00 \
        -- \
        python data_preprocessing/preprocess_babylm_fix.py \
        --input_file "$corpus_path" \
        --output_path "$dst_path" \
        --min-length 2 \
        --spacy-model en_core_web_lg \
        --batch-size 10000
done


# merge files
# dst_path="$DATA_DIR"/babylm2024_100M/deterministic_shuffles/Base/train.txt
# cat "$data_dir"/*.txt > "$dst_path"
