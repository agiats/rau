set -euo pipefail
. experiments/include.bash

exp_name="local_entropy_disjoint"
for grammar_dir in "$DATA_DIR"/PFSA/"$exp_name"/*; do
    grammar_name=$(basename "$grammar_dir")
    if [ ! -f "$grammar_dir"/main.tok ]; then
        mkdir -p "$grammar_dir"/datasets/validation
        mkdir -p "$grammar_dir"/datasets/test
        mv "$grammar_dir"/train.txt "$grammar_dir"/main.tok
        mv "$grammar_dir"/val.txt "$grammar_dir"/datasets/validation/main.tok
        mv "$grammar_dir"/test.txt "$grammar_dir"/datasets/test/main.tok
    fi

    submit_job \
    prepare_data+PFSA_"$exp_name"+"$grammar_name" \
    cpu \
    --time=4:00:00 \
    -- \
    python "$RAU_DIR"/src/rau/tasks/language_modeling/prepare_data.py \
        --training-data "$grammar_dir" \
        --more-data validation \
        --more-data test \
        --never-allow-unk
done
