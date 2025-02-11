set -euo pipefail
. experiments/include.bash

# data_name="PFSA"
# exp_names=("local_entropy_disjoint" "local_entropy_non_disjoint")
data_name="babylm2024_100K_sents"
exp_names=("deterministic_shuffles")

exp_base_dir="$DATA_DIR"/"$data_name"
split_names=("validation" "test")
ngram_method="kenlm"
for split_name in "${split_names[@]}"; do
    for exp_name in "${exp_names[@]}"; do
        OUTPUT_PATH="${RESULTS_DIR}"/"$data_name"/"$exp_name"/collected_results_"$ngram_method"_"$split_name".csv

        submit_job \
            collect_result+"$data_name"+"$exp_name"+"$ngram_method"+"$split_name" \
            cpu \
            --time=4:00:00 \
            -- \
            python analysis/collect_results.py \
            --data_dir "$exp_base_dir" \
            --result_base_dir "$RESULTS_DIR"/"$data_name" \
            --exp_name "$exp_name" \
            --architectures "${ARCHITECTURES[@]}" \
            --split_name "$split_name" \
            --output_path "$OUTPUT_PATH" \
            --metadata_filename "metadata_"$ngram_method".json"
    done
done
