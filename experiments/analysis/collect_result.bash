set -euo pipefail
. experiments/include.bash

# data_name="PFSA"
# exp_names=("local_entropy_disjoint" "local_entropy_non_disjoint")
data_name="BLLIP_XS"
exp_names=("deterministic_shuffles")

exp_base_dir="$DATA_DIR"/"$data_name"
split_names=("test")
for split_name in "${split_names[@]}"; do
    for exp_name in "${exp_names[@]}"; do
        OUTPUT_PATH="${RESULTS_DIR}"/"$data_name"/"$exp_name"/collected_results_"$split_name".csv

        submit_job \
            collect_result+"$data_name"+"$exp_name" \
            cpu \
            --time=4:00:00 \
            -- \
            python analysis/collect_results.py \
            --data_dir "$exp_base_dir" \
            --result_base_dir "$RESULTS_DIR"/"$data_name" \
            --exp_name "$exp_name" \
            --architectures "${ARCHITECTURES[@]}" \
            --split_name "$split_name" \
            --output_path "$OUTPUT_PATH"
    done
done
