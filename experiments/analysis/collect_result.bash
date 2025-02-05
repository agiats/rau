set -euo pipefail
. experiments/include.bash

data_name="BLLIP_XS"
exp_names=("deterministic_shuffles")
exp_base_dir="$DATA_DIR"/"$data_name"

for exp_name in "${exp_names[@]}"; do
    OUTPUT_PATH="${RESULTS_DIR}"/"$data_name"/"$exp_name"/collected_results.csv

    submit_job \
    collect_results+"$exp_name" \
    cpu \
    --time=4:00:00 \
    -- \
    python analysis/collect_results.py \
        --data_dir "$exp_base_dir" \
        --result_dir "$RESULTS_DIR" \
        --exp_name "$exp_name" \
        --architectures "${ARCHITECTURES[@]}" \
        --output_path "$OUTPUT_PATH"
done
