set -euo pipefail
. experiments/include.bash

for EXP_NAME in "${EXP_NAMES[@]}"; do
    OUTPUT_PATH="${RESULTS_DIR}/${EXP_NAME}/collected_results.csv"

    mkdir -p "$(dirname "$OUTPUT_PATH")"

    submit_job \
    collect_results_pfsa+"$EXP_NAME" \
    cpu \
    --time=4:00:00 \
    -- \
    python analysis/collect_results_pfsa.py \
        --data_dir "${DATA_DIR}/PFSA" \
        --result_dir "$RESULTS_DIR" \
        --exp_name "$EXP_NAME" \
        --architectures "${ARCHITECTURES[@]}" \
        --output_path "$OUTPUT_PATH"
done
