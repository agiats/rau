set -euo pipefail
. experiments/include.bash

for EXP_NAME in "${EXP_NAMES[@]}"; do
    RESULTS_PATH="${RESULTS_DIR}/${EXP_NAME}/collected_results.csv"
    PLOTS_DIR="${RESULTS_DIR}/${EXP_NAME}/plots"

    submit_job \
    plot_results+"$EXP_NAME" \
    cpu \
    --time=4:00:00 \
    -- \
    python analysis/plot_results_pfsa.py \
    --results_path "$RESULTS_PATH" \
    --output_dir "$PLOTS_DIR" \
    --architectures "${ARCHITECTURES[@]}" \
    --architecture_labels "${ARCHITECTURE_LABELS[@]}"
done
