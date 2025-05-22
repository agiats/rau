set -euo pipefail
. experiments/include.bash


data_name="PFSA"
exp_names=("local_entropy_XXX")
exp_base_dir="$DATA_DIR"/"$data_name"
split_names=("test" "validation")
for split_name in "${split_names[@]}"; do
    for exp_name in "${exp_names[@]}"; do
        RESULTS_PATH="${RESULTS_DIR}"/"$data_name"/"$exp_name"/collected_results_"$split_name".csv
        PLOTS_DIR="${RESULTS_DIR}"/"$data_name"/"$exp_name"/plots

        submit_job \
        plot_results_pfsa+"$data_name"+"$exp_name"+"$split_name" \
        cpu \
        --time=4:00:00 \
        -- \
        python analysis/plot_results_pfsa.py \
        --results_path "$RESULTS_PATH" \
        --output_dir "$PLOTS_DIR" \
        --architectures "${ARCHITECTURES[@]}" \
        --architecture_labels "${ARCHITECTURE_LABELS[@]}" \
        --split_name "$split_name"
    done
done

