set -euo pipefail
. experiments/include.bash


data_name="babylm2024_10M_fixed"
exp_names=("deterministic_shuffles")
exp_base_dir="$DATA_DIR"/"$data_name"
split_names=("validation" "test")
ngram_method="kenlm"
for split_name in "${split_names[@]}"; do
    for exp_name in "${exp_names[@]}"; do
        RESULTS_PATH="${RESULTS_DIR}"/"$data_name"/"$exp_name"/collected_results_"$ngram_method"_"$split_name".csv
        PLOTS_DIR="${RESULTS_DIR}"/"$data_name"/"$exp_name"/plots_"$ngram_method"

        submit_job \
        plot_results_natural_language+"$data_name"+"$exp_name"+"$ngram_method"+"$split_name" \
        cpu \
        --time=4:00:00 \
        -- \
        python analysis/plot_results_natural_language.py \
            --results_path "$RESULTS_PATH" \
            --output_dir "$PLOTS_DIR" \
            --architectures "${ARCHITECTURES[@]}" \
            --architecture_labels "${ARCHITECTURE_LABELS[@]}" \
            --split_name "$split_name"
    done
done
