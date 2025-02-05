set -euo pipefail
. experiments/include.bash

data_name="BLLIP_XS"
exp_names=("deterministic_shuffles")
exp_base_dir="$DATA_DIR"/"$data_name"

for exp_name in "${exp_names[@]}"; do
    for trial in $(seq 0 $(($NUM_TRIALS - 1))); do
        for data_dir in "$exp_base_dir"/"$exp_name"/*; do
            grammar_name=$(basename "$data_dir")
            submit_job \
            train_lstm+"$data_name"+"$exp_name"+"$grammar_name"+trial"$trial" \
            gpu \
            --gpus=1 \
            --mem-per-cpu=16g \
            --gres=gpumem:20g \
            --tmp=20g \
            --time=4:00:00 \
            -- \
            bash neural_networks/train_and_evaluate_lstm.sh \
                "$data_dir" \
                "$RESULTS_DIR"/"$data_name"/"$exp_name"/lstm/"$grammar_name"_trial"$trial"
        done
    done
done
