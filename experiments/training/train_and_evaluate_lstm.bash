set -euo pipefail
. experiments/include.bash

data_name="babylm2024_100K"
# exp_names=("local_entropy_disjoint" "local_entropy_non_disjoint")
exp_names=("deterministic_shuffles")
exp_base_dir="$DATA_DIR"/"$data_name"
examples_per_checkpoint=10000

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
                "$RESULTS_DIR"/"$data_name"/"$exp_name"/lstm/"$grammar_name"_trial"$trial" \
                "$examples_per_checkpoint"
        done
    done
done
