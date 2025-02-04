set -euo pipefail
. experiments/include.bash

exp_base_dir="$DATA_DIR"/PFSA

for exp_name in "${EXP_NAMES[@]}"; do
    for trial in $(seq 0 $(($NUM_TRIALS - 1))); do
        for grammar_dir in "$exp_base_dir"/"$exp_name"/*; do
            grammar_name=$(basename "$grammar_dir")
            submit_job \
            train_transformer+"$exp_name"+"$grammar_name"+trial"$trial" \
            gpu \
            --gpus=1 \
            --mem-per-cpu=16g \
            --gres=gpumem:20g \
            --tmp=20g \
            --time=4:00:00 \
            -- \
            bash neural_networks/train_and_evaluate_transformer.sh \
                "$grammar_dir" \
                "$exp_name" \
                "$grammar_name" \
                "$trial"
        done
    done
done
