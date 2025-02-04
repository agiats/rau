set -euo pipefail
. experiments/include.bash

exp_base_dir="$DATA_DIR"/PFSA
exp_name="local_entropy_non_disjoint"
for grammar_dir in "$exp_base_dir"/"$exp_name"/*; do
    grammar_name=$(basename "$grammar_dir")
    submit_job \
    train_lstm+"$exp_name"+"$grammar_name" \
    gpu \
    --gpus=1 \
    --mem-per-cpu=16g \
    --gres=gpumem:20g \
    --tmp=20g \
    --time=4:00:00 \
    -- \
    bash neural_networks/train_and_evaluate_lstm.sh \
        "$grammar_dir" \
        "$exp_name" \
        "$grammar_name"
done


exp_base_dir="$DATA_DIR"/PFSA
exp_name="local_entropy_disjoint"
for grammar_dir in "$exp_base_dir"/"$exp_name"/*; do
    grammar_name=$(basename "$grammar_dir")
    submit_job \
    train_lstm+"$exp_name"+"$grammar_name" \
    gpu \
    --gpus=1 \
    --mem-per-cpu=16g \
    --gres=gpumem:20g \
    --tmp=20g \
    --time=4:00:00 \
    -- \
    bash neural_networks/train_and_evaluate_lstm.sh \
        "$grammar_dir" \
        "$exp_name" \
        "$grammar_name"
done
