set -euo pipefail
. /cluster/home/tsomeya/projects/lm_inductive_bias/experiments/include.bash

# Get arguments
grammar_dir="$1"
exp_name="$2"
grammar_name="$3"
trial="$4"

python "$RAU_DIR"/src/rau/tasks/language_modeling/train.py \
    --training-data "$grammar_dir" \
    --architecture transformer \
    --num-layers 4 \
    --d-model 768 \
    --num-heads 12 \
    --feedforward-size 3072 \
    --dropout 0.1 \
    --init-scale 0.1 \
    --max-epochs 1000 \
    --max-tokens-per-batch 2048 \
    --optimizer Adam \
    --initial-learning-rate 0.0005 \
    --gradient-clipping-threshold 5 \
    --early-stopping-patience 10 \
    --learning-rate-patience 5 \
    --learning-rate-decay-factor 0.5 \
    --examples-per-checkpoint 10000 \
    --output "$RESULTS_DIR"/"$exp_name"/transformer/"$grammar_name"_trial"$trial"

eval_dir="$RESULTS_DIR"/"$exp_name"/transformer/"$grammar_name"_trial"$trial"/evaluation
mkdir -p "$eval_dir"
python "$RAU_DIR"/src/rau/tasks/language_modeling/evaluate.py \
    --load-model "$RESULTS_DIR"/"$exp_name"/transformer/"$grammar_name"_trial"$trial" \
    --training-data "$grammar_dir" \
    --input test \
    --batching-max-tokens 2048 > "$eval_dir"/test.json
