set -euo pipefail
. /cluster/home/tsomeya/projects/lm_inductive_bias/experiments/include.bash

# Get arguments
data_dir="$1"
output_dir="$2"

python "$RAU_DIR"/src/rau/tasks/language_modeling/train.py \
    --training-data "$data_dir" \
    --architecture lstm \
    --num-layers 1 \
    --hidden-units 512 \
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
    --output "$output_dir"

eval_dir="$output_dir"/evaluation
mkdir -p "$eval_dir"
python "$RAU_DIR"/src/rau/tasks/language_modeling/evaluate.py \
    --load-model "$output_dir" \
    --training-data "$data_dir" \
    --input test \
    --batching-max-tokens 2048 > "$eval_dir"/test.json
