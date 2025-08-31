#!/bin/bash
set -euo pipefail

# Usage:
#   experiments/language_sampling/sample_lstm_random.bash \
#     <output_root_dir> <vocab_size> <num_train> <num_val> <num_test> \
#     <max_length> <parameter_seed> <random_seed>

. experiments/include.bash

output_root_dir="$1"
vocab_size="$2"
num_train="$3"
num_val="$4"
num_test="$5"
max_length="$6"
parameter_seed="$7"
random_seed="$8"

model_tag="lstm_L${parameter_seed}_V${vocab_size}"
out_dir="$output_root_dir/$model_tag"
mkdir -p "$out_dir"/datasets/{validation,test}

# 1) Create dummy vocab file (tokens 0..V-1, allow_unk=True)
python - <<PY
import torch, pathlib
V=$vocab_size
vocab={'tokens':[str(i) for i in range(V)], 'allow_unk': False}
pathlib.Path('$out_dir').mkdir(parents=True, exist_ok=True)
torch.save(vocab, '$out_dir/main.vocab')
PY

# 2) Initialize random LSTM model (prefer local RAU via PYTHONPATH)
PYTHONPATH="$RAU_DIR/src${PYTHONPATH+:$PYTHONPATH}" \
python src/language_sampling/init_random_lm.py \
  --output "$out_dir/model" \
  --vocabulary-file "$out_dir/main.vocab" \
  --parameter-seed "$parameter_seed" \
  --architecture lstm \
  --num-layers 3 \
  --hidden-units 64 \
  --dropout 0.1 \
  --init-scale 0.1

# 3) Sample train/val/test
bash experiments/language_sampling/sample_lm_random.bash \
  "$out_dir" \
  "$out_dir/model" \
  "$num_train" "$num_val" "$num_test" \
  "$max_length" "$random_seed"

echo "[done] LSTM random sampling at $out_dir"

