#!/bin/bash
set -euo pipefail

# Usage:
#   experiments/language_sampling/sample_lm_random.bash \
#     <training_data_dir> <saved_model_dir> \
#     <num_train> <num_val> <num_test> <max_length> <random_seed>
#
# Notes:
# - <training_data_dir> must contain PFSA-like structure with main.vocab
# - <saved_model_dir> is a directory with a trained RAU LM (kwargs.json, parameters.pt)
# - Output directory will mirror PFSA structure and contain:
#     main.tok, main.prepared, main.vocab (copied), metadata.json
#     datasets/{validation,test}/main.tok and .prepared

training_data_dir="$1"
saved_model_dir="$2"
num_train="$3"
num_val="$4"
num_test="$5"
max_length="$6"
random_seed="$7"

. experiments/include.bash

# Resolve output dir name based on model kwargs for traceability
# Write outputs into the provided training_data_dir
out_dir="$training_data_dir"

# Create PFSA-like directory structure
mkdir -p "$out_dir"
mkdir -p "$out_dir"/datasets/validation
mkdir -p "$out_dir"/datasets/test

# Ensure vocabulary file exists at output; otherwise try to copy from training_data_dir
if [ ! -f "$out_dir/main.vocab" ]; then
  if [ -f "$training_data_dir/main.vocab" ]; then
    cp -n "$training_data_dir/main.vocab" "$out_dir/main.vocab"
  elif [ -f "$training_data_dir/../main.vocab" ]; then
    cp -n "$training_data_dir/../main.vocab" "$out_dir/main.vocab"
  else
    echo "[warn] main.vocab not found at output or training_data_dir; expecting caller created it." >&2
  fi
fi

# Helper to generate N random samples in a single call and write to a file
gen_random_n() {
  local N=$1
  local out_file=$2
  PYTHONPATH="$RAU_DIR/src${PYTHONPATH+:$PYTHONPATH}" \
  python -m rau.tasks.language_modeling.generate \
    --load-model "$saved_model_dir" \
    --vocabulary-file "$out_dir/main.vocab" \
    --mode random \
    --max-length "$max_length" \
    --num-samples "$N" \
    --random-seed "$random_seed" \
    --device cpu \
    > "$out_file"
}

echo "[sampling] train: $num_train | val: $num_val | test: $num_test | max_len: $max_length"

# Generate train/val/test .tok files (single call per split)
gen_random_n "$num_train" "$out_dir/main.tok"
gen_random_n "$num_val" "$out_dir/datasets/validation/main.tok"
gen_random_n "$num_test" "$out_dir/datasets/test/main.tok"

# Skipping prepare_data here; separate scripts exist for preparation if needed

# Write minimal metadata for provenance
metadata_file="$out_dir/metadata.json"
if [ -f "$saved_model_dir/kwargs.json" ]; then
  arch=$(python - "$saved_model_dir/kwargs.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
d=json.load(open(p))
print(d.get('architecture','unknown'))
PY
)
else
  arch="unknown"
fi

python - <<PY
import json,sys,os
meta={
  "source_model_dir": os.path.abspath("$saved_model_dir"),
  "training_data": os.path.abspath("$training_data_dir"),
  "num_train": int("$num_train"),
  "num_val": int("$num_val"),
  "num_test": int("$num_test"),
  "max_length": int("$max_length"),
  "num_samples_train": int("$num_train"),
  "num_samples_val": int("$num_val"),
  "num_samples_test": int("$num_test"),
  "initial_random_seed": int("$random_seed"),
  "architecture": "$arch"
}
json.dump(meta, open("$metadata_file","w"), indent=2)
PY

echo "[done] Wrote outputs to $out_dir"


