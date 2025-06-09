#!/bin/bash
set -euo pipefail
. experiments/include.bash

base_grammar_file="config/base-grammar_eos.gr"
exp_name="6switches_deterministic"
num_grammar=64
num_switches=6
include_all_deterministic_grammar=True

python src/pcfg/generate_variations.py \
    --input_file=$base_grammar_file \
    --output_dir="data/PCFG/$exp_name" \
    --num_grammar=$num_grammar \
    --num_switches=$num_switches \
    $(if [ "$include_all_deterministic_grammar" = True ]; then echo "--include_all_deterministic_grammar"; fi)
