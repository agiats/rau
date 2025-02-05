#!/bin/bash

DATA_DIR="data/fairseq_train/BLLIP_XS"
WORK_DIR="work"

WINDOW_SIZE=$1

if [ -z "$WINDOW_SIZE" ]; then
    echo "Usage: $0 WINDOW_SIZE"
    exit 1
fi

mkdir -p "$WORK_DIR"

find "$DATA_DIR" -type d -name "*LocalShuffle_seed*_window${WINDOW_SIZE}" | while read -r grammar_dir; do
    if [ -f "$grammar_dir/entropy.csv" ]; then
        echo "Skipping $grammar_dir as entropy.csv already exists."
        continue
    fi
    echo -e "\nProcessing $grammar_dir..."
    python ngram_entropy.py "$grammar_dir" --n 2 3 4 5 --num-processes 1 \
        --memory '8G' --work-dir "$WORK_DIR" || echo "Error processing $grammar_dir"
done


