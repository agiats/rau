#!/bin/bash

#SBATCH --ntasks=10
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=ptb_local_shuffle_lm_lstm_ptb_local_shuffle_LocalShuffle_seed19_window4_seed_4
#SBATCH --output=logs/ptb_local_shuffle/lm_lstm/ptb_local_shuffle_LocalShuffle_seed19_window4_seed_4.out
#SBATCH --error=logs/ptb_local_shuffle/lm_lstm/ptb_local_shuffle_LocalShuffle_seed19_window4_seed_4.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="0"

# Train model
fairseq-train \
    --task language_modeling \
    "data/fairseq_train_preprocessed/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4" \
    --save-dir "results/lstm_checkpoints/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4/seed4" \
    --arch lstm_lm \
    --share-decoder-input-output-embed \
    --dropout 0.3 \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --weight-decay 0.01 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 \
    --sample-break-mode none \
    --max-tokens 2048 \
    --patience 5 \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --validate-interval-updates	100 \
    --seed 4

mkdir -p "results/lstm_results/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4/seed4"
# Evaluate on dev set
fairseq-eval-lm \
    "data/fairseq_train_preprocessed/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4" \
    --path "results/lstm_checkpoints/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4/seed4/checkpoint_best.pt" \
    --tokens-per-sample 512 \
    --gen-subset "valid" \
    --output-word-probs \
    --quiet 2> "results/lstm_results/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4/seed4/dev.txt"

# Evaluate on test set
fairseq-eval-lm \
    "data/fairseq_train_preprocessed/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4" \
    --path "results/lstm_checkpoints/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4/seed4/checkpoint_best.pt" \
    --tokens-per-sample 512 \
    --gen-subset "test" \
    --output-word-probs \
    --quiet 2> "results/lstm_results/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4/seed4/test.txt"

# Calculate sentence scores
python scripts/get_sentence_scores.py \
    -i "results/lstm_results/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4/seed4/test.txt" \
    -O "results/lstm_results/ptb_local_shuffle/ptb_local_shuffle_LocalShuffle_seed19_window4/seed4/test.scores.txt"
