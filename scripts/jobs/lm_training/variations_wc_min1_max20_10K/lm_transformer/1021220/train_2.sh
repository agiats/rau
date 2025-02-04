#!/bin/bash

#SBATCH --ntasks=10
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=variations_wc_min1_max20_10K_lm_transformer_1021220_seed_2
#SBATCH --output=logs/variations_wc_min1_max20_10K/lm_transformer/1021220_seed_2.out
#SBATCH --error=logs/variations_wc_min1_max20_10K/lm_transformer/1021220_seed_2.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="0"

# Train model
fairseq-train \
    --task language_modeling \
    "data/fairseq_train_preprocessed/variations_wc_min1_max20_10K/1021220" \
    --save-dir "results/transformer_checkpoints/variations_wc_min1_max20_10K/1021220/seed2" \
    --arch transformer_lm_gpt \
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
    --seed 2

mkdir -p "results/transformer_results/variations_wc_min1_max20_10K/1021220/seed2"
# Evaluate on dev set
fairseq-eval-lm \
    "data/fairseq_train_preprocessed/variations_wc_min1_max20_10K/1021220" \
    --path "results/transformer_checkpoints/variations_wc_min1_max20_10K/1021220/seed2/checkpoint_best.pt" \
    --tokens-per-sample 512 \
    --gen-subset "valid" \
    --output-word-probs \
    --quiet 2> "results/transformer_results/variations_wc_min1_max20_10K/1021220/seed2/dev.txt"

# Evaluate on test set
fairseq-eval-lm \
    "data/fairseq_train_preprocessed/variations_wc_min1_max20_10K/1021220" \
    --path "results/transformer_checkpoints/variations_wc_min1_max20_10K/1021220/seed2/checkpoint_best.pt" \
    --tokens-per-sample 512 \
    --gen-subset "test" \
    --output-word-probs \
    --quiet 2> "results/transformer_results/variations_wc_min1_max20_10K/1021220/seed2/test.txt"

# Calculate sentence scores
python scripts/get_sentence_scores.py \
    -i "results/transformer_results/variations_wc_min1_max20_10K/1021220/seed2/test.txt" \
    -O "results/transformer_results/variations_wc_min1_max20_10K/1021220/seed2/test.scores.txt"
