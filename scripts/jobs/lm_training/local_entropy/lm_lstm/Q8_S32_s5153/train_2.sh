#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=16g
#SBATCH --gres=gpumem:20g
#SBATCH --tmp=20g
#SBATCH --job-name=local_entropy_lm_lstm_Q8_S32_s5153_seed_2
#SBATCH --output=logs/local_entropy/lm_lstm/Q8_S32_s5153_seed_2.out
#SBATCH --error=logs/local_entropy/lm_lstm/Q8_S32_s5153_seed_2.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="0"

# Train model
fairseq-train \
    --task language_modeling \
    "data/fairseq_train_preprocessed/local_entropy/Q8_S32_s5153" \
    --save-dir "results/lstm_checkpoints/local_entropy/Q8_S32_s5153/seed2" \
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
    --patience 10 \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --seed 2 

mkdir -p "results/lstm_results/local_entropy/Q8_S32_s5153/seed2"
# Evaluate on dev set
fairseq-eval-lm \
    "data/fairseq_train_preprocessed/local_entropy/Q8_S32_s5153" \
    --path "results/lstm_checkpoints/local_entropy/Q8_S32_s5153/seed2/checkpoint_best.pt" \
    --tokens-per-sample 512 \
    --gen-subset "valid" \
    --output-word-probs \
    --quiet 2> "results/lstm_results/local_entropy/Q8_S32_s5153/seed2/dev.txt"

# Evaluate on test set
fairseq-eval-lm \
    "data/fairseq_train_preprocessed/local_entropy/Q8_S32_s5153" \
    --path "results/lstm_checkpoints/local_entropy/Q8_S32_s5153/seed2/checkpoint_best.pt" \
    --tokens-per-sample 512 \
    --gen-subset "test" \
    --output-word-probs \
    --quiet 2> "results/lstm_results/local_entropy/Q8_S32_s5153/seed2/test.txt"

# Calculate sentence scores
python scripts/get_sentence_scores.py \
    -i "results/lstm_results/local_entropy/Q8_S32_s5153/seed2/test.txt" \
    -O "results/lstm_results/local_entropy/Q8_S32_s5153/seed2/test.scores.txt"
