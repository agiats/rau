#!/bin/bash

#SBATCH --ntasks=10
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=eos_zipf_min1_max_20_lm_transformer_Base_split_5
#SBATCH --output=logs/eos_zipf_min1_max_20/lm_transformer/Base/split_5.out
#SBATCH --error=logs/eos_zipf_min1_max_20/lm_transformer/Base/split_5.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="0"

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/eos_zipf_min1_max_20/Base/split_5.train" \
    --validpref "data/fairseq_train/eos_zipf_min1_max_20/Base/split_5.dev" \
    --testpref "data/fairseq_train/eos_zipf_min1_max_20/Base/split_5.test" \
    --destdir "data/fairseq_train_preprocessed/eos_zipf_min1_max_20/Base/split_5" \
    --workers 10

    # Train model
fairseq-train \
    --task language_modeling \
    "data/fairseq_train_preprocessed/eos_zipf_min1_max_20/Base/split_5" \
    --save-dir "results/transformer_checkpoints/Base/split_5" \
    --arch transformer_lm_gpt2_small \
    --share-decoder-input-output-embed \
    --dropout 0.3 \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' \
    --weight-decay 0.01 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --clip-norm 0.0 \
    --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 \
    --sample-break-mode none \
    --max-tokens 2048 \
    --update-freq 16 \
    --patience 5 \
    --max-update 10000 \
    --no-epoch-checkpoints \
    --no-last-checkpoints

mkdir -p "results/transformer_results/Base"
# Evaluate on dev set
fairseq-eval-lm \
    "data/fairseq_train_preprocessed/eos_zipf_min1_max_20/Base/split_5" \
    --path "results/transformer_checkpoints/Base/split_5/checkpoint_best.pt" \
    --tokens-per-sample 512 \
    --gen-subset "valid" \
    --output-word-probs \
    --quiet 2> "results/transformer_results/Base/split_5.dev.txt"

# Evaluate on test set
fairseq-eval-lm \
    "data/fairseq_train_preprocessed/eos_zipf_min1_max_20/Base/split_5" \
    --path "results/transformer_checkpoints/Base/split_5/checkpoint_best.pt" \
    --tokens-per-sample 512 \
    --gen-subset "test" \
    --output-word-probs \
    --quiet 2> "results/transformer_results/Base/split_5.test.txt"

# Calculate sentence scores
python scripts/get_sentence_scores.py \
    -i "results/transformer_results/Base/split_5.test.txt" \
    -O "results/transformer_results/Base/split_5.test.scores.txt"
