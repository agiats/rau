#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4g
#SBATCH --gres=gpumem:20g
#SBATCH --tmp=20g
#SBATCH --job-name=lm_transformer_4layer_Base_seed_3
#SBATCH --output=logs/lm_transformer_3_layer_384.out
#SBATCH --error=logs/lm_transformer_3_layer_384.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="0"

# Train model
fairseq-train \
    --task language_modeling \
    "data/fairseq_train_preprocessed/BLLIP_XS/Base" \
    --save-dir "results" \
    --arch transformer_lm_gpt \
    --decoder-layers 4 \
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
    --seed 3

