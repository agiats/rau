#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=10g
#SBATCH --job-name=local_entropy_preprocess_Q32_S16_s17218
#SBATCH --output=logs/local_entropy/preprocess/Q32_S16_s17218.out
#SBATCH --error=logs/local_entropy/preprocess/Q32_S16_s17218.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/local_entropy/Q32_S16_s17218/train.txt" \
    --validpref "data/fairseq_train/local_entropy/Q32_S16_s17218/dev.txt" \
    --testpref "data/fairseq_train/local_entropy/Q32_S16_s17218/test.txt" \
    --destdir "data/fairseq_train_preprocessed/local_entropy/Q32_S16_s17218" \
    --workers 4
