#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=variations_wc_min1_max20_10K_preprocess_0121200
#SBATCH --output=logs/variations_wc_min1_max20_10K/preprocess/0121200.out
#SBATCH --error=logs/variations_wc_min1_max20_10K/preprocess/0121200.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/variations_wc_min1_max20_10K/0121200/train.txt" \
    --validpref "data/fairseq_train/variations_wc_min1_max20_10K/0121200/dev.txt" \
    --testpref "data/fairseq_train/variations_wc_min1_max20_10K/0121200/test.txt" \
    --destdir "data/fairseq_train_preprocessed/variations_wc_min1_max20_10K/0121200" \
    --workers 64
