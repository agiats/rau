#!/bin/bash

#SBATCH --ntasks=16
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=6switches_3values_min1_max20_10K_preprocess_200102
#SBATCH --output=logs/6switches_3values_min1_max20_10K/preprocess/200102.out
#SBATCH --error=logs/6switches_3values_min1_max20_10K/preprocess/200102.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/6switches_3values_min1_max20_10K/200102/train.txt" \
    --validpref "data/fairseq_train/6switches_3values_min1_max20_10K/200102/dev.txt" \
    --testpref "data/fairseq_train/6switches_3values_min1_max20_10K/200102/test.txt" \
    --destdir "data/fairseq_train_preprocessed/6switches_3values_min1_max20_10K/200102" \
    --workers 16
