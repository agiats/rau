#!/bin/bash

#SBATCH --ntasks=16
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=6switches_3values_min1_max20_10K_preprocess_201110
#SBATCH --output=logs/6switches_3values_min1_max20_10K/preprocess/201110.out
#SBATCH --error=logs/6switches_3values_min1_max20_10K/preprocess/201110.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/6switches_3values_min1_max20_10K/201110/train.txt" \
    --validpref "data/fairseq_train/6switches_3values_min1_max20_10K/201110/dev.txt" \
    --testpref "data/fairseq_train/6switches_3values_min1_max20_10K/201110/test.txt" \
    --destdir "data/fairseq_train_preprocessed/6switches_3values_min1_max20_10K/201110" \
    --workers 16
