#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=eos_zipf_min1_max_20_100K_dedup_preprocess_Base
#SBATCH --output=logs/eos_zipf_min1_max_20_100K_dedup/preprocess/Base.out
#SBATCH --error=logs/eos_zipf_min1_max_20_100K_dedup/preprocess/Base.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/Base/train.txt" \
    --validpref "data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/Base/dev.txt" \
    --testpref "data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/Base/test.txt" \
    --destdir "data/fairseq_train_preprocessed/eos_zipf_min1_max_20_100K_dedup/Base" \
    --workers 64
