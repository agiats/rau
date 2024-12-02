#!/bin/bash

#SBATCH --ntasks=64
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=20g
#SBATCH --job-name=eos_zipf_min1_max_20_100K_dedup_preprocess_EvenOddShuffle
#SBATCH --output=logs/eos_zipf_min1_max_20_100K_dedup/preprocess/EvenOddShuffle.out
#SBATCH --error=logs/eos_zipf_min1_max_20_100K_dedup/preprocess/EvenOddShuffle.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13 cuda/12.1.1
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/EvenOddShuffle/train.txt" \
    --validpref "data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/EvenOddShuffle/dev.txt" \
    --testpref "data/fairseq_train/eos_zipf_min1_max_20_100K_dedup/EvenOddShuffle/test.txt" \
    --destdir "data/fairseq_train_preprocessed/eos_zipf_min1_max_20_100K_dedup/EvenOddShuffle" \
    --workers 64
