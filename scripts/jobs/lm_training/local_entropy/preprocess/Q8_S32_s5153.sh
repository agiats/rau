#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --tmp=10g
#SBATCH --job-name=local_entropy_preprocess_Q8_S32_s5153
#SBATCH --output=logs/local_entropy/preprocess/Q8_S32_s5153.out
#SBATCH --error=logs/local_entropy/preprocess/Q8_S32_s5153.err

module load stack/2024-05 gcc/13.2.0 python/3.10.13
source .venv/bin/activate
export PYTHONPATH=.

# Preprocess data
fairseq-preprocess \
    --only-source \
    --trainpref "data/fairseq_train/local_entropy/Q8_S32_s5153/train.txt" \
    --validpref "data/fairseq_train/local_entropy/Q8_S32_s5153/dev.txt" \
    --testpref "data/fairseq_train/local_entropy/Q8_S32_s5153/test.txt" \
    --destdir "data/fairseq_train_preprocessed/local_entropy/Q8_S32_s5153" \
    --workers 4
