# Mission: Impossible Language Model II

## Data Preparation
```bash
# length-constrained sampling
bash scripts/jobs/perturb_sentences.sh

# [optional] length-balanced filtering of the generated sentences
python scripts/length_balanced_filtering.py

# calculate the true probability of the sampled sentences
python scripts/calculate_true_prob.py

# sanity check
# Does the monte-carlo sampling work for the length-constrained sampling?
bash scripts/sanity_check.sh

# perturb sentences
python scripts/perturb_sentences.py
python scripts/samples_to_sentence_counts.py

# calculate lower bound perplexity for each grammar
python scripts/calculate_lower_bound_perplexity.py
python scripts/calculate_entropy_of_samples.py

# see entropy for each grammar
notebooks/compare_complexity_measures.ipynb

```

## Train (white&cotterell)
```bash
# split th data according to White&Cotterell (2021)
# create 10 data of 10k samples and save them to /data folder
python scripts/split_sampled_data.py \
    --input_file results/length_sampling/100M_samples_eos_zipf_min1_max20_DeterministicShuffle/samples.txt.gz \
    --output_dir data/fairseq_train/eos_zipf_min1_max_20/DeterministicShuffle \
    --num_splits 10 \
    --num_samples_per_split 10000


# train the language model for each split
python scripts/generate_job_for_fairseq_training_white.py
```

## Train (no split)
```bash
# train using 10M sentences (and no split)
python scripts/split_sampled_data.py \
    --input_file results/length_sampling/10M_samples_eos_zipf_min1_max20_DeterministicShuffle/samples.txt.gz \
    --output_dir data/fairseq_train/eos_zipf_min1_max_20_10M/DeterministicShuffle \
    --num_splits 1 \
    --num_samples_per_split 10_000_000

# train the language model for each split
python scripts/generate_job_for_fairseq_training.py
```

## Evaluate
```bash
# evaluate the delta-perplexity of the trained models
notebooks/evaluate_delta_perplexity.ipynb
```

```


