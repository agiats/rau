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


# Counterfactuals using W&C
## Data Generation
```bash
# generate counterfactual grammars
bash scripts/generate_grammar_variasions.sh

# length-constrained sampling
python scripts/length_sampling.py \
    --grammar_file $grammar_file \
    --start_symbol "S" \
    --normalize \
    --seed 42 \
    --min_length $min_length \
    --max_length $max_length \
    --num_samples $n_samples \
    --output_path "${output_dir}/${grammar_name}/samples.txt.gz" \
    --num_workers $n_processes \
    --output_sent

# convert the samples to sentence counts
python scripts/samples_to_sentence_counts.py \
    --input_file "$input_file" \
    --output_file "$output_file" \
    --sample_size 10_000_000

# calculate the true probability of the sampled sentences
python scripts/calculate_true_prob.py \
    --grammar_file data/grammars/variations//$grammar_name \
    --start_symbol "S" \
    --normalize \
    --sentence_counts_path "results/length_sampling/{exp_name}/sample_counts.csv.gz" \
    --output_path "${output_path}.csv.gz" \
    --num_workers $n_processes


# calculate the lower bound perplexity for each grammar
python scripts/calculate_lower_bound_perplexity.py \
    --grammar_file data/grammars/variations//${grammar_name} \
    --start_symbol "S" \
    --normalize \
    --min_length $min_length \
    --max_length $max_length \
    --exp_dir $exp_dir \
    --input_suffix ".csv.gz"

```
## Train/Evaluate
```bash
# train  (and no split)
input_file="${input_dir}/samples.txt.gz"
output_dir="data/fairseq_train/variations_wc_min1_max20_10K/$(basename ${input_dir})"
python scripts/split_sampled_data.py \
    --input_file $input_file \
    --output_dir $output_dir

# train the language model for each split
python scripts/generate_job_for_fairseq_training.py

# evaluate the delta-perplexity of the trained models
notebooks/evaluate_delta_perplexity.ipynb
```
