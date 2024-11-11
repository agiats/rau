n_samples=100_000_000
grammar_name="base-grammar_eos_zipf.gr"
min_length=1
max_length=20
exp_dir="results/length_sampling/100M_samples_eos_zipf_min${min_length}_max${max_length}"


export PYTHONPATH="."
python scripts/calculate_lower_bound_perplexity.py \
    --grammar_file data_gen/${grammar_name} \
    --start_symbol "S" \
    --normalize \
    --min_length $min_length \
    --max_length $max_length \
    --exp_dir $exp_dir \
    --input_suffix ".json.gz"
