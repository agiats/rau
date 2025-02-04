sample_size=10_000
sentence_counts_dir="data/variations_wc/6switches_3values_min1_max20_10K"

for grammar_file in data/grammars/variations/6switches_3values/*.gr; do
    grammar_name=$(basename ${grammar_file} .gr)
    job_base_dir="scripts/jobs/6switches_3values_min1_max20_10K/${grammar_name}"
    sentence_counts_path="${sentence_counts_dir}/${grammar_name}/sample_counts.csv.gz"
    output_dir="${sentence_counts_dir}/${grammar_name}/true_prob"
    mkdir -p $output_dir

    python scripts/generate_job_for_true_prob.py \
        --grammar_file $grammar_file \
        --job_output_dir $job_base_dir \
        --num_jobs 1 \
        --log_dir "6switches_3values_min1_max20_10K/${grammar_name}" \
        --sentence_counts_path $sentence_counts_path \
        --output_dir $output_dir \
        --sample_size $sample_size
done
