n_samples=100_000_000
n_processes=15
output_dir="results"
output_name="100M_samples_expansion_50_zipf"
max_expansions=50
batch_size=50000
grammar_classes=("PCFG")


# grammar_file="data_gen/base-grammar_eos.gr"
grammar_file="data_gen/base-grammar_eos_zipf.gr"


for grammar_class in "${grammar_classes[@]}"
do
    echo "Grammar class: $grammar_class"
    python monte_carlo_simulation.py \
        --grammar_class $grammar_class \
        --grammar_file $grammar_file \
        --n_samples $n_samples \
        --max_expansions $max_expansions \
        --n_processes $n_processes \
        --output_dir $output_dir \
        --batch_size $batch_size \
        --output_name $output_name
done

# for grammar_class in "${grammar_classes[@]}"
# do
#     echo "Grammar class: $grammar_class"
#     # jq '.Entropy' results/$output_name/$grammar_class/results.json
#     cat results/$output_name/$grammar_class/results.json
# done
