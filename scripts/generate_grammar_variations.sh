input_file="data/grammars/base-grammar_kuribayashi_zipf.gr"
output_dir="data/grammars/variations/6switches_3values_zipf"
num_grammar=200
mkdir -p $output_dir

python scripts/generate_grammar_variations.py \
    --input_file $input_file \
    --output_dir $output_dir \
    --num_grammar $num_grammar \
    --num_switches 6 \
    --include_all_deterministic_grammar
