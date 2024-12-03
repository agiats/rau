input_path="data/grammars/base-grammar.gr"
output_dir="data/grammars/variations/initial_exp"
num_grammar=20
mkdir -p $output_dir
python scripts/generate_grammar_variations.py $input_path $output_dir -n $num_grammar
