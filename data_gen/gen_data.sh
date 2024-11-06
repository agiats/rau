# python sample_sentences.py -g base-grammar.gr -n 10000 -O . -b False
# python permute_sentences.py -s sample_base-grammar.txt -O permuted_samples/
# python make_splits.py -S permuted_samples/ -O permuted_splits/

python sample_sentences.py -g base-grammar_eos.gr -n 10000 -O . -b False
python permute_sentences.py -s sample_base-grammar_eos.txt -O permuted_samples/
python make_splits.py -S permuted_samples/ -O permuted_splits/
