# python sample_sentences.py -g base-grammar.gr -n 10000 -O . -b False
# python permute_sentences.py -s sample_base-grammar.txt -O permuted_samples/
# python make_splits.py -S permuted_samples/ -O permuted_splits/

# python sample_sentences.py -g base-grammar_eos_zipf.gr -n 100000 -O . -b False
# python permute_sentences.py -s sample_base-grammar_eos_zipf.txt -O permuted_samples/
# python make_splits.py -S permuted_samples/ -O permuted_splits/



# kallini
python generate_sentences_kallini.py -s sample_base-grammar_eos_zipf.txt -O counterfactual_kallini_samples/PCFG.txt

grammar_classes=("PCFG" "PCFGDeterministicShuffle" "PCFGNonDeterministicShuffle" "PCFGLocalShuffle" "PCFGEvenOddShuffle" "PCFGNoReverse" "PCFGPartialReverse" "PCFGFullReverse")
for grammar_class in "${grammar_classes[@]}"
do
    python perturb_sentences_kallini.py --grammar_class $grammar_class --base_sentence_path counterfactual_kallini_samples/PCFG.txt --output_dir counterfactual_kallini_samples/
done

python make_splits_kallini.py -S counterfactual_kallini_samples/ -O counterfactual_kallini_splits/
