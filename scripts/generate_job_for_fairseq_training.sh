# model_types=("transformer" "lstm")
# exp_name="6switches_3values_min1_max20_10K"

# for grammar_path in data/grammars/variations/6switches_3values/*.gr; do
#     grammar_name=$(basename ${grammar_path} .gr)
#     for model_type in ${model_types[@]}; do
#         python scripts/generate_job_for_fairseq_training.py \
#             --model_type $model_type \
#             --grammar_name $grammar_name \
#             --output_dir scripts/jobs/lm_training/${exp_name}
#     done
# done



# model_types=("transformer_4layer" "lstm")
# exp_name="local_entropy"

# for exp_dir in data/fairseq_train/local_entropy/*; do
#     dir_name=$(basename "$exp_dir")

#     echo $dir_name

#     for model_type in "${model_types[@]}"; do
#         python scripts/generate_job_for_fairseq_training.py \
#             --model_type "$model_type" \
#             --grammar_name "$dir_name" \
#             --output_dir "scripts/jobs/lm_training/${exp_name}" \
#             --num_seeds 5
#     done
# done


# for f in scripts/jobs/lm_training/local_entropy/preprocess/OddEvenShuffle.sh; do
#     if [[ $f != *"LocalShuffle"* && $f != *"Base"* ]]; then
#         sbatch $f
#     fi
# done

# for f in scripts/jobs/lm_training/local_entropy/lm_transformer_4layer/OddEvenShuffle/train*.sh; do
#     sbatch $f
#     # echo $f
# done
# for f in scripts/jobs/lm_training/local_entropy/lm_lstm/OddEvenShuffle/train*.sh; do
#     sbatch $f
#     # echo $f
# done
# for f in scripts/jobs/lm_training/BLLIP_XS/lm*/*/train*.sh; do sbatch $f; done




# Local entropy

# model_types=("transformer_4layer" "lstm")
# exp_name="local_entropy"

# for exp_dir in data/fairseq_train/local_entropy/*; do
#     dir_name=$(basename "$exp_dir")

#     echo $dir_name

#     for model_type in "${model_types[@]}"; do
#         python scripts/generate_job_for_fairseq_training.py \
#             --model_type "$model_type" \
#             --exp_name "$exp_name" \
#             --grammar_name "$dir_name" \
#             --output_dir "scripts/jobs/lm_training/${exp_name}" \
#             --num_seeds 5
#     done
# done

# # Rename val.txt to dev.txt
# for exp_dir in data/fairseq_train/local_entropy/*; do
#     if [ -f "$exp_dir/val.txt" ]; then
#         mv "$exp_dir/val.txt" "$exp_dir/dev.txt"
#     fi
# done


# Remove last token from each line
# for exp_dir in data/fairseq_train/local_entropy/*; do
#     for file in "$exp_dir"/{train,dev,test}.txt; do
#         if [ -f "$file" ]; then
#             python scripts/remove_last_token.py "$file" "${file}.tmp"
#             mv "${file}.tmp" "$file"
#         fi
#     done
# done


# for f in scripts/jobs/lm_training/local_entropy/preprocess/*.sh; do
#     sbatch $f
#     # echo $f
# done

# for f in scripts/jobs/lm_training/local_entropy/lm*/*/train*.sh; do
#     sbatch $f
#     # echo $f
# done



python evaluate_model_performance.py \
    --exp-name $exp_name \
    --data-dir data/fairseq_train \
    --results-dir results \
    --output-dir results \
    --model-names lstm transformer_4layer \
    --num-seeds 5
