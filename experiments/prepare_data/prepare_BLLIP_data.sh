
set -euo pipefail
. experiments/include.bash


exp_name="local_entropy_non_disjoint"

          submit_job \
            train+"$language"+"$architecture"+"${loss_terms//+/_}"+"$validation_data"+"$trial_no" \
            cpu \
            --time=4:00:00 \
            -- \
            bash recognizers/neural_networks/train_and_evaluate.bash \
              "$BASE_DIR" \
              "$language" \
              "$architecture" \
              "$loss_terms" \
              "$validation_data" \
              "$trial_no" \
              --no-progress

python src/rau/tasks/language_modeling/prepare_data.py \
    --training-data language-modeling-example \
    --more-data validation \
    --more-data test \
    --never-allow-unk
