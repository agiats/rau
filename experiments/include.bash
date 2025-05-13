SINGULARITY_IMAGE_FILE=/cluster/work/cotterell/bdusell/container-images/private-nn-formal-languages.sif
DATA_DIR=/cluster/work/cotterell/tsomeya/projects/lm_inductive_bias/data
RESULTS_DIR=/cluster/work/cotterell/tsomeya/projects/lm_inductive_bias/results
RAU_DIR=/cluster/home/tsomeya/projects/lm_inductive_bias/src/rau
FIGURES_DIR=$RESULTS_DIR/figures

PERTURBATION_CONFIG_FILE=/cluster/home/tsomeya/projects/lm_inductive_bias/config/perturbation_func.json
ARCHITECTURES=("lstm" "transformer" "stack-rnn_lstm_vector-nondeterministic-3-3-5" "stack-transformer_nondeterministic")
ARCHITECTURE_LABELS=("LSTM" "Transformer" "VRNS-3-3-5" "Tf+Nd")


NUM_TRIALS=5
submit_job() {
  bash experiments/submit_job.bash "$@"
}
