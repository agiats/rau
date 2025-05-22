SINGULARITY_IMAGE_FILE=/cluster/work/cotterell/bdusell/container-images/private-nn-formal-languages.sif
DATA_DIR=/cluster/work/cotterell/tsomeya/projects/lm_inductive_bias/data
RESULTS_DIR=/cluster/work/cotterell/tsomeya/projects/lm_inductive_bias/results
RAU_DIR=/cluster/home/tsomeya/projects/lm_inductive_bias/src/rau
FIGURES_DIR=$RESULTS_DIR/figures

PERTURBATION_CONFIG_FILE=/cluster/home/tsomeya/projects/lm_inductive_bias/config/perturbation_func.json
ARCHITECTURES=("lstm" "transformer" "stack-rnn_lstm_superposition-20" "stack-transformer_32-2.superposition-32.2" "stack-rnn_lstm_512_superposition-64" "stack-transformer_768-2.superposition-64.2")
ARCHITECTURE_LABELS=("LSTM" "Transformer" "RNN+Sup. 20" "Tf+Sup. 32" "RNN+Sup. 64" "Tf+Sup. 64")


NUM_TRIALS=5
submit_job() {
  bash experiments/submit_job.bash "$@"
}
