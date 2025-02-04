SINGULARITY_IMAGE_FILE=/cluster/work/cotterell/bdusell/container-images/private-nn-formal-languages.sif
BASE_DIR=/cluster/work/cotterell/tsomeya/projects/lm_inductive_bias/results
FIGURES_DIR=$BASE_DIR/figures


ARCHITECTURES=(transformer rnn lstm)
LOSS_TERMS=(rec rec+lm rec+ns rec+lm+ns)
VALIDATION_SETS=(validation-{short,long})
TRIALS=({1..10})
RANDOM_SEED=123456789
RANDOM_AUTOMATON_CONFIGS=(
  finite-automaton,30,8,1
)

submit_job() {
  bash experiments/submit_job.bash "$@"
}


