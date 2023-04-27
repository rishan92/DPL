import getpass

user_name = getpass.getuser()

IS_NEMO = 0 if user_name == 'fr_rs442' else 1
IS_WANDB = 0

PLOT_PRED_CURVES = 1
PLOT_PRED_DIST = 1
