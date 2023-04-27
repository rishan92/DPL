import getpass

user_name = getpass.getuser()

IS_NEMO = 1 if user_name == 'fr_rs442' else 0
IS_WANDB = 0

PLOT_PRED_CURVES = 0
PLOT_PRED_DIST = 0
