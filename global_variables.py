import getpass

user_name = getpass.getuser()

IS_NEMO = 0 if user_name == 'fr_rs442' else 1
IS_WANDB = 1

PLOT_PRED_CURVES = 1
PLOT_INIT_DIST = 0
