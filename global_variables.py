import getpass
import os

user_name = getpass.getuser()

IS_NEMO = 1 if user_name == 'fr_rs442' else 0

IS_WANDB = 1
if IS_WANDB and "WANDB_API_KEY" not in os.environ:
    IS_WANDB = 0

PLOT_SUGGEST_LR = 1

PLOT_PRED_CURVES = 0
PLOT_PRED_DIST = 0
PLOT_GRADIENTS = 1
PLOT_ACQ = 1
PLOT_PRED_TREND = 0
