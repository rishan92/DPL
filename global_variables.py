import getpass
import os

user_name = getpass.getuser()

IS_NEMO = 1 if user_name == 'fr_rs442' or user_name == 'senanaya' else 0

IS_WANDB = 1
if IS_WANDB and "WANDB_API_KEY" not in os.environ:
    IS_WANDB = 0

PLOT_SUGGEST_LR = 0

PLOT_PRED_CURVES = 0
PLOT_PRED_DIST = 0
PLOT_GRADIENTS = 0
PLOT_ACQ = 0
PLOT_PRED_TREND = 0
