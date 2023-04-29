import argparse
import json
import os
import sys
import numpy as np
import random
import torch
from loguru import logger
from pathlib import Path
import wandb
import signal
import time
import matplotlib

from framework import Framework
import global_variables as gv

if gv.IS_NEMO:
    matplotlib.use('Agg')


def main():
    current_seed = os.environ.get("PYTHONHASHSEED")
    if current_seed is None or current_seed != '0':
        logger.warning(
            f'Environment variable not set PYTHONHASHSEED="0". Results may not be reproducible due to hash randamization.')

    parser = argparse.ArgumentParser(
        description='DPL publication experiments.',
    )
    parser.add_argument(
        '--index',
        type=int,
        default=1,
        help='The worker index. Every worker runs the same experiment, however, with a different seed.',
    )
    parser.add_argument(
        '--fantasize_step',
        type=int,
        default=1,
        help='The step used in fantasizing the next learning curve value from the last'
             'observed one for a certain hyperparameter configuration.',
    )
    parser.add_argument(
        '--budget_limit',
        type=int,
        default=1000,
        help='The maximal number of HPO iterations.',
    )
    parser.add_argument(
        '--ensemble_size',
        type=int,
        default=5,
        help='The ensemble size for the DPL surrogate.',
    )
    parser.add_argument(
        '--nr_epochs',
        type=int,
        default=250,
        help='The number of epochs used to train (not refine) the HPO surrogate.',
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='credit-g',
        help='The name of the dataset used in the experiment.'
             'The dataset names must be matched with the benchmark they belong to.',
    )
    parser.add_argument(
        '--benchmark_name',
        type=str,
        default='lcbench',
        help='The name of the benchmark used in the experiment. '
             'Every benchmark offers its own distinctive datasets. Available options are lcbench, taskset and pd1.',
    )
    parser.add_argument(
        '--surrogate_name',
        type=str,
        default='power_law',
        help='The method that will be run.',
    )
    parser.add_argument(
        '--project_dir',
        type=str,
        default='.',
        help='The directory where the project files are located.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='The directory where the project output files will be stored.',
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='The file where the project configuration is stored or configuration in json syntax',
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help="Enable the debug logging"
    )

    args = parser.parse_args()
    seeds = np.arange(10)
    seed = seeds[args.index - 1]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    config = None
    if args.config is not None and args.config != "":
        if os.path.isfile(args.config):
            # Read and parse the JSON file
            with open(args.config, 'r') as file:
                config = json.load(file)
        else:
            # Attempt to parse the input string as JSON
            try:
                config = json.loads(args.config)
            except json.JSONDecodeError:
                raise json.JSONDecodeError("The --config input is neither a file path nor a valid JSON string")

    framework = Framework(args=args, seed=seed, config=config)

    def signal_handler(sig, frame):
        framework.finish(is_failed=True)
        sys.exit(0)

    # Register the signal handler for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        framework.run()
    except Exception as ex:
        framework.finish(is_failed=True)
        raise ex


if __name__ == "__main__":
    main()
