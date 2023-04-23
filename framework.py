import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
import wandb
import hashlib
from typing import List, Tuple, Dict, Optional, Any, Type
from loguru import logger
from pathlib import Path

from src.benchmarks.lcbench import LCBench
from src.benchmarks.taskset import TaskSet
# from benchmarks.hyperbo import PD1
from src.surrogate_models.hyperparameter_optimizer import HyperparameterOptimizer
from src.surrogate_models.asha import AHBOptimizer
from src.surrogate_models.dehb.interface import DEHBOptimizer
from src.surrogate_models.random_search import RandomOptimizer
import global_variables as gv
import subprocess


# if warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     from surrogate_models.dragonfly import DragonFlyOptimizer


class Framework:
    surrogate_types = {
        'power_law': HyperparameterOptimizer,
        'dyhpo': HyperparameterOptimizer,
        'asha': AHBOptimizer,
        'dehb': DEHBOptimizer,
        # 'dragonfly': DragonFlyOptimizer,
        'random': RandomOptimizer,
    }

    benchmark_types = {
        'lcbench': LCBench,
        'taskset': TaskSet,
        'lcbench_mini': LCBench,
        # 'pd1': PD1,
    }

    def __init__(
        self,
        args: argparse.Namespace,
        seed: int,
        configs: Dict[str, Any] = None
    ):
        """
        Args:
            args: Namespace
                Includes all the arguments given as variables to the main_experiment
                script.
            seed: int
                The seed for the experiment.
        """

        logger.remove()
        # logger.add(sys.stderr, format="{level} | {message}")
        self.log_path = Path(f'./logs/power_law_surrogate_{args.dataset_name}_{seed}.log')
        logger.add(self.log_path, mode='w', format="{level} | {message}")

        if args.benchmark_name == 'lcbench':
            benchmark_extension = os.path.join('lc_bench', 'results', 'data_2k.json')
        elif args.benchmark_name == 'lcbench_mini':
            benchmark_extension = os.path.join('lc_bench', 'results', 'lcbench_airlines.json')
        elif args.benchmark_name == 'taskset':
            benchmark_extension = os.path.join('data', 'taskset')
        elif args.benchmark_name == 'pd1':
            benchmark_extension = 'pd1'
        else:
            raise ValueError(f'Benchmark {args.benchmark_name} not supported')

        benchmark_data_path = os.path.join(
            args.project_dir,
            benchmark_extension,
        )

        benchmark_types = Framework.benchmark_types
        surrogate_types = Framework.surrogate_types

        disable_preprocessing = {
            'dehb',
            # 'asha'
        }

        self.benchmark = benchmark_types[args.benchmark_name](benchmark_data_path, args.dataset_name)
        self.dataset_name = args.dataset_name
        self.seed = seed
        self.max_value = self.benchmark.max_value
        self.min_value = self.benchmark.min_value
        self.total_budget = args.budget_limit
        self.fantasize_step = args.fantasize_step

        self.categorical_indicator = self.benchmark.categorical_indicator
        self.log_indicator = self.benchmark.log_indicator
        self.hp_names = self.benchmark.hp_names
        self.minimization_metric = self.benchmark.minimization_metric
        self.info_dict = dict()

        # set up wandb
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        framework_meta = Framework.set_meta(args.surrogate_name, configs)
        serialized_dict = json.dumps(framework_meta, sort_keys=True, ensure_ascii=True).encode('utf-8')
        config_hash = hashlib.md5(serialized_dict).hexdigest()
        local_name = "local" if gv.IS_DEBUG else "nemo"
        group_name = f"{args.benchmark_name}_{args.surrogate_name}_{commit_hash}_{config_hash}_{local_name}"
        wandb.init(
            project="power_law_hpo",
            config=framework_meta,
            group=group_name,
            tags=[args.benchmark_name, args.dataset_name, args.surrogate_name, config_hash, commit_hash,
                  str(self.seed), local_name],
            job_type=args.dataset_name,
            name=f"{args.dataset_name}_{self.seed}",
        )

        self.result_dir = os.path.join(
            args.output_dir,
            args.benchmark_name,
            f"{args.surrogate_name}_{commit_hash}_{config_hash}_{local_name}",
        )
        os.makedirs(self.result_dir, exist_ok=True)

        self.result_file = os.path.join(
            self.result_dir,
            f'{self.dataset_name}_{self.seed}.json',
        )

        self.incumbent_hp_index = self.benchmark.get_incumbent_config_id()
        self.pred_curves_path = os.path.join(self.result_dir, "pred_curves", self.dataset_name, str(self.seed))
        os.makedirs(self.pred_curves_path, exist_ok=True)

        if args.surrogate_name not in disable_preprocessing:
            self.hp_candidates = self.preprocess(self.benchmark.get_hyperparameter_candidates())
        else:
            self.hp_candidates = self.benchmark.get_hyperparameter_candidates()

        if args.surrogate_name == 'power_law' or args.surrogate_name == 'dyhpo':
            gv.IS_DYHPO = (args.surrogate_name == 'dyhpo')
            self.surrogate = surrogate_types[args.surrogate_name](
                self.hp_candidates,
                surrogate_name=args.surrogate_name,
                seed=seed,
                max_benchmark_epochs=self.benchmark.max_budget,
                fantasize_step=self.fantasize_step,
                minimization=self.minimization_metric,
                total_budget=args.budget_limit,
                device='cpu',
                dataset_name=args.dataset_name,
                output_path=self.result_dir,
                max_value=self.max_value,
                min_value=self.min_value,
            )
        else:
            self.surrogate = surrogate_types[args.surrogate_name](
                hyperparameter_candidates=self.hp_candidates,
                param_space=self.benchmark.param_space,
                min_budget=self.benchmark.min_budget,
                max_budget=self.benchmark.max_budget,
                eta=3,
                seed=seed,
                max_nr_trials=args.budget_limit,
                maximization=not self.benchmark.minimization_metric,
            )

    def finish(self):
        wandb.log_artifact(str(self.log_path), name='debug_log', type='log')
        wandb.log_artifact(str(self.result_file), name='result_json', type='result')
        wandb.finish()

    @classmethod
    def set_meta(cls, surrogate_name, configs):
        model_class = cls.surrogate_types[surrogate_name]
        meta = model_class.set_meta(surrogate_name, configs)
        return meta

    def run(self):

        evaluated_configs = dict()
        surrogate_budget = 0

        if self.benchmark.minimization_metric:
            best_value = np.inf
        else:
            best_value = 0

        incumbent_value = self.benchmark.get_best_performance()

        while surrogate_budget < self.total_budget:

            start_time = time.time()
            hp_index, budget = self.surrogate.suggest()

            if budget == 10 or budget == 20 or budget == 40:
                self.surrogate.plot_pred_curve(hp_index, self.benchmark, surrogate_budget, self.pred_curves_path)
                self.surrogate.plot_pred_curve(self.incumbent_hp_index, self.benchmark, surrogate_budget,
                                               self.pred_curves_path, prefix="incumbent_")

            hp_curve = self.benchmark.get_curve(hp_index, budget)
            self.surrogate.observe(hp_index, budget, hp_curve)
            time_duration = time.time() - start_time

            if hp_index in evaluated_configs:
                previous_budget = evaluated_configs[hp_index]
            else:
                previous_budget = 0

            budget_cost = budget - previous_budget
            evaluated_configs[hp_index] = budget

            step_time_duration = time_duration / budget_cost

            for epoch in range(previous_budget + 1, budget + 1):
                epoch_performance = float(hp_curve[epoch - 1])
                if self.benchmark.minimization_metric:
                    if best_value > epoch_performance:
                        best_value = epoch_performance
                else:
                    if best_value < epoch_performance:
                        best_value = epoch_performance

                surrogate_budget += 1

                if surrogate_budget > self.total_budget:
                    self.finish()
                    return

                if self.minimization_metric:
                    regret = best_value - incumbent_value
                else:
                    regret = incumbent_value - best_value

                self.log_info(
                    int(hp_index),
                    epoch_performance,
                    epoch,
                    best_value,
                    step_time_duration,
                )
                metrics = {
                    'hpo/hp': int(hp_index),
                    'hpo/scores': epoch_performance,
                    'hpo/epochs': epoch,
                    'hpo/curve': best_value,
                    'hpo/overhead': step_time_duration,
                    'hpo/surrogate_budget': surrogate_budget,
                    'hpo/regret': regret,
                }
                wandb.log(metrics)

        self.finish()

    def preprocess(self, hp_candidates: np.ndarray) -> np.ndarray:
        """Preprocess the hyperparameter candidates.

        Performs min-max standardization for the numerical attributes and
        additionally one-hot encoding for the categorical attributes.

        Args:
            hp_candidates: np.ndarray
                The hyperparameter candidates in their raw form as taken
                from the benchmark.

        Returns:
            preprocessed_candidates: np.ndarray
                The transformed hyperparameter candidates after being
                preprocessed.
        """
        column_transformers = []
        numerical_columns = [
            col_index for col_index, category_indicator in enumerate(self.categorical_indicator)
            if not category_indicator
        ]
        categorical_columns = [
            col_index for col_index, category_indicator in enumerate(self.categorical_indicator)
            if category_indicator
        ]

        general_transformers = []

        if len(numerical_columns) > 0:

            if self.log_indicator is not None and any(self.log_indicator):
                log_columns = [col_index for col_index, log_indicator in enumerate(self.log_indicator) if log_indicator]
                log_transformer = FunctionTransformer(np.log)
                column_transformers.append(
                    (
                        'log_pre',
                        ColumnTransformer(
                            [('log', log_transformer, log_columns)],
                            remainder='passthrough'
                        )
                    )
                )

            general_transformers.append(('num', MinMaxScaler(), numerical_columns))

        if len(categorical_columns) > 0:
            general_transformers.append(
                (
                    'cat',
                    OneHotEncoder(
                        categories=[self.hp_names] * hp_candidates.shape[1],
                        sparse=False,
                    ),
                    categorical_columns,
                )
            )
        column_transformers.append(
            ('feature_types_pre', ColumnTransformer(general_transformers, remainder='passthrough')))

        preprocessor = Pipeline(
            column_transformers
        )

        sklearn.set_config(transform_output="pandas")
        hp_candidates_pd = pd.DataFrame(hp_candidates, columns=self.hp_names)

        preprocessed_candidates = preprocessor.fit_transform(hp_candidates_pd)

        # log preprocessing will push numerical columns to the right
        # so a mapping has to happen for the feature_types_pre
        new_column_map = []
        for name in hp_candidates_pd.columns:
            for new_name in preprocessed_candidates.columns:
                if name in new_name:
                    new_column_map.append(new_name)

        preprocessed_candidates = preprocessed_candidates[new_column_map]
        preprocessed_candidates = preprocessed_candidates.to_numpy()

        return preprocessed_candidates

    def log_info(
        self,
        hp_index: int,
        performance: float,
        budget: int,
        best_value_observed: float,
        time_duration: float,
    ):
        """Log information after every HPO iteration.

        Args:
            hp_index: int
                The index of the suggested hyperparameter candidate.
            performance: float
                The performance of the hyperparameter candidate.
            budget: int
                The budget at which the hyperpararameter candidate has been evaluated so far.
            best_value_observed: float
                The incumbent value observed so far during the optimization.
            time_duration: float
                The time taken for the HPO iteration.
        """
        if 'hp' in self.info_dict:
            self.info_dict['hp'].append(hp_index)
        else:
            self.info_dict['hp'] = [hp_index]

        accuracy_performance = performance

        if 'scores' in self.info_dict:
            self.info_dict['scores'].append(accuracy_performance)
        else:
            self.info_dict['scores'] = [accuracy_performance]

        incumbent_acc_performance = best_value_observed

        if 'curve' in self.info_dict:
            self.info_dict['curve'].append(incumbent_acc_performance)
        else:
            self.info_dict['curve'] = [incumbent_acc_performance]

        if 'epochs' in self.info_dict:
            self.info_dict['epochs'].append(budget)
        else:
            self.info_dict['epochs'] = [budget]

        if 'overhead' in self.info_dict:
            self.info_dict['overhead'].append(time_duration)
        else:
            self.info_dict['overhead'] = [time_duration]

        with open(self.result_file, 'w') as fp:
            json.dump(self.info_dict, fp)
