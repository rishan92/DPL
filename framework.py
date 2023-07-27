import argparse
import json
import os
import time
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
import wandb
import hashlib
from typing import List, Tuple, Dict, Optional, Any, Type, Union
from loguru import logger
from pathlib import Path

from src.benchmarks.base_benchmark import BaseBenchmark
from src.benchmarks.lcbench import LCBench
from src.benchmarks.taskset import TaskSet
from src.benchmarks.hyperbo import PD1
from src.benchmarks.synthetic import SyntheticBench
from src.benchmarks.yahpo import YAHPOGym
from src.benchmarks.hpobench import HPOBench
from src.benchmarks.jahsbench import JAHSBench
from src.benchmarks.synthetic_mf import SyntheticMFBench
from src.surrogate_models.hyperparameter_optimizer import HyperparameterOptimizer
from src.surrogate_models.asha import AHBOptimizer
from src.surrogate_models.dehb.interface import DEHBOptimizer
from src.surrogate_models.random_search import RandomOptimizer
from src.surrogate_models.many_fidelity_hyperparameter_optimizer import MFHyperparameterOptimizer
from src.surrogate_models.dragonfly import DragonFlyOptimizer
import global_variables as gv
import subprocess
from src.utils.utils import delete_folder_content
from src.history.fidelity_manager import FidelityManager


# if warnings.catch_warnings():
#     warnings.simplefilter('ignore')
#     from surrogate_models.dragonfly import DragonFlyOptimizer


class Framework:
    surrogate_types = {
        'power_law': HyperparameterOptimizer,
        'dyhpo': HyperparameterOptimizer,
        'asha': AHBOptimizer,
        'dehb': DEHBOptimizer,
        'dragonfly': DragonFlyOptimizer,
        'random': RandomOptimizer,
        'many_fidelity': MFHyperparameterOptimizer
    }

    benchmark_types = {
        'lcbench': LCBench,
        'taskset': TaskSet,
        'lcbench_mini': LCBench,
        'pd1': PD1,
        'synthetic': SyntheticBench,
        'yahpo': YAHPOGym,
        'hpobench': HPOBench,
        'jahsbench': JAHSBench,
        'synthetic_mf': SyntheticMFBench,
    }

    def __init__(
        self,
        args: argparse.Namespace,
        seed: int,
        config: Dict[str, Any] = None
    ):
        """
        Args:
            args: Namespace
                Includes all the arguments given as variables to the main_experiment
                script.
            seed: int
                The seed for the experiment.
        """

        self.start_time = time.perf_counter()

        self.benchmark_name: str = args.benchmark_name
        self.dataset_name: str = args.dataset_name
        self.surrogate_name: str = args.surrogate_name
        self.total_budget: int = args.budget_limit
        self.fantasize_step: int = args.fantasize_step
        self.output_dir: Path = Path(args.output_dir)
        self.verbose: bool = args.verbose
        self.project_dir: Path = Path(args.project_dir)

        benchmark_extensions: Dict[str, Path] = {
            'lcbench': Path('lc_bench', 'results', 'data_2k.json'),
            'lcbench_mini': Path('lc_bench', 'results', 'lcbench_airlines.json'),
            'taskset': Path('data', 'taskset'),
            'pd1': Path('data', 'pd1'),
            'synthetic': Path('synthetic'),
            'yahpo': Path('yahpo_data'),
            'hpobench': Path('hpobench'),
            'jahsbench': Path('jahsbench'),
            'synthetic_mf': Path('synthetic_mf'),
        }

        if self.benchmark_name in benchmark_extensions:
            benchmark_extension = benchmark_extensions[self.benchmark_name]
        else:
            raise ValueError(f'Benchmark {self.benchmark_name} not supported')

        benchmark_data_path: Path = self.project_dir / benchmark_extension

        disable_preprocessing = {
            'dehb',
        }

        self.benchmark: BaseBenchmark = Framework.benchmark_types[self.benchmark_name](
            benchmark_data_path,
            self.dataset_name,
            seed=seed
        )
        self.seed = seed
        self.max_value = self.benchmark.max_value
        self.min_value = self.benchmark.min_value
        self.surrogate_budget = 0

        self.categorical_indicator: List[bool] = self.benchmark.categorical_indicator
        self.categories = self.benchmark.categories
        self.log_indicator: List[bool] = self.benchmark.log_indicator
        self.hp_names: List[str] = self.benchmark.hp_names
        self.info_dict: Dict[str, Any] = dict()

        # set up wandb
        commit_hash: str = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        framework_meta: Dict[str, Any] = self.set_meta(self.surrogate_name, config=config)
        serialized_dict = json.dumps(framework_meta, sort_keys=True, ensure_ascii=True).encode('utf-8')
        config_hash: str = hashlib.md5(serialized_dict).hexdigest()
        local_name = "nemo" if gv.IS_NEMO else "local"
        group_name = f"{self.benchmark_name}_{self.surrogate_name}_{commit_hash}_{config_hash}_{local_name}"
        if gv.IS_WANDB:
            wandb.init(
                project="power_law_hpo",
                config=framework_meta,
                group=group_name,
                tags=[self.benchmark_name, self.dataset_name, self.surrogate_name, config_hash, commit_hash,
                      str(self.seed), local_name],
                job_type=self.dataset_name,
                name=f"{self.dataset_name}_{self.seed}",
            )
        else:
            wandb.init(mode="disabled")

        self.result_dir = \
            self.output_dir / self.benchmark_name / f"{self.surrogate_name}_{commit_hash}_{config_hash}_{local_name}"
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.result_file = self.result_dir / f'{self.dataset_name}_{self.seed}.json'

        self.config_folder_path = self.result_dir / 'configuration'
        self.config_folder_path.mkdir(parents=True, exist_ok=True)
        config_file_path = self.config_folder_path / f'{self.surrogate_name}_configurations.json'
        with open(config_file_path, 'w') as f:
            json.dump(framework_meta, f, indent=4)

        logger.remove()
        self.log_path = \
            self.result_dir / 'logs' / f'{self.surrogate_name}_surrogate_{self.dataset_name}_{self.seed}.log'
        if self.verbose:
            log_level = "TRACE"
            format_str = "{level} | {message}"
            logger.add(self.log_path, mode='w', format=format_str, level=log_level)
        else:
            log_level = "SUCCESS"
            logger.add(self.log_path, mode='w', level=log_level)

        print(f"group_name {group_name}")
        logger.success(f"group_name {group_name}")

        # self.incumbent_hp_index = self.benchmark.get_incumbent_config_id()
        self.incumbent_hp_index = 0
        self.pred_curves_path = None
        if gv.PLOT_PRED_CURVES:
            self.pred_curves_path = self.result_dir / "pred_curves" / self.dataset_name / str(self.seed)
            self.pred_curves_path.mkdir(parents=True, exist_ok=True)
            delete_folder_content(self.pred_curves_path)

        if gv.PLOT_PRED_DIST:
            self.pred_dist_path = self.result_dir / "pred_dist" / self.dataset_name / str(self.seed)
            self.pred_dist_path.mkdir(parents=True, exist_ok=True)
            delete_folder_content(self.pred_dist_path)

        self.pred_trend_path = None
        if gv.PLOT_PRED_TREND:
            self.pred_trend_path = self.result_dir / "pred_trend" / self.dataset_name / str(self.seed)
            self.pred_trend_path.mkdir(parents=True, exist_ok=True)
            delete_folder_content(self.pred_trend_path)

        self.hp_candidates: NDArray
        if self.surrogate_name not in disable_preprocessing:
            self.hp_candidates = self.preprocess(self.benchmark.get_hyperparameter_candidates())
        else:
            self.hp_candidates = self.benchmark.get_hyperparameter_candidates()

        self.fidelity_manager: FidelityManager = self.benchmark.fidelity_manager

        if hasattr(self.benchmark, 'fidelity_names') and self.benchmark.fidelity_names is not None:
            max_budgets = self.benchmark.max_budgets
            min_budgets = self.benchmark.min_budgets
            fantasize_step = {k: self.fantasize_step * v for k, v in self.benchmark.fidelity_interval.items()}
            fidelity_names = self.benchmark.fidelity_names
            # surrogate_class = Framework.surrogate_types['many_fidelity']
            surrogate_class = Framework.surrogate_types[self.surrogate_name]
        else:
            max_budgets = self.benchmark.max_budget
            min_budgets = self.benchmark.min_budget
            fantasize_step = self.fantasize_step
            fidelity_names = 'epochs'
            surrogate_class = Framework.surrogate_types[self.surrogate_name]

        if self.surrogate_name in ['power_law', 'dyhpo', 'random']:
            self.surrogate = surrogate_class(
                hp_candidates=self.hp_candidates,
                fidelity_manager=self.benchmark.fidelity_manager,
                surrogate_name=self.surrogate_name,
                seed=seed,
                max_budgets=max_budgets,
                min_budgets=min_budgets,
                fantasize_step=fantasize_step,
                minimization=self.benchmark.minimization_metric,
                total_budget=self.total_budget,
                device='cpu',
                dataset_name=self.dataset_name,
                output_path=self.result_dir,
                max_value=self.max_value,
                min_value=self.min_value,
                benchmark=self.benchmark,
                pred_trend_path=self.pred_trend_path,
                fidelity_names=fidelity_names
            )
        else:
            self.surrogate = surrogate_class(
                hyperparameter_candidates=self.hp_candidates,
                param_space=self.benchmark.param_space,
                min_budget=self.benchmark.min_budget,
                max_budget=self.benchmark.max_budget,
                eta=3,
                seed=seed,
                max_nr_trials=self.total_budget,
                maximization=not self.benchmark.minimization_metric,
            )

    @classmethod
    def set_meta(cls, surrogate_name, config=None):
        config = {} if config is None else config
        model_class = cls.surrogate_types[surrogate_name]
        meta = model_class.set_meta(surrogate_name=surrogate_name, config=config)
        return meta

    def run(self):

        evaluated_configs = dict()

        if self.benchmark.minimization_metric:
            best_value = np.PINF
        else:
            best_value = np.NINF

        incumbent_value = self.benchmark.get_best_performance()

        plot_pred_curves_fidelity_percentile = np.array([5, 10, 20, 40]) / 50
        plot_pred_curves_fidelity = []
        for k in self.fidelity_manager.fidelity_names:
            n = len(self.fidelity_manager.fidelity_space[k])
            indices = [int(round(p * (n - 1))) for p in plot_pred_curves_fidelity_percentile]
            plot_pred_curves_fidelity.append(set(indices))

        while self.surrogate_budget < self.total_budget:

            start_time = time.time()
            hp_indices, fidelity_ids = self.surrogate.suggest()

            if gv.PLOT_PRED_CURVES:
                is_budget = False
                if self.benchmark_name == 'yahpo' or self.benchmark_name == 'synthetic_mf':
                    for i, pred_curves in enumerate(plot_pred_curves_fidelity):
                        is_budget = fidelity_ids[i] in pred_curves
                        if is_budget:
                            break
                else:
                    is_budget = fidelity_ids in plot_pred_curves_fidelity
                if gv.PLOT_PRED_CURVES and (
                    is_budget
                    or self.benchmark_name == 'synthetic'
                    or self.benchmark_name == 'yahpo'
                    # or self.benchmark_name == 'lcbench_mini'
                    or self.benchmark_name == 'synthetic_mf'
                ):
                    hp_index = hp_indices
                    if isinstance(hp_indices, List):
                        hp_index = hp_indices[0]
                    self.surrogate.plot_pred_curve(
                        hp_index=hp_index,
                        benchmark=self.benchmark,
                        surrogate_budget=self.surrogate_budget,
                        output_dir=self.pred_curves_path
                    )
                    # self.surrogate.plot_pred_curve(
                    #     hp_index=self.incumbent_hp_index,
                    #     benchmark=self.benchmark,
                    #     surrogate_budget=self.surrogate_budget,
                    #     output_dir=self.pred_curves_path,
                    #     prefix="incumbent_"
                    # )

            if gv.PLOT_PRED_DIST and self.surrogate_budget % 100 == 1:
                self.surrogate.plot_pred_dist(
                    benchmark=self.benchmark,
                    surrogate_budget=self.surrogate_budget,
                    output_dir=self.pred_dist_path
                )
            if not isinstance(hp_indices, List):
                hp_indices = [hp_indices]
                fidelity_ids = [fidelity_ids]

            hp_curves = []
            for hp_index, fidelity_id in zip(hp_indices, fidelity_ids):
                hp_curve, fidelity_id_out = self.benchmark.get_objective_function_performance(hp_index, fidelity_id)
                # hp_curve = self.benchmark.get_performance(hp_index, budget)
                hp_curves.append(hp_curve)
                self.surrogate.observe(hp_index, fidelity_id_out, hp_curve)

            # if len(hp_indices) == 1:
            #     hp_indices = hp_indices[0]
            #     budgets = budgets[0]
            #     hp_curves = hp_curves[0]

            # self.surrogate.observe(hp_indices, budgets, hp_curves)

            time_duration = time.time() - start_time

            # if hp_index in evaluated_configs:
            #     previous_budget = evaluated_configs[hp_index]
            # else:
            #     previous_budget = 0

            # budget_cost = budget - previous_budget
            # evaluated_configs[hp_index] = budget

            # step_time_duration = time_duration / budget_cost
            step_time_duration = time_duration

            if self.benchmark.minimization_metric:
                best_index = np.argmin(hp_curves)
                budget_performance = hp_curves[best_index]
                budget_performance = np.min(budget_performance)
                if best_value > budget_performance:
                    best_value = budget_performance
            else:
                best_index = np.argmax(hp_curves)
                budget_performance = hp_curves[best_index]
                budget_performance = np.max(budget_performance)
                if best_value < budget_performance:
                    best_value = budget_performance

            best_hp_index = hp_indices[best_index]
            best_fidelity_id = fidelity_ids[best_index]

            self.surrogate_budget += 1

            if self.benchmark.minimization_metric:
                regret = best_value - incumbent_value
            else:
                regret = incumbent_value - best_value

            # log_budget = []
            # for v in best_budget.values():
            #     if isinstance(v, float):
            #         v = float(v)
            #     elif isinstance(v, int):
            #         v = int(v)
            #     else:
            #         raise NotImplementedError
            #     log_budget.append(v)
            log_fidelity = self.fidelity_manager.get_fidelities(best_fidelity_id)
            self.log_info(
                int(best_hp_index),
                float(budget_performance),
                log_fidelity,
                float(best_value),
                step_time_duration,
            )
            metrics = {
                'hpo/hp': int(best_hp_index),
                'hpo/scores': budget_performance,
                'hpo/curve': best_value,
                'hpo/overhead': step_time_duration,
                'hpo/surrogate_budget': self.surrogate_budget,
                'hpo/regret': regret,
            }
            for i, name in enumerate(self.fidelity_manager.fidelity_names):
                metrics[f'hpo/{name}'] = log_fidelity[i]
            wandb.log(metrics)

            if self.surrogate_budget >= self.total_budget or self.surrogate_budget >= self.benchmark.size():
                self.finish()
                return

        self.finish()

    def preprocess(self, hp_candidates: Union[NDArray, pd.DataFrame]) -> NDArray:
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

        categories = "auto"
        if self.categories is not None and len(categorical_columns) != 0:
            categories = [
                self.categories[col_index] for col_index, category_indicator in enumerate(self.categorical_indicator)
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
                        sparse_output=False,
                        categories=categories,
                        handle_unknown='ignore'
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
        preprocessed_candidates.fillna(0, inplace=True)

        # # This is only required to reproduce the results of original DPL or DyHPO.
        # # log preprocessing will push numerical columns to the right
        # # so a mapping has to happen for the feature_types_pre
        # new_column_map = []
        # for name in hp_candidates_pd.columns:
        #     for new_name in preprocessed_candidates.columns:
        #         if name in new_name:
        #             new_column_map.append(new_name)
        # preprocessed_candidates = preprocessed_candidates[new_column_map]

        preprocessed_candidates = preprocessed_candidates.to_numpy()

        return preprocessed_candidates

    def log_info(
        self,
        hp_index: int,
        performance: float,
        budget: List,
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

        if 'budgets' in self.info_dict:
            self.info_dict['budgets'].append(budget)
        else:
            self.info_dict['budgets'] = [budget]

        if 'overhead' in self.info_dict:
            self.info_dict['overhead'].append(time_duration)
        else:
            self.info_dict['overhead'] = [time_duration]

        with open(self.result_file, 'w') as fp:
            json.dump(self.info_dict, fp)

    def finish(self, is_failed=False):
        self.benchmark.close()

        wandb.log_artifact(str(self.log_path), name='debug_log', type='log')
        wandb.log_artifact(str(self.result_file), name='result_json', type='result')

        if gv.IS_WANDB and gv.PLOT_PRED_CURVES and self.benchmark_name != "synthetic":
            file_list = os.listdir(self.pred_curves_path)
            if len(file_list) > 0:
                table = wandb.Table(columns=["id", "plot"])
                for i, file_name in enumerate(file_list):
                    file_path = str(self.pred_curves_path / str(file_name))
                    table.add_data(i, wandb.Image(file_path))

                wandb.log({"table_of_prediction_curves": table})

        if gv.IS_WANDB and gv.PLOT_PRED_DIST and self.benchmark_name != "synthetic":
            file_list = os.listdir(self.pred_dist_path)
            if len(file_list) > 0:
                table = wandb.Table(columns=["id", "plot"])
                for i, file_name in enumerate(file_list):
                    file_path = str(self.pred_dist_path / str(file_name))
                    table.add_data(i, wandb.Image(file_path))

                wandb.log({"table_of_prediction_distributions": table})

        wandb.finish()

        end_time = time.perf_counter()
        run_time = (end_time - self.start_time) / 60
        if not is_failed:
            logger.success(f"Successfully finished. Execution time: {run_time} minutes.")
        else:
            logger.error(
                f"Successfully halted at {self.surrogate_budget} surrogate iteration. "
                f"Execution time: {run_time} minutes."
            )
