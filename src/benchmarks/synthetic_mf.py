import math
import time
from collections import OrderedDict
from typing import List
import numpy as np
# from numpy.typing import NDArray
import pandas as pd
from pathlib import Path
import json
import ConfigSpace as CS
from ConfigSpace import Float, Integer, OrdinalHyperparameter, Categorical
from typing import List, Union, Dict, Tuple, Optional

from src.benchmarks.base_benchmark import BaseBenchmark
from src.history.fidelity_manager import FidelityManager


class SyntheticMFBench(BaseBenchmark):
    nr_hyperparameters = 2000

    param_space = None
    max_budget = 51
    min_budget = 1
    # max_budgets = None
    # min_budgets = None
    max_steps = 10

    hp_names = None
    fidelity_names = None
    log_indicator = None
    fidelity_space = None
    fidelity_interval = None
    # fidelity_curve_points = None
    # fidelity_steps = None

    # if the best value corresponds to a lower value
    minimization_metric = False

    scenario_metric_is_minimize = {
        "hartmann6_4": False,
        "branin": False,
        "borehole_6": False,
        "park2_3": False,
    }

    is_metric_best_end = {
        'iter': False,
        'subsample': True,
    }

    def __init__(self, path_to_json_files: Path = "", dataset_name: str = "", seed=None):
        super().__init__(path_to_json_files, seed=seed)
        self.dataset_name = dataset_name
        self.dataset_names = None
        self.categorical_indicator = None
        self.benchmark_data = None
        self.config_ids = {}
        self.mf_objective = None
        self.mf_cost = None
        self.benchmark_config = None
        self.benchmark_config_pd = None
        self.param_dimensions = None
        self.fidelity_dimensions = None
        self.objective_function_input_config = {}
        self.objective_function_input_fidelity = {}
        self.root_save_path = Path('./data/synthetic_mf')
        self.benchmark_results = None
        self.fidelity_manager: Optional[FidelityManager] = None
        self.is_new_data_added = False

        self.init_benchmark()

        self.nr_hyperparameters = SyntheticMFBench.nr_hyperparameters

    def init_benchmark(self):
        if self.dataset_name == 'hartmann6_4':
            from dragonfly_synthetic.hartmann6_4.hartmann6_4_mf import objective as mf_objective
            from dragonfly_synthetic.hartmann6_4.hartmann6_4_mf import cost as mf_cost
        elif self.dataset_name == 'branin':
            from dragonfly_synthetic.branin.branin_mf import objective as mf_objective
            from dragonfly_synthetic.branin.branin_mf import cost as mf_cost
        elif self.dataset_name == 'borehole_6':
            from dragonfly_synthetic.borehole_6.borehole_6_mf import objective as mf_objective
            from dragonfly_synthetic.borehole_6.borehole_6_mf import cost as mf_cost
        elif self.dataset_name == 'park2_3':
            from dragonfly_synthetic.park2_3.park2_3_mf import objective as mf_objective
            from dragonfly_synthetic.park2_3.park2_3_mf import cost as mf_cost
        else:
            raise NotImplementedError

        self.mf_objective = mf_objective
        self.mf_cost = mf_cost
        SyntheticMFBench.minimization_metric = SyntheticMFBench.scenario_metric_is_minimize[self.dataset_name]
        self.minimization_metric = SyntheticMFBench.scenario_metric_is_minimize[self.dataset_name]

        config_file = f'./dragonfly_synthetic/{self.dataset_name}/config_mf.json'
        with open(config_file, 'r') as _file:
            config = json.load(_file, object_pairs_hook=OrderedDict)

        config_space, config_names, dimensions = self.dragonfly_config_to_config_space(config["domain"])
        self.param_dimensions = dimensions
        SyntheticMFBench.param_space = self.extract_hyperparameter_info(config_space=config_space)
        # # sort so that categorical columns will be at the end
        # SyntheticMFBench.param_space = OrderedDict(
        #     sorted(SyntheticMFBench.param_space.items(), key=lambda item: item[1][2] == 'str'))
        SyntheticMFBench.hp_names = config_names

        raw_fidelity_space, config_names, dimensions = self.dragonfly_config_to_config_space(config["fidel_space"])
        self.fidelity_dimensions = dimensions
        SyntheticMFBench.fidelity_space = self.extract_hyperparameter_info(config_space=raw_fidelity_space)
        SyntheticMFBench.fidelity_names = config_names

        SyntheticMFBench.fidelity_interval = {}
        SyntheticMFBench.max_budgets = {}
        SyntheticMFBench.min_budgets = {}
        SyntheticMFBench.fidelity_curve_points = {}
        # SyntheticMFBench.fidelity_steps = {}

        # for k, v in SyntheticMFBench.fidelity_space.items():
        #     max_budget = v[1]
        #     if v[2] == 'ord':
        #         SyntheticMFBench.fidelity_curve_points[k] = np.array(v[4])
        #         max_budget = SyntheticMFBench.fidelity_curve_points[k][-1]
        #         interval = SyntheticMFBench.fidelity_curve_points[k][1] - SyntheticMFBench.fidelity_curve_points[k][0]
        #     else:
        #         SyntheticMFBench.fidelity_curve_points[k] = np.around(
        #             np.linspace(v[0], v[1], self.max_steps), decimals=4
        #         )
        #         if v[2] == 'int':
        #             interval = 1
        #             size = v[1] - v[0] + 1
        #             rounded_values = np.round(SyntheticMFBench.fidelity_curve_points[k]).astype(int)
        #             curve_values = np.unique(rounded_values)
        #             curve_values = curve_values.astype(float)
        #             SyntheticMFBench.fidelity_curve_points[k] = curve_values
        #         elif v[2] == 'float':
        #             interval = (v[1] - v[0]) / self.max_steps
        #         else:
        #             raise NotImplementedError
        #
        #     SyntheticMFBench.min_budgets[k] = SyntheticMFBench.fidelity_curve_points[k][0]
        #     SyntheticMFBench.max_budgets[k] = SyntheticMFBench.fidelity_curve_points[k][-1]
        #     SyntheticMFBench.fidelity_interval[k] = interval
        self.categorical_indicator = [v[2] == 'str' for v in SyntheticMFBench.param_space.values()]
        self.categories = [v[4] for v in SyntheticMFBench.param_space.values() if v[2] == 'str']
        SyntheticMFBench.log_indicator = [v[3] for v in SyntheticMFBench.param_space.values()]

        self.benchmark_config = self.generate_hyperparameter_candidates(config_space)
        self.benchmark_config_pd = pd.DataFrame(self.benchmark_config, columns=SyntheticMFBench.hp_names)

        self.benchmark_results = self.load_benchmark()

        self.fidelity_manager = FidelityManager(
            fidelity_space=raw_fidelity_space,
            num_configurations=SyntheticMFBench.nr_hyperparameters,
            max_steps=self.max_steps,
        )

        self.max_budgets = self.fidelity_manager.get_max_fidelity()
        self.min_budgets = self.fidelity_manager.get_min_fidelity()

    def dragonfly_config_to_config_space(self, config):
        all_dimensions = []
        config_names = []
        config_spaces = []

        def add_config_space(space, name, dimensions, **kwargs):
            if dimensions > 1:
                for d in range(dimensions):
                    config_spaces.append(space(name=f'{name}_{d}', **kwargs))
            else:
                config_spaces.append(space(name=f'{name}', **kwargs))

        for k, v in config.items():
            if "dim" in v:
                dimensions = v["dim"] if v["dim"] != "" else 1
                all_dimensions.append(v["dim"] if v["dim"] != "" else 0)
            else:
                dimensions = 1
                all_dimensions.append(0)

            if v["type"] == "discrete_numeric":
                start, step, stop = map(float, v["items"].split(':'))
                discrete_space = np.arange(start, stop, step).tolist()
                space = OrdinalHyperparameter
                kwargs = {'sequence': discrete_space}
            else:
                if v["type"] == "float":
                    space = Float
                    kwargs = {'bounds': (v["min"], v["max"])}
                elif v["type"] == "int":
                    space = Integer
                    kwargs = {'bounds': (v["min"], v["max"])}
                elif v["type"] == "discrete":
                    space = Categorical
                    categories = v["items"].split('-')
                    kwargs = {'items': categories}
                else:
                    raise NotImplementedError

            add_config_space(space=space, name=k, dimensions=dimensions, **kwargs)

        for space in config_spaces:
            config_names.append(space.name)

        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(config_spaces)

        return config_space, config_names, all_dimensions

    def generate_hyperparameter_candidates(self, benchmark: CS.ConfigurationSpace) -> List[Dict]:
        if self.seed is not None:
            benchmark.seed(seed=self.seed)
        configs = benchmark.sample_configuration(SyntheticMFBench.nr_hyperparameters)
        hp_configs = [config.get_dictionary() for config in configs]

        for i, config in enumerate(hp_configs):
            self.objective_function_input_config[i] = self.convert_to_dragonfly_objective_input(
                config=config, ordered_names=self.hp_names, dimensions=self.param_dimensions
            )

        return hp_configs

    def get_hyperparameter_candidates(self) -> np.ndarray:
        return self.benchmark_config_pd

    def get_benchmark_config(self, config_id):
        column_name = self.config_ids[config_id]
        config = {}
        for i, hp_name in enumerate(self.hp_names):
            config[hp_name] = column_name[i]
        return config

    def get_benchmark_objective_function(self, configuration_id, fidelity):
        config = self.benchmark_config[configuration_id]

        row_values = {**config, **fidelity}
        row_values_tuple = tuple([row_values[k] for k in self.benchmark_results.index.names])

        if row_values_tuple in self.benchmark_results.index:
            result = self.benchmark_results.loc[row_values_tuple, 'result']
            return result, 0

        # calculate metrics
        converted_config = self.objective_function_input_config[configuration_id]
        converted_fidelity = self.convert_to_dragonfly_objective_input(
            config=fidelity, ordered_names=self.fidelity_names, dimensions=self.fidelity_dimensions
        )
        start_time = time.perf_counter()
        result_data = self.mf_objective(z=converted_fidelity, x=converted_config)
        end_time = time.perf_counter()
        eval_time = end_time - start_time

        self.benchmark_results.loc[row_values_tuple, 'result'] = result_data
        self.is_new_data_added = True

        return result_data, eval_time

    def convert_to_dragonfly_objective_input(self, config, ordered_names, dimensions):
        config = [config[key] for key in ordered_names]
        index = 0
        result = []
        for d in dimensions:
            if d == 0:
                result.append(config[index])
                index += 1
            else:
                result.append(config[index:index + d])
                index += d
        return result

    def get_curve(self, hp_index: int, budget: Union[int, Dict]) -> Tuple[List[float], List[Dict]]:
        # config_dict = self.benchmark_config[hp_index]

        if isinstance(budget, int):
            budget = {self.fidelity_names[0]: budget}

        valid_curves = []
        for k in self.fidelity_names:
            curve = self.fidelity_manager.fidelity_space[k]
            # if k in SyntheticMFBench.is_metric_best_end and SyntheticMFBench.is_metric_best_end[k]:
            #     valid_curves.append(curve[curve == budget[k]])
            # else:
            valid_curves.append(curve[curve <= budget[k]])

        mesh = np.meshgrid(*valid_curves)

        # Stack the meshgrid to get 2D array of coordinates and reshape it
        fidelity_product = np.dstack(mesh).reshape(-1, len(self.fidelity_names))
        fidelity_dicts = [
            {k: (int(v) if SyntheticMFBench.fidelity_space[k][2] == 'int' else v)
             for k, v in zip(self.fidelity_names, values)}
            for values in fidelity_product]

        # config_dict = [{**config_dict, **dict} for dict in fidelity_dicts]

        metrics = []
        for fidelity_dict in fidelity_dicts:
            metric, _ = self.get_benchmark_objective_function(
                configuration_id=hp_index, fidelity=fidelity_dict
            )
            metrics.append(metric)
        # metrics = self.benchmark.objective_function(configuration=config_dict, fidelity=fidelity_dicts, seed=self.seed)
        # metric = [v[self.metric_name] for v in metrics]

        return metrics, fidelity_dicts

    def get_performance(self, hp_index: int, fidelity_id: Tuple[int]) -> float:
        config_dict = self.benchmark_config[hp_index]

        # if isinstance(budget, List):
        #     fidelity_dict = {k: v for k, v in zip(SyntheticMFBench.fidelity_names, budget)}
        #     config_dict.update(fidelity_dict)
        # elif isinstance(budget, int):
        #     fidelity_dict = {self.fidelity_names[0]: budget}
        # else:
        #     fidelity_dict = budget
        # config_dict.update(fidelity_dict)
        # fidelity_dict = self.fidelity_manager.get_fidelities(fidelity_id, return_dict=True)
        fidelity_dict = dict(zip(self.fidelity_names, fidelity_id))

        metric, eval_time = self.get_benchmark_objective_function(configuration_id=hp_index, fidelity=fidelity_dict)

        return metric, eval_time

    def get_fidelity_manager(self):
        return self.fidelity_manager

    def save_benchmark(self):
        if self.is_new_data_added:
            print("Saving benchmark data")
            self.root_save_path.mkdir(parents=True, exist_ok=True)
            benchmark_path = self.root_save_path / f"{self.dataset_name}_seed_{self.seed}.parquet"
            self.benchmark_results.to_parquet(path=benchmark_path)

    def load_benchmark(self):
        print("Loading benchmark data")
        benchmark_path = self.root_save_path / f"{self.dataset_name}_seed_{self.seed}.parquet"
        if benchmark_path.is_file():
            benchmark = pd.read_parquet(benchmark_path)
        else:
            columns = [*SyntheticMFBench.hp_names, *SyntheticMFBench.fidelity_names, 'result']
            benchmark = pd.DataFrame(columns=columns)
            multi_index_columns = [col for col in benchmark.columns if col != 'result']
            benchmark.set_index(multi_index_columns, inplace=True)

        return benchmark

    def calc_benchmark_stats(self):
        super().calc_benchmark_stats()
        self.save_benchmark()

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name
        return self

    def close(self):
        self.save_benchmark()
