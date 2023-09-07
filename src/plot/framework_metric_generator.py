import json

import matplotlib.axes
import seaborn as sns
import pandas as pd

# from docutils.nodes import title
from matplotlib import pyplot as plt
from typing import Dict, Union, List, Set, Any, Tuple
from pathlib import Path
import numpy as np
import os
from functools import partial

from src.benchmarks.lcbench import LCBench
from src.benchmarks.taskset import TaskSet
from src.benchmarks.hyperbo import PD1
from src.benchmarks.synthetic_mf import SyntheticMFBench
from src.benchmarks.nanogptbench import NanoGPTBench


class FrameworkMetricsGenerator:
    instance = None

    def __init__(self, path: Path, method_names: List[str] = None, benchmark_names: List[str] = None,
                 dataset_names: List[str] = None):
        self.result_path: Path = path
        self._all_result_data: pd.DataFrame = pd.DataFrame()
        self._metric_data = {}
        self.tag_names: Set[str] = set()
        self.req_methods: List[str] = method_names
        self.req_benchmarks: List[str] = benchmark_names
        self.req_datasets: List[str] = dataset_names
        self.benchmarks = {}
        self.benchmark_infos: Dict[str, Dict[str, Any]] = {
            'lcbench': {
                'class': LCBench,
                'kwargs': {
                    'path_to_json_files': str(Path('./lc_bench/results/data_2k.json')),
                    'dataset_name': 'credit_g'
                }
            },
            'lcbench_mini': {
                'class': LCBench,
                'kwargs': {
                    'path_to_json_files': str(Path('./lc_bench/results/lcbench_airlines.json')),
                    'dataset_name': 'airlines'
                }
            },
            'taskset': {
                'class': TaskSet,
                'kwargs': {
                    'path_to_json_files': str(Path('./data/taskset')),
                    'dataset_name': 'FixedTextRNNClassification_imdb_patch32_GRU128_bs128'
                }
            },
            'pd1': {
                'class': PD1,
                'kwargs': {
                    'path_to_json_files': str(Path('pd1')),
                    'dataset_name': 'uniref50_transformer_batch_size_128'
                }
            },
            'synthetic_mf': {
                'class': SyntheticMFBench,
                'kwargs': {
                    'path_to_json_files': str(Path('synthetic_mf')),
                    'dataset_name': 'park2_3'
                }
            },
            'nanogpt': {
                'class': NanoGPTBench,
                'kwargs': {
                    'path_to_json_files': str(Path('data', 'nanogpt')),
                    'dataset_name': 'openwebtext'
                }
            }
        }
        self.result_combinations: List[Tuple[str, str, str, str]] = []

        self._load_results()

    def _load_benchmark(self, benchmark_name):
        if benchmark_name not in self.benchmarks:
            self.benchmarks[benchmark_name] = self.benchmark_infos[benchmark_name]['class'](
                **self.benchmark_infos[benchmark_name]['kwargs'])
        return self.benchmarks[benchmark_name]

    def get_regret(self):
        regret_combinations = set()
        for benchmark_name, dataset_name, _, _ in self.result_combinations:
            regret_combinations.add((benchmark_name, dataset_name))
        column_list = []
        for benchmark_name, dataset_name in regret_combinations:
            benchmark = self._load_benchmark(benchmark_name)
            benchmark = benchmark.set_dataset_name(dataset_name)
            # incumbent_best_performance = benchmark.get_best_performance()
            incumbent_best_performance = 16.0

            baseline_incumbent_curve = \
                self._all_result_data.loc[:, ('curve', benchmark_name, dataset_name, slice(None), slice(None))]
            if not benchmark.is_minimize:
                regret = incumbent_best_performance - baseline_incumbent_curve
            else:
                regret = baseline_incumbent_curve - incumbent_best_performance

            column_list.append(regret)

        regret_all = pd.concat(column_list, axis=1)
        regret_all.columns = regret_all.columns.set_levels(['regret'], level='tag')

        return regret_all

    def get_metric(self, name: str):
        if name in self._metric_data:
            return self._metric_data[name]

        if name == 'regret':
            result = self.get_regret()
        else:
            raise NotImplementedError
        self._metric_data[name] = result

        return result

    @staticmethod
    def get_instance(path: Path, method_names: List[str] = None, benchmark_names: List[str] = None,
                     dataset_names: List[str] = None):
        if not FrameworkMetricsGenerator.instance:
            FrameworkMetricsGenerator.instance = FrameworkMetricsGenerator(path=path, benchmark_names=benchmark_names,
                                                                           dataset_names=dataset_names,
                                                                           method_names=method_names)
        return FrameworkMetricsGenerator.instance

    def _load_results(self):
        seed_count = 10
        seeds = range(seed_count)

        dev_ds = ['blood-transfusion-service-center', 'vehicle',
                  'FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128',
                  'FixedTextRNNClassification_imdb_patch32_GRU128_bs128']

        self.result_combinations = []
        subdirectory_paths = self._get_all_result_files(path=self.result_path)
        for subdirectory_path in subdirectory_paths:
            sub_path: Path = Path(subdirectory_path)
            sub_benchmark_name = sub_path.parts[1].strip()
            sub_method_name = sub_path.parts[2].strip()
            split_index = sub_path.stem.rfind('_')
            sub_dataset_name = sub_path.stem[:split_index].strip()
            sub_repeat_nr = sub_path.stem[split_index + 1:].strip()
            # if sub_dataset_name not in dev_ds:
            #     continue
            if (self.req_benchmarks is not None and sub_benchmark_name not in self.req_benchmarks) or \
                (self.req_methods is not None and sub_method_name not in self.req_methods) or \
                (self.req_datasets is not None and sub_dataset_name not in self.req_datasets):
                continue
            self.result_combinations.append((sub_benchmark_name, sub_dataset_name, sub_method_name, sub_repeat_nr))

        column_list = []
        for benchmark_name, dataset_name, method_name, repeat_nr in self.result_combinations:
            result_file_path = self.result_path / benchmark_name / method_name / f'{dataset_name}_{repeat_nr}.json'

            if not result_file_path.exists():
                print(f"Warning: Result File Does Not Exist {benchmark_name=} {method_name=} "
                      f"{dataset_name=} {repeat_nr=}")
                continue
            with open(result_file_path, 'r') as fp:
                result_info = json.load(fp)

            for tag, y in result_info.items():
                tag = tag.strip()
                if tag not in self.tag_names:
                    self.tag_names.add(tag)
                # if tag == "epochs":
                #     assert y[0] == 1, "Regret calculation not implemented for epochs not equal to one."
                column = pd.Series(y, name=(tag, benchmark_name, dataset_name, method_name, repeat_nr))
                column_list.append(column)

        index_names = ["tag", "benchmark", "dataset", "method", "repeat_nr"]
        self._all_result_data: pd.DataFrame = pd.concat(column_list, axis=1)
        self._all_result_data.columns.set_names(index_names, inplace=True)
        self._all_result_data = self._all_result_data.dropna(how='all', axis=1)

    @property
    def all_result_data(self):
        return self._all_result_data

    def get_tag_names(self) -> List[str]:
        return list(self.tag_names)

    def align_x_and_y(self, x_df, y_df):
        unique_x = pd.unique(x_df.values.ravel('K'))  # Get unique x values across the entire x dataframe
        unique_x = unique_x[~np.isnan(unique_x)]  # Exclude NaNs
        unique_x.sort()  # Ensure the x values are sorted

        aligned_y_df = pd.DataFrame(columns=y_df.columns)

        for x_col, y_col in zip(x_df.columns, y_df.columns):
            x_values = x_df[x_col].dropna()
            y_values = y_df[y_col]

            aligned_y = []
            y_idx = 0

            for x in unique_x:
                # Keep incrementing y_idx while x values are equal
                while y_idx < len(x_values) and x == x_values.iloc[y_idx]:
                    y_idx += 1

                # Append the last matching y value
                aligned_y.append(y_values.iloc[y_idx - 1] if y_idx > 0 else y_values.iloc[0])

            aligned_y_df[y_col] = aligned_y

        unique_x = pd.Series(unique_x)

        return unique_x, aligned_y_df

    def get_result_data(self, aggregate_level=None, metric=None, x_metric=None, x_max=None):

        if metric in self.tag_names:
            tag_data = self._all_result_data.loc[:, (metric, slice(None), slice(None), slice(None), slice(None))]
        elif metric is None:
            tag_data = self._all_result_data
        # elif metric == 'regret':
        #     tag_data = self.get_metric(metric)
        elif metric == 'best_regret':
            tag_data = self._all_result_data.loc[:, ('regret', slice(None), slice(None), slice(None), slice(None))]
            tag_data = tag_data.iloc[-1:, :]
        elif metric == 'first_curve':
            tag_data = self._all_result_data.loc[:, ('curve', slice(None), slice(None), slice(None), slice(None))]
            tag_data = tag_data.iloc[:1, :]
        else:
            raise NotImplementedError

        x_tag_data = None
        if x_metric in self.tag_names:
            x_tag_data = self._all_result_data.loc[:, (x_metric, slice(None), slice(None), slice(None), slice(None))]
        elif x_metric == 'epochs':
            #     x_tag_data = self._all_result_data.index
            pass
        else:
            raise NotImplementedError

        if x_metric is None:
            if aggregate_level is None:
                tag_groups = tag_data.groupby(by=['tag', 'benchmark', 'dataset', 'repeat_nr'], axis=1)
                for ((tag, benchmark_name, dataset_name, repeat_nr), data) in tag_groups:
                    data = data.droplevel(["tag", "benchmark", "dataset", "repeat_nr"], axis=1)
                    if self.req_methods is not None:
                        data = data[self.req_methods]
                    yield (tag, benchmark_name, dataset_name, repeat_nr), data
                return tag_data
            elif aggregate_level == "dataset":
                tag_groups = tag_data.groupby(by=['tag', 'benchmark', 'dataset'], axis=1)
                for (tag, benchmark_name, dataset_name), dataset_data in tag_groups:
                    grouped_data = dataset_data.groupby(by=["method"], axis=1)
                    mean_data, std_data = grouped_data.mean(), grouped_data.std()
                    if self.req_methods is not None:
                        mean_data, std_data = mean_data[self.req_methods], std_data[self.req_methods]
                    yield (tag, benchmark_name, dataset_name), mean_data, std_data
            elif aggregate_level == "benchmark":
                tag_groups = tag_data.groupby(by=['tag', 'benchmark'], axis=1)
                for (tag, benchmark_name), dataset_data in tag_groups:
                    grouped_data = dataset_data.groupby(by=["method"], axis=1)
                    mean_data, std_data = grouped_data.mean(), grouped_data.std()
                    if self.req_methods is not None:
                        mean_data, std_data = mean_data[self.req_methods], std_data[self.req_methods]
                    yield (tag, benchmark_name), mean_data, std_data
            else:
                raise NotImplementedError
        else:
            if aggregate_level is None:
                tag_groups = tag_data.groupby(by=['tag', 'benchmark', 'dataset', 'repeat_nr'], axis=1)
                for ((tag, benchmark_name, dataset_name, repeat_nr), data) in tag_groups:
                    data = data.droplevel(["tag", "benchmark", "dataset", "repeat_nr"], axis=1)
                    if self.req_methods is not None:
                        data = data[self.req_methods]
                    yield (tag, benchmark_name, dataset_name, repeat_nr), data
                return tag_data
            elif aggregate_level == "dataset":
                tag_groups = tag_data.groupby(by=['tag', 'benchmark', 'dataset'], axis=1)
                x_tag_groups = x_tag_data.groupby(by=['tag', 'benchmark', 'dataset'], axis=1)

                for ((tag, benchmark_name, dataset_name), dataset_data), x_data in zip(tag_groups, x_tag_groups):
                    grouped_data = dataset_data.groupby(by=["method"], axis=1)
                    x_dataset_data = x_data[-1]
                    x_grouped_data = x_dataset_data.groupby(by=["method"], axis=1)
                    aligned_x_df = pd.DataFrame()
                    aligned_y_mean_df = pd.DataFrame()
                    aligned_y_std_df = pd.DataFrame()
                    for (group_name, group_df), (x_group_name, x_group_df) in zip(grouped_data, x_grouped_data):
                        group_name = group_name[-1]
                        aligned_x_data, aligned_y_data = self.align_x_and_y(x_df=x_group_df, y_df=group_df)
                        if len(aligned_x_data) > len(aligned_x_df):
                            aligned_x_df = aligned_x_df.reindex(index=range(len(aligned_x_data)))
                            aligned_y_mean_df = aligned_y_mean_df.reindex(index=range(len(aligned_y_data)))
                            aligned_y_std_df = aligned_y_std_df.reindex(index=range(len(aligned_y_data)))
                        aligned_x_df[group_name] = aligned_x_data
                        aligned_y_mean_df[group_name] = aligned_y_data.mean(axis=1)
                        aligned_y_std_df[group_name] = aligned_y_data.std(axis=1)
                    x_mean_data, mean_data = self.align_x_and_y(x_df=aligned_x_df, y_df=aligned_y_mean_df)
                    x_std_data, std_data = self.align_x_and_y(x_df=aligned_x_df, y_df=aligned_y_std_df)
                    if self.req_methods is not None:
                        mean_data, std_data = mean_data[self.req_methods], std_data[self.req_methods]
                    if x_max is not None and (x_mean_data > x_max).any():
                        truncate_index = x_mean_data[x_mean_data > x_max].index[0]
                        x_mean_data = x_mean_data.iloc[:truncate_index]
                        mean_data = mean_data.iloc[:truncate_index]
                        truncate_index = x_std_data[x_std_data > x_max].index[0]
                        x_std_data = x_std_data.iloc[:truncate_index]
                        std_data = std_data.iloc[:truncate_index]
                    plot_data = mean_data
                    plot_data['x'] = x_mean_data
                    yield (tag, benchmark_name, dataset_name), plot_data, std_data
            elif aggregate_level == "benchmark":
                tag_groups = tag_data.groupby(by=['tag', 'benchmark'], axis=1)
                x_tag_groups = x_tag_data.groupby(by=['tag', 'benchmark'], axis=1)
                aligned_x_df = pd.DataFrame()
                aligned_y_mean_df = pd.DataFrame()
                aligned_y_std_df = pd.DataFrame()
                for ((tag, benchmark_name), dataset_data), x_data in zip(tag_groups, x_tag_groups):
                    grouped_data = dataset_data.groupby(by=["method"], axis=1)
                    x_dataset_data = x_data[-1]
                    x_grouped_data = x_dataset_data.groupby(by=["method"], axis=1)
                    aligned_x_df = pd.DataFrame()
                    aligned_y_mean_df = pd.DataFrame()
                    aligned_y_std_df = pd.DataFrame()
                    for (group_name, group_df), (x_group_name, x_group_df) in zip(grouped_data, x_grouped_data):
                        group_name = group_name[-1]
                        aligned_x_data, aligned_y_data = self.align_x_and_y(x_df=x_group_df, y_df=group_df)
                        if len(aligned_x_data) > len(aligned_x_df):
                            aligned_x_df = aligned_x_df.reindex(index=range(len(aligned_x_data)))
                            aligned_y_mean_df = aligned_y_mean_df.reindex(index=range(len(aligned_y_data)))
                            aligned_y_std_df = aligned_y_std_df.reindex(index=range(len(aligned_y_data)))
                        aligned_x_df[group_name] = aligned_x_data
                        aligned_y_mean_df[group_name] = aligned_y_data.mean(axis=1)
                        aligned_y_std_df[group_name] = aligned_y_data.std(axis=1)
                    x_mean_data, mean_data = self.align_x_and_y(x_df=aligned_x_df, y_df=aligned_y_mean_df)
                    x_std_data, std_data = self.align_x_and_y(x_df=aligned_x_df, y_df=aligned_y_std_df)
                    if self.req_methods is not None:
                        mean_data, std_data = mean_data[self.req_methods], std_data[self.req_methods]
                    if x_max is not None and (x_mean_data > x_max).any():
                        truncate_index = x_mean_data[x_mean_data > x_max].index[0]
                        x_mean_data = x_mean_data.iloc[:truncate_index]
                        mean_data = mean_data.iloc[:truncate_index]
                        truncate_index = x_std_data[x_std_data > x_max].index[0]
                        x_std_data = x_std_data.iloc[:truncate_index]
                        std_data = std_data.iloc[:truncate_index]
                    plot_data = mean_data
                    plot_data['x'] = x_mean_data
                    yield (tag, benchmark_name), plot_data, std_data

    def _get_all_result_files(self, path: Path):
        subdirectories = []
        for dirpaths, dirnames, filenames in os.walk(str(path)):
            parent_folder = os.path.basename(dirpaths)
            if parent_folder != 'configuration':
                for filename in filenames:
                    if filename.endswith('.json'):
                        subdirectories.append(os.path.join(dirpaths, filename))
        return subdirectories

    def calc_regret_v0(self):
        baseline_epochs = []
        baseline_incumbent_curves = []
        seen = set()
        results = []
        for benchmark_name, dataset_name, method_name, _ in self.result_combinations:
            self._load_benchmark(benchmark_name=benchmark_name)
            identifier = (benchmark_name, dataset_name, method_name, '0')
            if identifier in seen:
                continue
            for repeat_nr in range(10):
                repeat_nr = str(repeat_nr)
                identifier = (benchmark_name, dataset_name, method_name, repeat_nr)
                seen.add(identifier)
                repeat_epochs_cost = []
                configs_evaluated = dict()
                baseline_incumbent_curve = self._all_result_data.loc[:,
                                           ('curve', benchmark_name, dataset_name, method_name, repeat_nr)]

                # if len(baseline_incumbent_curve) < 1000:
                #     continue
                # elif len(baseline_incumbent_curve) > 1000:
                #     baseline_incumbent_curve = baseline_incumbent_curve[0:1000]
                baseline_incumbent_curves.append(baseline_incumbent_curve)

                evaluated_hps = self._all_result_data.loc[:,
                                ('hp', benchmark_name, dataset_name, method_name, repeat_nr)]
                budgets_evaluated = self._all_result_data.loc[:,
                                    ('epochs', benchmark_name, dataset_name, method_name, repeat_nr)]

                for evaluated_hp, budget_evaluated in zip(evaluated_hps, budgets_evaluated):
                    if evaluated_hp in configs_evaluated:
                        budgets = configs_evaluated[evaluated_hp]
                        max_budget = max(budgets)
                        cost = budget_evaluated - max_budget
                        repeat_epochs_cost.append(cost)
                        configs_evaluated[evaluated_hp].append(budget_evaluated)
                    else:
                        repeat_epochs_cost.append(budget_evaluated)
                        configs_evaluated[evaluated_hp] = [budget_evaluated]

                baseline_epochs.append(repeat_epochs_cost)

            if len(baseline_incumbent_curves) == 0:
                return [], [], []

            mean_cost_values = []
            try:
                for curve_point in range(0, len(baseline_incumbent_curves[0])):
                    mean_config_point_values = []
                    for curve_nr in range(0, len(baseline_epochs)):
                        config_curve = baseline_epochs[curve_nr]
                        if len(config_curve) > curve_point:
                            mean_config_point_values.append(config_curve[curve_point])
                        else:
                            continue
                    if len(mean_config_point_values) > 0:
                        mean_cost_values.append(np.mean(mean_config_point_values))
            except Exception:
                return [], [], []

            if len(mean_cost_values) < len(baseline_incumbent_curves[0]):
                mean_cost_values = [1 for _ in range(1, len(baseline_incumbent_curves[0]) + 1)]
            iteration_cost = []

            total_iteration_cost = 0
            for iteration_cost_value in mean_cost_values:
                total_iteration_cost += iteration_cost_value
                iteration_cost.append(total_iteration_cost)

            baseline_incumbent_curve = np.mean(baseline_incumbent_curves, axis=0)
            baseline_incumbent_std = np.std(baseline_incumbent_curves, axis=0)

            incumbent_best_performance = self.benchmarks[benchmark_name].get_best_performance()

            worst_performance = self.benchmarks[benchmark_name].get_worst_performance()

            if benchmark_name == 'lcbench':
                # convert the incumbent iteration performance to the normalized regret for every iteration
                baseline_incumbent_curve = [incumbent_best_performance - incumbent_it_performance
                                            for incumbent_it_performance in baseline_incumbent_curve]
            elif benchmark_name == 'taskset':
                if method_name != 'power_law' and method_name != 'smac':
                    baseline_incumbent_curve = [worst_performance - curve_element for curve_element in
                                                baseline_incumbent_curve]

                baseline_incumbent_curve = [(incumbent_it_performance - incumbent_best_performance)  # / gap_performance
                                            for incumbent_it_performance in baseline_incumbent_curve]
            results.append((benchmark_name, dataset_name, method_name, iteration_cost, baseline_incumbent_curve,
                            baseline_incumbent_std))
        return results
