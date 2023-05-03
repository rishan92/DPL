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
import matplotlib as mpl

from src.plot.framework_metric_generator import FrameworkMetricsGenerator
from src.plot.utils import plot_line, plot_scatter


# from plots.normalized_regret import plot_all_baselines_epoch_performance, plot_all_baselines_time_performance


# mpl.rc_file_defaults()
# sns.reset_orig()
# #
# sns.set_palette(sns.color_palette('deep'))
# sns.set_style("white")


# sns.set()


def plot_results(result_path: Path, x_metric: str = None, y_metric: str = None, aggregate_level: str = None,
                 plot_path: Path = Path("."), method_names: List[str] = None, benchmark_names: List[str] = None,
                 dataset_names: List[str] = None, **kwargs):
    plot_path.mkdir(parents=True, exist_ok=True)
    result_loader = FrameworkMetricsGenerator.get_instance(path=result_path, benchmark_names=benchmark_names,
                                                           dataset_names=dataset_names, method_names=method_names)

    x_metric = 'epochs' if x_metric is None else x_metric
    assert x_metric in ['epochs', 'first_curve']
    if 'x_label' not in kwargs:
        kwargs['x_label'] = x_metric
    if 'y_label' not in kwargs:
        kwargs['y_label'] = y_metric

    x_data_iter = result_loader.get_result_data(aggregate_level=aggregate_level, metric=x_metric)
    result_data_iter = result_loader.get_result_data(aggregate_level=aggregate_level, metric=y_metric)
    if x_metric == 'first_curve' and y_metric == 'best_regret' and aggregate_level == 'dataset':
        data = pd.DataFrame(columns=['first_curve', 'best_regret'])
        for i, ((tag, benchmark_name, dataset_name), mean_data, std_data) in enumerate(x_data_iter):
            data.loc[i, 'first_curve'] = mean_data.iloc[0, 0]
        for i, ((tag, benchmark_name, dataset_name), mean_data, std_data) in enumerate(result_data_iter):
            data.loc[i, 'best_regret'] = mean_data.iloc[0, 0]
        out_path = plot_path / f"performance_scatter.png"
        if 'title' not in kwargs:
            kwargs['title'] = "Initial performance vs best regret"
        plot_scatter(data=data, x='first_curve', y='best_regret', path=out_path, **kwargs)
    else:
        if aggregate_level is None:
            for (tag, benchmark_name, dataset_name, repeat_nr), data in result_data_iter:
                out_path = plot_path / f"{benchmark_name}_{dataset_name}_{repeat_nr}_{tag}.png"
                if 'title' not in kwargs:
                    kwargs['title'] = f"{benchmark_name}_{dataset_name}_{repeat_nr}_{tag}"
                plot_line(ydata=data, path=out_path, **kwargs)
        elif aggregate_level == 'dataset':
            for (tag, benchmark_name, dataset_name), mean_data, std_data in result_data_iter:
                out_path = plot_path / f"{benchmark_name}_{dataset_name}_aggregated_{tag}.png"
                if 'title' not in kwargs:
                    kwargs['title'] = f"{benchmark_name}_{dataset_name}_{tag}"
                plot_line(ydata=mean_data, std_data=std_data, path=out_path, **kwargs)
        elif aggregate_level == 'benchmark':
            for (tag, benchmark_name), mean_data, std_data in result_data_iter:
                out_path = plot_path / f"{benchmark_name}_aggregated_{tag}.png"
                title = f"{benchmark_name}_{tag}"
                if 'title' not in kwargs:
                    kwargs['title'] = f"{benchmark_name}_{tag}"
                plot_line(ydata=mean_data, std_data=std_data, path=out_path, **kwargs)


def main():
    result_path = Path("./Results_PL")
    plot_path = Path("./Results_PL")

    project_folder = Path(".")
    dataset_files_path = Path("./bash_scripts")

    method_names = ['power_law', 'dyhpo', 'dehb', 'random']
    method_names = ['power_law_benchmark', 'power_law_complex6', 'power_law_complex5', 'dyhpo_benchmark', 'dehb',
                    'random']
    benchmark_names = ['taskset', 'lcbench']
    benchmark_names = ['lcbench']
    dataset_names = None
    # dataset_names = ['FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128',
    #                  'FixedTextRNNClassification_imdb_patch32_GRU128_bs128']
    # dataset_names = ['airlines']

    # plot_all_baselines_time_performance(project_folder, result_path)
    # plot_all_baselines_epoch_performance(project_folder, result_path)
    # # plot_config_distribution(benchmark_data_path, surrogate_results_path, benchmark_name)

    plot_results_f = partial(plot_results, method_names=method_names, benchmark_names=benchmark_names,
                             dataset_names=dataset_names, result_path=result_path, plot_path=plot_path)

    # plot_results_f(y_metric=None, x_metric='epochs', aggregate_level=None)
    # plot_results_f(y_metric='curve', x_metric='epochs', aggregate_level=None)
    # plot_results_f(y_metric='curve', x_metric='epochs', aggregate_level='dataset')
    # plot_results_f(y_metric='curve', x_metric='epochs', aggregate_level='benchmark')
    # plot_results_f(y_metric='regret', x_metric='epochs', aggregate_level=None)
    plot_results_f(y_metric='regret', x_metric='epochs', aggregate_level='dataset', y_log=True, plot_std=False)
    plot_results_f(y_metric='regret', x_metric='epochs', aggregate_level='benchmark', y_log=True, plot_std=False)
    # plot_results_f(y_metric='best_regret', x_metric='first_curve', aggregate_level='dataset', plot_std=False)


if __name__ == "__main__":
    main()
