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


def add_plot_legend(ax: matplotlib.axes.Axes, n: int):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    handles, legends = ax.get_legend_handles_labels()
    ax.legend(handles, legends, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize="7",
              fancybox=True, shadow=True, ncol=n)


def plot_line(data: pd.DataFrame, x_label: str, y_label: str, title: str, path: Union[Path, str],
              std_data: pd.DataFrame = None, **kwargs):
    plt.clf()
    p = sns.lineplot(data=data)

    plot_std = kwargs['plot_std'] if 'plot_std' in kwargs else True
    if std_data is not None and plot_std:
        for col in data.columns:
            p.axes.fill_between(data.index, data[col] + std_data[col], data[col] - std_data[col], alpha=0.3)

    if 'x_log' in kwargs and kwargs['x_log']:
        plt.xscale('log')
    if 'y_log' in kwargs and kwargs['y_log']:
        plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(title)
    add_plot_legend(p.axes, len(data.columns))
    # plt.tight_layout()
    plt.savefig(path, dpi=200)
    # plt.show()
