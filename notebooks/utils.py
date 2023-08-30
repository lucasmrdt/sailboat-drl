import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib.patches import Rectangle
from glob import glob
from collections import defaultdict
from typing import Callable


def get_dfs(name):
    if not name.endswith('.csv'):
        pattern = f'../runs/{name}/*.csv'
    else:
        pattern = f'../runs/{name}'
    files = list(glob(pattern, recursive=True))
    dfs = {}
    for file in tqdm(files, desc=name, leave=False):
        try:
            df = pd.read_csv(file)
            dfs[file] = df
        except:
            pass
    return dfs


def _draw_trajectories_of_name(name, ax, color='C0'):
    if isinstance(name, tuple):
        name, label = name
    else:
        label = name
    files = list(enumerate(glob(f'../runs/{name}/**/*.csv')))
    nb_fails = 0
    ax.plot([], [], alpha=1, color=color, label=label)
    for i, file in tqdm(files, desc=name, leave=False):
        try:
            df = pd.read_csv(file)
            ax.plot(df["obs/p_boat/0"], df["obs/p_boat/1"],
                    alpha=0.4, color=color, linewidth=1.5)
        except:
            nb_fails += 1
        # ax.scatter(df["obs/p_boat/0"], df["obs/p_boat/1"],
        #            alpha=0.2, color=color, s=.1)
    if nb_fails > 0:
        print(f'Failed to load {nb_fails} files for {name}')


def draw_trajectories(names, xte_delta=10, hide_legend=False):
    fig, ax = plt.subplots(figsize=(8, 3), dpi=200)

    ax.plot([0, 200], [0, 0], 'k--', label='Reference path')
    ax.plot([0, 200], [xte_delta, xte_delta], 'r:', label='XTE > 10m')
    ax.plot([0, 200], [-xte_delta, -xte_delta], 'r:')
    for i, name in enumerate(names):
        _draw_trajectories_of_name(name, ax=ax, color=f'C{i}')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    # ax.add_patch(Rectangle((0, xte_delta), 100, xte_delta, facecolor='red',
    #              alpha=0.2, label='XTE > 10m'))
    # ax.add_patch(Rectangle((0, -xte_delta), 100, -xte_delta, facecolor='red',
    #              alpha=0.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-xte_delta * 1.2, xte_delta * 1.2)
    if not hide_legend:
        fig.legend(loc='upper center',
                   bbox_to_anchor=(0.5, -.05),
                   ncol=len(names) + 2)

# elif metric == 'nb_step_by_sec':
#     max_time = int(df["relative_time"].iloc[-1])
#     sums, bins = np.histogram(df["relative_time"][:-1],
#                                 bins=range(max_time))
#     last_idx = np.where(sums > 0)[0][-1]
#     score = sums[:last_idx]
#     df_key = bins[:last_idx]


def get_metric(name, metric, plot_type='mean+std'):
    if not name.endswith('.csv'):
        pattern = f'../runs/{name}/*.csv'
    else:
        pattern = f'../runs/{name}'
    files = list(glob(pattern, recursive=True))

    scores_by_key = defaultdict(list)
    nb_fails = 0
    for file in tqdm(files, desc=name, leave=False):
        try:
            df = pd.read_csv(file)
            if len(df) == 0:
                continue
            if isinstance(metric, Callable):
                keys, scores = metric(file, df)
            else:
                scores = df[metric].values
                keys = np.arange(len(scores))
            if isinstance(scores, np.ndarray) and isinstance(keys, np.ndarray):
                for k, s in zip(keys, scores):
                    scores_by_key[k].append(s)
            else:
                key = keys
                scores_by_key[key].append(scores)
        except Exception as e:
            nb_fails += 1

    if nb_fails > 0:
        print(f'Failed to load {nb_fails} files for {name}')
    keys = list(scores_by_key.keys())
    if plot_type == 'mean+std':
        scores_mean = np.array([np.mean(scores_by_key[k]) for k in keys])
        scores_std = np.array([np.std(scores_by_key[k]) for k in keys])
        return keys, scores_mean, scores_std
    elif plot_type == 'mean':
        scores_mean = np.array([np.mean(scores_by_key[k]) for k in keys])
        return keys, scores_mean
    elif plot_type == 'sum':
        scores_sum = np.array([sum(scores_by_key[k]) for k in keys])
        return keys, scores_sum
    else:
        raise ValueError(f'Unknown plot_type {plot_type}')


def plot_metric(names, metric, ax=None, x_label='Timesteps', y_label=None, plot_type='mean+std', hide_legend=False):
    if not isinstance(names, list):
        names = [names]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4), dpi=150)
    fig = ax.figure
    nb_ploted = 0
    for i, name in enumerate(names):
        if isinstance(name, tuple):
            name, label = name
        else:
            label = name
        if plot_type == 'mean+std':
            keys, scores_mean, scores_std = get_metric(
                name, metric, plot_type=plot_type)
            if not len(keys):
                continue
            ax.plot(keys, scores_mean,
                    label=label if not hide_legend else None, color=f'C{i}')
            ax.fill_between(keys, scores_mean - scores_std, scores_mean + scores_std,
                            alpha=0.2, color=f'C{i}')
        elif plot_type == 'mean':
            keys, scores_mean = get_metric(
                name, metric, plot_type=plot_type)
            if not len(keys):
                continue
            ax.plot(keys, scores_mean,
                    label=label if not hide_legend else None, color=f'C{i}')
        elif plot_type == 'sum':
            keys, scores_sum = get_metric(
                name, metric, plot_type=plot_type)
            if not len(keys):
                continue
            ax.bar(keys, scores_sum, label=label, color=f'C{i}')
        else:
            raise ValueError(f'Unknown plot_type {plot_type}')
        nb_ploted += 1

    ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(metric)
    if not hide_legend:
        fig.legend(loc='upper center',
                   bbox_to_anchor=(0.5, -.05),
                   ncol=min(3, len(names)))
    return ax
