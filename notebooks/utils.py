import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict


def _draw_trajectories_of_name(name, ax, color='C0'):
    for i, file in enumerate(glob(f'../runs/{name}/eval-*deg/**/*.csv')):
        df = pd.read_csv(file)
        ax.plot(df["obs/p_boat/0"], df["obs/p_boat/1"],
                alpha=0.3, color=color, linewidth=1, label=name if i == 0 else None)
        # ax.scatter(df["obs/p_boat/0"], df["obs/p_boat/1"],
        #            alpha=0.2, color=color, s=.1)


def draw_trajectories(names):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.plot([0, 200], [0, 0], 'k--', label='Reference path')

    for i, name in enumerate(names):
        print(f'Drawing trajectories of {name}')
        _draw_trajectories_of_name(name, ax=ax, color=f'C{i}')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(0, 125)
    fig.legend(loc='upper center',
               bbox_to_anchor=(0.5, -.05),
               ncol=min(3, len(names) + 1))


def get_metric(name, metric='cum_vmc', key=lambda df: 0):
    scores_by_key = defaultdict(list)
    for file in glob(f'../runs/{name}/eval-*deg/**/*.csv'):
        df = pd.read_csv(file)
        df_key = key(df)
        if metric == 'cum_vmc':
            score = df["obs/cum_obs/vmc/0"].iloc[-1]
        elif metric == 'mean_xte':
            score = df["obs/xte/0"].abs().mean()
        elif metric == 'dist':
            score = df["obs/cum_obs/gain_dist/0"].iloc[-1]
        elif metric == 'duration':
            score = df["relative_time"].iloc[-1]
        elif metric == 'nb_step_by_sec':
            counts, bins = np.histogram(df["relative_time"][:-1],
                                        bins=range(360))
            score = counts
            df_key = bins[:-1]
        else:
            score = df[metric].values
            df_key = np.arange(len(score))
        if isinstance(score, np.ndarray) and isinstance(df_key, np.ndarray):
            for k, s in zip(df_key, score):
                scores_by_key[k].append(s)
        else:
            scores_by_key[df_key].append(score)

    keys = list(scores_by_key.keys())
    scores_mean = np.array([np.mean(scores_by_key[k]) for k in keys])
    scores_std = np.array([np.std(scores_by_key[k]) for k in keys])
    return keys, scores_mean, scores_std
