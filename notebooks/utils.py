import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib.patches import Rectangle
from glob import glob
from collections import defaultdict


def _draw_trajectories_of_name(name, ax, color='C0'):
    if isinstance(name, tuple):
        name, label = name
    else:
        label = name
    files = list(enumerate(glob(f'../runs/{name}/**/*.csv')))
    nb_fails = 0
    ax.plot([], [], alpha=1, color=color, label=label)
    for i, file in tqdm(files, desc=name):
        try:
            df = pd.read_csv(file)
            ax.plot(df["obs/p_boat/0"], df["obs/p_boat/1"],
                    alpha=0.4, color=color, linewidth=1.5)
        except:
            nb_fails += 1
        # ax.scatter(df["obs/p_boat/0"], df["obs/p_boat/1"],
        #            alpha=0.2, color=color, s=.1)
    print(f'Failed to load {nb_fails} files for {name}')


def draw_trajectories(names, xte_delta=10):
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
    fig.legend(loc='upper center',
               bbox_to_anchor=(0.5, -.05),
               ncol=len(names) + 2)


def get_metric(name, metric='cum_vmc', key=lambda df: 0):
    scores_by_key = defaultdict(list)
    files = list(glob(f'../runs/{name}/**/*.csv'))
    nb_fails = 0
    for file in tqdm(files, desc=name):
        try:
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
                max_time = int(df["relative_time"].iloc[-1])
                counts, bins = np.histogram(df["relative_time"][:-1],
                                            bins=range(max_time))
                last_idx = np.where(counts > 0)[0][-1]
                score = counts[:last_idx]
                df_key = bins[:last_idx]
            else:
                score = df[metric].values
                df_key = np.arange(len(score))
            if isinstance(score, np.ndarray) and isinstance(df_key, np.ndarray):
                for k, s in zip(df_key, score):
                    scores_by_key[k].append(s)
            else:
                scores_by_key[df_key].append(score)
        except:
            nb_fails += 1

    print(f'Failed to load {nb_fails} files for {name}')
    keys = list(scores_by_key.keys())
    scores_mean = np.array([np.mean(scores_by_key[k]) for k in keys])
    scores_std = np.array([np.std(scores_by_key[k]) for k in keys])
    return keys, scores_mean, scores_std
