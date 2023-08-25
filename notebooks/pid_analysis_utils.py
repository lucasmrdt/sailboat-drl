import re
import optuna
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from collections import defaultdict
from tqdm import tqdm
from prettytable import PrettyTable


def get_coeffs(wind_dirs, k=1, statistic='median', ascending=False):
    best_coeffs_by_wind_dir = {}
    for wind_dir in tqdm(wind_dirs, desc='Processing tuning results'):
        study = optuna.load_study(
            study_name=f'pid-coeffs-tuning-{wind_dir}deg',
            storage=f'sqlite:///../optuna.db')
        df = study.trials_dataframe()
        df = df.sort_values('value', ascending=ascending).head(k)
        params = df[['params_Kp', 'params_Kd', 'params_Ki']].values
        if statistic == 'median':
            Kp, Kd, Ki = np.median(params, axis=0)
        elif statistic == 'mean':
            Kp, Kd, Ki = np.mean(params, axis=0)
        else:
            raise ValueError(f'Unknown statistic: {statistic}')
        best_coeffs_by_wind_dir[wind_dir] = {
            'Kp': Kp,
            'Kd': Kd,
            'Ki': Ki,
        }
    return best_coeffs_by_wind_dir


def get_scores(name, metric='cum_vmc'):
    regex = re.compile(f'../runs/{name}-(.*?)deg')

    def extract_wind_dir(f):
        return float(re.search(regex, f).group(1))

    scores_by_wind_dir = defaultdict(list)
    folders = glob(f'../runs/{name}-*deg')
    for folder in tqdm(folders, total=len(folders), desc='Processing folders'):
        for file in glob(f'{folder}/eval-*deg/**/*.csv'):
            df = pd.read_csv(file)
            if metric == 'cum_vmc':
                score = df["obs/cum_obs/vmc/0"].iloc[-1]
            elif metric == 'mean_xte':
                score = df["obs/xte/0"].abs().mean()
            elif metric == 'path_dist':
                p_start, p_end = df[["obs/p_boat/0",
                                     "obs/p_boat/1"]].iloc[[0, -1]].values
                score = (p_end - p_start)[0]
            else:
                raise ValueError(f'Unknown metric: {metric}')
            wind_dir = extract_wind_dir(file)
            scores_by_wind_dir[wind_dir].append(score)

    wind_dirs = np.array(sorted(scores_by_wind_dir.keys()))
    scores_mean = np.array([np.mean(scores_by_wind_dir[wind_dir])
                           for wind_dir in wind_dirs])
    scores_std = np.array([np.std(scores_by_wind_dir[wind_dir])
                          for wind_dir in wind_dirs])

    return wind_dirs, scores_mean, scores_std


def compare_cum_vmc_vs_path_dist(name):
    cum_vmcs, path_dists, time_factors = [], [], []
    csv_files = glob(f'../runs/{name}-*deg/eval-*deg/**/*.csv')
    for file in tqdm(csv_files, total=len(csv_files), desc='Processing CSV files'):
        df = pd.read_csv(file)
        cum_vmc = df["obs/cum_obs/vmc/0"].iloc[-1]
        mean_vmc = df["obs/vmc/0"].mean()
        p_start, p_end = df[["obs/p_boat/0",
                             "obs/p_boat/1"]].iloc[[0, -1]].values
        path_dist = (p_end - p_start)[0]
        time_factor = df["time/factor"].iloc[-1]
        cum_vmcs.append(cum_vmc)
        path_dists.append(path_dist)
        time_factors.append(time_factor)

    cum_vmcs = np.array(cum_vmcs)
    path_dists = np.array(path_dists)
    time_factors = np.array(time_factors)

    return cum_vmcs, path_dists, time_factors


def compare_performance(names, metrics, names_labels=None, metrics_labels=None):
    fig, axs = plt.subplots(1, len(metrics),
                            figsize=(4 * len(metrics), 4), dpi=150)
    fig.subplots_adjust(wspace=.3)

    if len(metrics) == 1:
        axs = [axs]

    result_name_metric = defaultdict(dict)
    for i, (metric, label) in enumerate(zip(metrics, metrics_labels)):
        ax = axs[i]
        for (name, name_label) in zip(names, names_labels):
            wind_dirs, mean_by_wind, std_by_wind = get_scores(name,
                                                              metric=metric)
            ax.plot(wind_dirs,
                    mean_by_wind,
                    label=name_label if i == 0 else None)
            ax.fill_between(wind_dirs,
                            mean_by_wind - std_by_wind,
                            mean_by_wind + std_by_wind,
                            alpha=0.2)
            mean = np.mean(mean_by_wind)
            std = np.mean(std_by_wind)
            result_name_metric[name][metric] = (mean, std)
        ax.set_xlabel('Wind direction (deg)')
        ax.set_ylabel(label)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -.05), ncol=3)

    table = PrettyTable()
    table.field_names = ['', *metrics_labels]
    for name, name_label in zip(names, names_labels):
        table.add_row(
            [name_label, *[f'{mean:.2f} ± {std:.2f}' for mean, std in result_name_metric[name].values()]])
    print(table)


def draw_trajectories_of_name(name, ax=None, color='C0'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.plot([0, 200], [0, 0], 'k--', label='Reference path')

    for file in glob(f'../runs/{name}/eval-*deg/**/*.csv'):
        df = pd.read_csv(file)
        ax.plot(df["obs/p_boat/0"], df["obs/p_boat/1"], alpha=0.2, color=color)


def draw_trajectories(names, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.plot([0, 200], [0, 0], 'k--', label='Reference path')

    for name in names:
        print(f'Drawing trajectories of {name}')
        draw_trajectories_of_name(name, ax=ax)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend()
