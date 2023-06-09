from IPython.display import HTML
from base64 import b64encode
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt
import glob
import re

current_dir = osp.dirname(osp.abspath(__file__))
project_dir = osp.join(current_dir, '..')

def load_run(run_id: str, eval=False):
    job_type = 'eval' if eval else 'train'
    pattern = f'runs/{run_id}/{job_type}-*/**/*.csv'

    csv_length = None
    csvs = []

    for file_path in glob.glob(osp.join(project_dir, pattern)):
        try:
            run_idx = re.search(f'{job_type}-(\d+)/', file_path).group(1)
            episode_idx = re.search('/(\d+)/', file_path).group(1)
        except AttributeError:
            continue

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            continue

        if csv_length is None:
            csv_length = len(df)
        elif len(df) != csv_length:
            continue

        df['run_idx'] = run_idx
        df['episode_idx'] = episode_idx
        df['step_idx'] = df.index

        csvs.append(df)

    df = pd.concat(csvs, ignore_index=True)
    df.set_index(['run_idx', 'episode_idx', 'step_idx'], inplace=True)
    return df

def load_runs(run_ids: list, eval=False):
    runs = []
    for run_id in tqdm(run_ids, desc='loading', total=len(run_ids), leave=False):
        try:
            runs.append(load_run(run_id, eval=eval))
        except Exception:
            print(f'[WARNING] cannot load {run_id}')
    return runs

def show_videos(run_id: str, selected_eps=None):
    def extract_episode_idx(file_name):
        return int(re.search("rl-video-episode-(\d+)", file_name).group(1))

    def extract_env_idx(file_name):
        return int(re.search("eval-(\d+)", file_name).group(1))

    pattern = f'runs/{run_id}/eval-*/videos/rl-video-episode-*.mp4'
    filepaths = glob.glob(osp.join('..', pattern))
    filepaths.sort(key=extract_env_idx)
    filepaths.sort(key=extract_episode_idx)

    episodes = defaultdict(list)
    for filepath in filepaths:
        episodes[extract_episode_idx(filepath)].append(filepath)

    # remove last episode as it is not complete
    episodes.pop(max(episodes.keys()))

    html_str=""
    for episode_idx, filepaths in episodes.items():
        if selected_eps is not None and episode_idx not in selected_eps:
            continue
        html_str += f'<h2>{(episode_idx+1)/len(episodes):.0%}</h2>'
        for filepath in filepaths:
            mp4 = open(filepath,'rb').read()
            data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
            html_str += f"""
            <video width={1/len(filepaths):.0%} controls autoplay loop muted>
                <source src="{data_url}" type="video/mp4">
            </video>
            """
        html_str += '<br>'
    return HTML(html_str)

def plot_cum_vmc(df, ax=None, label=lambda x: f'{x:.0%}', selected_eps=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    df_cum_vmc = df['obs/cum_obs/vmc']
    cum_vmc_by_episode = df_cum_vmc.groupby('episode_idx')

    for idx, (_, ep_cum_vmc) in enumerate(cum_vmc_by_episode):
        if selected_eps is not None and idx not in selected_eps:
            continue
        ep_cum_vmc = ep_cum_vmc.groupby('step_idx')
        mean_cum_vmc = ep_cum_vmc.mean()
        std_cum_vmc = ep_cum_vmc.std()
        ax.plot(mean_cum_vmc, label=label((idx+1)/cum_vmc_by_episode.ngroups))
        ax.fill_between(mean_cum_vmc.index, mean_cum_vmc - std_cum_vmc, mean_cum_vmc + std_cum_vmc, alpha=0.2)

    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    ax.set_xlabel('step')
    ax.set_ylabel('cumulative vmc')


def compare_video_of_runs(run_ids: list, selected_eps=None):
    html_str=""
    for run_id in tqdm(run_ids, desc='plotting', total=len(run_ids), leave=False):
        try:
            html_video = show_videos(run_id, selected_eps)._repr_html_()
        except Exception:
            print(f'[WARNING] cannot load {run_id}')
            continue
        html_str += f'<h1>{run_id}</h1>'
        html_str += html_video
    return HTML(html_str)

def compare_cum_vmc_of_runs(run_ids: list, eval=False, selected_eps=None):
    _, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    runs = load_runs(run_ids, eval=eval)
    for run_id, df in tqdm(zip(run_ids, runs), desc='plotting', total=len(run_ids), leave=False):
        plot_cum_vmc(df, ax=ax, label=lambda x: f'{run_id} {x:.0%}', selected_eps=selected_eps)