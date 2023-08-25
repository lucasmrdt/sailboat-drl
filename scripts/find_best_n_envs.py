import allow_local_package_imports

import time
import pickle
import os
import os.path as osp
from tqdm import trange, tqdm
from itertools import count
from stable_baselines3.common.vec_env import SubprocVecEnv
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sailboat_drl.env import create_env

current_dir = osp.dirname(osp.abspath(__file__))
save_dir = osp.join(current_dir, '../output/pkl')


def get_args(overwrite_args={}):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--container-tag', type=str, default='mss1-ode',
                        help='container tag')
    args, unknown = parser.parse_known_args()

    args.__dict__ = {k: v for k, v in vars(args).items()
                     if k not in overwrite_args}
    args.__dict__.update(overwrite_args)
    return args


def prepare_env(args, idx, n_envs):
    name = f'find-best-n-envs-{idx}-{n_envs}'

    def _init():
        return create_env(env_id=name,
                          is_eval=True,
                          keep_sim_running=True,
                          episode_duration=50,
                          prepare_env_for_nn=False,
                          container_tag=args.container_tag,
                          logger_prefix=name)
    return _init


def find_best_n_envs():
    args = get_args()

    time_per_step_by_n_envs = {}

    for n_envs in tqdm(range(1, 40 + 1, 2), desc='running experiments'):
        env = SubprocVecEnv([prepare_env(args, i, n_envs)
                            for i in range(n_envs)])

        start = time.time()
        env.reset()
        for i in tqdm(count(), desc='running episodes', leave=False):
            act = [env.action_space.sample() for _ in range(n_envs)]
            obs, _, done, _ = env.step(act)
            if any(done):
                break
        end = time.time()

        time_per_step = (end - start) / (n_envs * i)
        time_per_step_by_n_envs[n_envs] = time_per_step
        print(
            f'n_envs = {n_envs}, time/step = {time_per_step*1000:.2f} ms, time/step/env = {time_per_step*1000/n_envs:.2f} ms')

    return time_per_step_by_n_envs


if __name__ == '__main__':
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, 'best_n_envs.pkl')

    pickle.dump(find_best_n_envs(), open(save_path, 'wb'))
    print(f'Best n_envs saved to {save_path}')

    # kill all running docker containers
    os.system('docker kill $(docker ps -q)')
