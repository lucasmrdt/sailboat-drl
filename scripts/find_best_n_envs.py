import sys
sys.path.append('..')  # noqa
sys.path.append('.')  # noqa

import time
import pickle
import os
import os.path as osp
from tqdm import trange, tqdm
from itertools import count
from stable_baselines3.common.vec_env import SubprocVecEnv

from sailboat_drl import prepare_env

current_dir = osp.dirname(osp.abspath(__file__))


def find_best_n_envs():
    time_per_step_by_n_envs = {}

    for n_envs in trange(1, 32+1, desc='running experiments'):
        env = SubprocVecEnv([prepare_env(f'train-{i}') for i in range(n_envs)])

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
        print(f'n_envs = {n_envs}, time/step = {time_per_step*1000:.2f} ms')

        # res = input('Continue? [y]/n ')
        # if res == 'n':
        #     break

    return time_per_step_by_n_envs


if __name__ == '__main__':
    save_dir = osp.join(current_dir, '../output/pkl')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, 'best_n_envs.pkl')
    pickle.dump(find_best_n_envs(), open(save_path, 'wb'))
    print(f'Best n_envs saved to {save_path}')
