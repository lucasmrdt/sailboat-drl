from sailboat_drl import prepare_env, args
from sailboat_gym import SailboatLSAEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from itertools import count
import time
import tqdm
import numpy as np
import wandb

if __name__ == '__main__':
    group_id = wandb.util.generate_id()
    wandb_args = {
        'project': 'sailboat-drl',
        'config': vars(args),
        'group': group_id,
        'mode': 'online',
    }
    wandb.init(**wandb_args, name='train.py', sync_tensorboard=True)


    envs = SubprocVecEnv([
        prepare_env(f'eval-{i}', eval=True, record=True, wandb_args=wandb_args) for i in range(args.n_eval_envs)
    ])

    obs = envs.reset()
    for _ in tqdm.tqdm(count()):
        act = envs.action_space.sample()
        obs, reward, done, info = envs.step([act for _ in range(args.n_eval_envs)])
        if any(done):
            break

    envs.close()

# 1199it [00:52, 22.65it/s] (eval=False, record=False)