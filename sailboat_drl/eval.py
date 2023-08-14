from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from torch import nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pickle

from .logger import Logger
from .train import prepare_env


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name',
                        type=str, required=True, help='experiment name')
    parser.add_argument('--n-envs', type=int, default=20,
                        help='number of environments')
    args = parser.parse_args()

    return args


def eval():
    args = get_args()
    path = f'runs/{args.name}'

    Logger.configure(f'{args.name}/eval.py')

    model = PPO.load(f'{path}/final.model.zip')
    train_args = pickle.load(open(f'{path}/final.args.pkl', 'rb'))

    env = SubprocVecEnv(
        [prepare_env(train_args, i, is_eval=True) for i in range(args.n_envs)])
    env = VecNormalize.load(f'{path}/final.envstats.pkl', env)
    env.training = False
    env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=args.n_envs)

    print(f'mean_reward = {mean_reward}')
    print(f'std_reward = {std_reward}')
    return mean_reward, std_reward


if __name__ == '__main__':
    eval()
