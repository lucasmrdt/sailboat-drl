import allow_local_package_imports

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from torch import nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pickle

from sailboat_drl.env import available_water_current_generators, available_wind_generators
from sailboat_drl.logger import Logger
from sb3_train import prepare_env, parse_args as parse_train_args
from utils import evaluate_policy


def parse_args(overwrite_args={}):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name',
                        type=str, required=True, help='name of the train experiment')
    parser.add_argument('--seed',
                        type=int, default=0, help='random seed')
    parser.add_argument('--checkpoint-step', type=int,
                        help='checkpoint step')
    parser.add_argument('--log-name', type=str, help='log name')
    parser.add_argument('--keep-sim-running', action='store_true',
                        help='keep the simulator running after training')
    parser.add_argument('--water-current', choices=list(available_water_current_generators.keys()),
                        default='none', help='water current generator')
    parser.add_argument('--wind', choices=list(available_wind_generators.keys()),
                        default='constant', help='wind generator')
    parser.add_argument('--n-envs', type=int, default=7,
                        help='number of environments')
    parser.add_argument('--episode-duration', type=int, default=200,
                        help='episode duration (in seconds)')
    args, unknown = parser.parse_known_args()

    args.__dict__ = {k: v for k, v in vars(args).items()
                     if k not in overwrite_args}
    args.__dict__.update(overwrite_args)
    return args


def eval_model(overwrite_args={}):
    args = parse_args(overwrite_args)

    log_name = args.log_name or args.name

    print('Evaluating with the following arguments:')
    for k, v in vars(args).items():
        print(f'{k} = {v}')

    Logger.configure(f'{log_name}/eval.py')
    Logger.log_hyperparams(args.__dict__)

    if args.checkpoint_step is None:
        model_name = 'final'
    else:
        model_name = f'{args.checkpoint_step}_steps'

    path = f'runs/{args.name}'
    model = PPO.load(f'{path}/model_{model_name}')
    train_args = pickle.load(open(f'{path}/model_args.pkl', 'rb'))
    train_args.__dict__.update(args.__dict__)

    if args.log_name is not None:
        train_args.__dict__['name'] = args.log_name

    assert args.n_envs % len(train_args.wind_dirs) == 0, \
        'n_envs must be a multiple of len(wind_dirs)'

    env = SubprocVecEnv(
        [prepare_env(train_args, i, is_eval=True) for i in range(args.n_envs)])
    env = VecNormalize.load(f'{path}/model_vecnormalize_{model_name}.pkl', env)
    env.training = False
    env.norm_reward = False

    mean_reward, std_reward = evaluate_policy(model, env,
                                              n_eval_episodes=args.n_envs)
    if isinstance(mean_reward, list):
        mean_reward = mean_reward[0]
    if isinstance(std_reward, list):
        std_reward = std_reward[0]

    env.close()

    print(f'mean_reward = {mean_reward}')
    print(f'std_reward = {std_reward}')
    return mean_reward, std_reward


if __name__ == '__main__':
    eval_model()
