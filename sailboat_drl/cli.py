import uuid
import numpy as np
from torch import nn as nn
from functools import cache
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from sailboat_gym import env_by_name

def extended_eval(s):
    return eval(s, {'pi': np.pi})


@cache
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name',
                        type=str, default=str(uuid.uuid4()), help='experiment name')
    parser.add_argument('--seed',
                        type=int, default=0, help='random seed')
    # parser.add_argument('--container-tag',
    #                     choices=['realtime', *[f'mss{i}' for i in range(11)]], default='mss1', help='environment container tag (see https://github.com/lucasmrdt/sailboat-gym/blob/main/DOCUMENTATION.md#container-tags-container_tag for more information)')
    parser.add_argument('--env-name', choices=list(env_by_name.keys()),
                        default=list(env_by_name.keys())[0], help='environment name')
    parser.add_argument('--obs', help='observation used by the agent')
    parser.add_argument('--act', help='action used by the agent')
    parser.add_argument('--reward', help='reward function')
    parser.add_argument('--reward-kwargs', type=extended_eval,
                        default={}, help='reward function arguments')
    parser.add_argument('--train-episode-duration', type=int,
                        default=100, help='episode duration (in seconds)')
    parser.add_argument('--eval-episode-duration', type=int,
                        default=200, help='episode duration (in seconds)')
    parser.add_argument('--eval-freq', type=float,
                        default=.1, help='evaluation frequency (in percentage of total steps, should be in [0, 1])')
    parser.add_argument('--n-train-envs', type=int, default=20,
                        help='number of training environments')
    parser.add_argument('--n-eval-envs', type=int, default=4,
                        help='number of evaluation environments')
    parser.add_argument('--use-same-sim', action='store_true',
                        help='use the same simulation for evaluation and training')
    # parser.add('--log-freq', type=int, default=100, help='log frequency (in number of steps)')

    # stable-baselines3 arguments
    parser.add_argument('--n-steps', type=int, default=1000,
                        help='number of steps')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--ent-coef', type=float, default=0.0,
                        help='entropy coefficient')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='clip range')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max grad norm')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='value function coefficient')
    parser.add_argument('--policy-kwargs', type=extended_eval,
                        default={'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': False}, help='policy kwargs')
    parser.add_argument('--total-steps', type=int, default=1e6,
                        help='total steps')
    args = parser.parse_args()

    return args


args = parse_args()

runtime_env = Namespace(
    nb_steps_per_second=env_by_name[args.env_name].NB_STEPS_PER_SECONDS,
)

