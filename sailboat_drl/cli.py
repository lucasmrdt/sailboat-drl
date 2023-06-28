import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import cache

from .rewards import available_rewards
from .wrappers import available_obs_wrappers, available_act_wrappers


def extended_eval(s):
    return eval(s, {'pi': np.pi})


@cache
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed',
                        type=int, default=0, help='random seed')
    parser.add_argument('--container-tag',
                        choices=['realtime', *[f'mss{i}' for i in range(11)]], default='mss1', help='environment container tag (see https://github.com/lucasmrdt/sailboat-gym/blob/main/DOCUMENTATION.md#container-tags-container_tag for more information)')
    parser.add_argument('--reward', choices=available_rewards.keys(),
                        default=list(available_rewards.keys())[0], help='reward function')
    parser.add_argument('--reward-args', type=extended_eval,
                        default={'target': [1, .5], 'radius': 10}, help='reward function arguments')
    parser.add_argument('--obs', choices=available_obs_wrappers.keys(),
                        default=list(available_obs_wrappers.keys())[0], help='observation used by the agent')
    parser.add_argument('--act', choices=available_act_wrappers.keys(),
                        default=list(available_act_wrappers.keys())[0], help='action used by the agent')

    # stable-baselines3 arguments
    parser.add_argument('--n-train-envs', type=int, default=1,
                        help='number of training environments')
    parser.add_argument('--n-eval-envs', type=int, default=1,
                        help='number of evaluation environments')
    parser.add_argument('--n-steps-per-rollout', type=int, default=2048,
                        help='number of steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--eval-every-n-rollout', type=int, default=1,
                        help='eval every n rollout')
    parser.add_argument('--total-steps', type=int, default=1e6,
                        help='total steps')
    args = parser.parse_args()

    return args


args = parse_args()
