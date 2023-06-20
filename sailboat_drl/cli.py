import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import cache

from .rewards import available_rewards


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
    parser.add_argument('--selected-obs', type=extended_eval,
                        default=['p_boat'], help='selected observations')
    args = parser.parse_args()

    print('Arguments:')
    for k, v in vars(args).items():
        print(f'{k} = {v}')
    return args


args = parse_args()
