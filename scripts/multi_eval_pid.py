import allow_local_package_imports

import time
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial
from multiprocessing import Pool

from eval_pid import eval_pid


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, required=True,
                        help='experiment name')
    parser.add_argument('--wind-dirs', type=eval, required=True,
                        help='wind directions (in deg)')
    args, unknown = parser.parse_known_args()
    return args


def eval_pid_for_wind_dir(args, wind_dir):
    try:
        time.sleep(random.random() * 5)
        overwrite_args = {
            'name': f'{args.name}-{wind_dir}deg',
            'wind_dir': wind_dir,
            'keep_sim_running': False,
        }
        mean_reward, std_reward = eval_pid(overwrite_args)
        print(f'[{wind_dir}deg] mean_reward={mean_reward}, std_reward={std_reward}')
    except Exception as e:
        print(f'[{wind_dir}deg] {e}')


def multi_eval_pid():
    args = parse_args()

    wind_dirs = args.wind_dirs
    assert isinstance(wind_dirs, list), 'wind_dirs must be a list'

    _eval_pid_for_wind_dir = partial(eval_pid_for_wind_dir, args)
    with Pool(len(wind_dirs)) as p:
        p.map(_eval_pid_for_wind_dir, wind_dirs)


if __name__ == '__main__':
    multi_eval_pid()
