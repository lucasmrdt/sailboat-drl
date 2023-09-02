import allow_local_package_imports

import optuna
import numpy as np
import time
import os
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sailboat_drl.env import available_water_current_generators, available_wind_generators
from pid_eval import eval_pid


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, required=True,
                        help='experiment name')
    parser.add_argument('--wind-dirs', type=eval, required=True,
                        help='wind directions (in deg)')
    parser.add_argument('--index', type=int, required=True,
                        help='index of the job')
    parser.add_argument('--Kp', type=float, help='Kp')
    parser.add_argument('--Ki', type=float, help='Ki')
    parser.add_argument('--Kd', type=float, help='Kd')
    parser.add_argument('--los-radius', type=float, help='los_radius')
    parser.add_argument('--water-current', choices=list(available_water_current_generators.keys()),
                        default='none', help='water current generator')
    parser.add_argument('--wind', choices=list(available_wind_generators.keys()),
                        default='constant', help='wind generator')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='number of trials')
    args, unknown = parser.parse_known_args()
    return args


def prepare_objective(args, study_name, wind_dir):
    def objective(trial: optuna.Trial):
        Kp = trial.suggest_float('Kp', 2e-4, 2, log=True) \
            if args.Kp is None else args.Kp
        Ki = trial.suggest_float('Ki', 2e-4, 2, log=True) \
            if args.Ki is None else args.Ki
        Kd = trial.suggest_float('Kd', 2e-4, 2, log=True) \
            if args.Kd is None else args.Kd
        los_radius = trial.suggest_float('los_radius', 0.1, 400) \
            if args.los_radius is None else args.los_radius

        overwrite_args = {
            'name': f'{study_name}-{trial.number}',
            'keep_sim_running': True,
            'Kp': Kp,
            'Ki': Ki,
            'Kd': Kd,
            'los_radius': los_radius,
            'wind_dir': wind_dir,
            'prefix_env_id': f'{args.name}-{args.index}-',
        }

        mean_reward, std_reward = eval_pid(overwrite_args)
        return mean_reward
    return objective


def optimize_for_wind_dir(args, wind_dir):
    time.sleep(np.random.rand() * 5)

    study_name = f'{args.name}-{wind_dir}deg'
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f'sqlite:///optuna.db',
        load_if_exists=True)

    objective = prepare_objective(args, study_name, wind_dir)
    try:
        study.optimize(objective,
                       n_trials=args.n_trials,
                       n_jobs=1,
                       gc_after_trial=True,
                       show_progress_bar=True)
    except KeyboardInterrupt:
        pass


def optimize():
    args = parse_args()

    wind_dirs = args.wind_dirs
    assert isinstance(wind_dirs, list), 'wind_dirs must be a list'

    _optimize_for_wind_dir = partial(optimize_for_wind_dir, args)
    with Pool(len(wind_dirs)) as p:
        p.map(_optimize_for_wind_dir, wind_dirs)


if __name__ == '__main__':
    optimize()
    os.system('docker kill $(docker ps -q)')
