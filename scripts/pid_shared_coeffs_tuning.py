import optuna
import os
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from eval_pid import eval_pid


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
    parser.add_argument('--n-trials', type=int, default=100,
                        help='number of trials')
    args, unknown = parser.parse_known_args()
    return args


def eval_pid_for_wind_dir(args, wind_dir):
    overwrite_args = {
        **args,
        'wind_dir': wind_dir,
    }
    mean_reward, std_reward = eval_pid(overwrite_args)
    return mean_reward


def prepare_objective(args, study_name):
    def objective(trial: optuna.Trial):
        Kp = trial.suggest_float('Kp', 2e-4, 2, log=True) \
            if args.Kp is None else args.Kp
        Ki = trial.suggest_float('Ki', 2e-4, 2, log=True) \
            if args.Ki is None else args.Ki
        Kd = trial.suggest_float('Kd', 2e-4, 2, log=True) \
            if args.Kd is None else args.Kd
        los_radius = trial.suggest_float('los_radius', 0.1, 400) \
            if args.los_radius is None else args.los_radius

        wind_dirs = args.wind_dirs
        assert isinstance(wind_dirs, list), 'wind_dirs must be a list'

        overwrite_args = {
            'name': f'{study_name}-{trial.number}',
            'keep_sim_running': True,
            'Kp': Kp,
            'Ki': Ki,
            'Kd': Kd,
            'los_radius': los_radius,
            'prefix_env_id': f'{args.name}-{args.index}-',
        }

        _eval_pid_for_wind_dir = partial(eval_pid_for_wind_dir, overwrite_args)
        with Pool(len(wind_dirs)) as p:
            rewards = p.map(_eval_pid_for_wind_dir, wind_dirs)
        return sum(rewards)
    return objective


def optimize():
    args = parse_args()

    study_name = args.name
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f'sqlite:///optuna.db',
        load_if_exists=True)

    objective = prepare_objective(args, study_name)
    try:
        study.optimize(objective,
                       n_trials=args.n_trials,
                       n_jobs=1,
                       gc_after_trial=True,
                       show_progress_bar=True)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    optimize()
    os.system('docker kill $(docker ps -q)')
