import optuna
import numpy as np
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sailboat_drl import evaluate_pid_algo


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, required=True,
                        help='experiment name')
    args, unknown = parser.parse_known_args()
    return args


def prepare_objective(study_name, theta_wind):
    def objective(trial: optuna.Trial):
        Kp = trial.suggest_float('Kp', 0, 10)
        Ki = trial.suggest_float('Ki', 0, 10)
        Kd = trial.suggest_float('Kd', 0, 10)
        los_radius = trial.suggest_float('los_radius', 0.1, 200)

        overwrite_args = {
            'name': f'{study_name}-{trial.number}',
            'keep_sim_running': True,
            'Kp': Kp,
            'Ki': Ki,
            'Kd': Kd,
            'los_radius': los_radius,
        }

        mean_reward, std_reward = evaluate_pid_algo(overwrite_args, theta_wind)
        return mean_reward
    return objective


def optimize_for_theta_wind(name, theta_wind):
    study_name = f'{name}-{round(np.rad2deg(theta_wind))}deg'
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=f'sqlite:///optuna.db',
        load_if_exists=True)

    objective = prepare_objective(study_name, theta_wind)
    try:
        study.optimize(objective, n_trials=1000, n_jobs=1,
                       gc_after_trial=True, show_progress_bar=True)
    except KeyboardInterrupt:
        pass


def optimize():
    args = parse_args()

    no_go_zone = np.deg2rad(30)
    thetas = np.linspace(-np.pi + no_go_zone,
                         np.pi - no_go_zone,
                         30,
                         endpoint=True)

    _optimize_for_theta_wind = partial(optimize_for_theta_wind, args.name)
    with Pool(len(thetas)) as p:
        p.map(_optimize_for_theta_wind, thetas.tolist())


if __name__ == '__main__':
    optimize()
