import allow_local_package_imports

import optuna
import time
import random
import torch.nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from rl_zoo3.hyperparams_opt import sample_ppo_params

from sailboat_drl.env import available_water_current_generators, available_wind_generators
from sb3_train import train_model
from sb3_eval import eval_model


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, required=True,
                        help='experiment name')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='number of trials')
    parser.add_argument('--water-current', choices=list(available_water_current_generators.keys()),
                        default='none', help='water current generator')
    parser.add_argument('--wind', choices=list(available_wind_generators.keys()),
                        default='constant', help='wind generator')
    parser.add_argument('--index', type=int, required=True,
                        help='index of the job')
    args, unknown = parser.parse_known_args()
    return args


def prepare_objective(args, idx):
    def objective(trial: optuna.Trial):
        overwrite_args = {
            'name': f'{args.name}-{trial.number}',
            'keep_sim_running': True,
            'total_steps': 10_000,
            'prefix_env_id': idx,
            'policy_kwargs': {'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False},
            'batch_size': 16,
            'n_steps': 1024,
            'gamma': 0.999,
            'gae_lambda': 0.9,
            'max_grad_norm': 0.6,
            'learning_rate': 3e-5,
            'vf_coef': 0.2,
            'n_epochs': 10,
            'wind_dirs': [45, 90, 135, 180, 225, 270, 315],
            'reward': 'max_vmc_custom_shape_tuning',
            'obs': 'basic_2d_obs_v5',
            'total': 10_000,
            'n_envs': 7,
            'reward_kwargs': {
                'path': [[0, 0], [100, 0]],
                'xte_threshold': .1,
                'rudder_coef': trial.suggest_float('rudder_coef', 0.01, 1, log=True),
                'vmc_coef': 1,
                'xte_coef': 1,
                'xte_params': dict(
                    steepness=trial.suggest_float(
                        'xte_steepness', 1, 20, log=True),
                ),
                'vmc_params': dict(
                    steepness=trial.suggest_float(
                        'vmc_steepness', 1, 20, log=True),
                    start_penality=trial.suggest_float('start_penality', 0, 1),
                ),
            }
        }

        train_model(overwrite_args)
        mean_reward, std_reward = eval_model(overwrite_args)
        print(f'{args.name}-{trial.number}: {mean_reward}')
        return mean_reward
    return objective


def optimize():
    args = parse_args()

    time.sleep(random.random() * 5)
    objective = prepare_objective(args, args.index)
    study = optuna.create_study(
        direction='maximize',
        study_name=args.name,
        storage=f'sqlite:///optuna.db',
        load_if_exists=True)
    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1,
                       gc_after_trial=True, show_progress_bar=True)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    optimize()
