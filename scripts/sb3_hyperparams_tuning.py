import allow_local_package_imports

import optuna
import time
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from rl_zoo3.hyperparams_opt import sample_ppo_params

from train_model import train_model
from eval_model import eval_model


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, required=True,
                        help='experiment name')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='number of trials')
    parser.add_argument('--index', type=int, required=True,
                        help='index of the job')
    args, unknown = parser.parse_known_args()
    return args


def prepare_objective(args, idx):
    def objective(trial: optuna.Trial):
        sampled_args = sample_ppo_params(trial)
        overwrite_args = {
            'name': f'{args.name}-{trial.number}',
            'keep_sim_running': True,
            'total_steps': 10_000,
            'prefix_env_id': f'{args.name}-{idx}-',
            **sampled_args,
        }

        train_model(overwrite_args)
        mean_reward, std_reward = eval_model(overwrite_args)
        print(f'{args.name}-{trial.number}: {mean_reward}')
        return mean_reward
    return objective


def optimize():
    args = parse_args()

    random.seed()
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
