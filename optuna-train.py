import optuna
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from rl_zoo3.hyperparams_opt import sample_ppo_params

from sailboat_drl import train_model, eval_model


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, required=True,
                        help='experiment name')
    args, unknown = parser.parse_known_args()
    return args


def prepare_objective(study_name):
    def objective(trial: optuna.Trial):
        sampled_args = sample_ppo_params(trial)
        overwrite_args = {
            'name': f'{study_name}-{trial.number}',
            'keep_sim_running': True,
            **sampled_args,
        }

        train_model(overwrite_args)
        mean_reward, std_reward = eval_model(overwrite_args)
        return mean_reward
    return objective


def optimize():
    args = parse_args()

    study = optuna.create_study(
        # sampler=optuna.samplers.RandomSampler(),
        direction='maximize',
        study_name=args.name,
        storage=f'sqlite:///optuna.db',
        load_if_exists=True)

    objective = prepare_objective(args.name)

    try:
        study.optimize(objective, n_trials=1000, n_jobs=1,
                       gc_after_trial=True, show_progress_bar=True)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))


if __name__ == '__main__':
    optimize()
