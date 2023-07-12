import optuna
from rl_zoo3.hyperparams_opt import sample_ppo_params

from sailboat_drl import train, args

init_args = args.__dict__.copy()

def objective(trial: optuna.Trial) -> float:
    args.__dict__ = init_args.copy()
    sampled_args = sample_ppo_params(trial)
    args.__dict__.update(sampled_args)
    args.__dict__['name'] += f'-{trial.number}'
    return train(trial=trial)

def optimize():
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(),
        direction='maximize',
        study_name=args.name,
        storage=f'sqlite:///optuna.db',
        load_if_exists=True)

    try:
        study.optimize(objective, n_trials=1000, n_jobs=1, gc_after_trial=True, show_progress_bar=True)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

if __name__ == '__main__':
    optimize()