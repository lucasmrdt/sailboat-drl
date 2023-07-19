from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from .env import prepare_env
from .cli import args
from .callbacks import TimeLoggerCallback
from .logger import Logger


def train(trial=None) -> float:
    print('Training with the following arguments:')
    for k, v in vars(args).items():
        print(f'{k} = {v}')

    # wandb_args = {
    #     'project': 'sailboat-drl',
    #     'config': vars(args),
    #     'group': args.name,
    #     'reinit': True,
    #     'settings': wandb.Settings(start_method="fork"),
    #     # 'mode': 'offline',
    # }
    # wandb.init(**wandb_args, name='train.py', sync_tensorboard=True)
    Logger.configure('train.py')

    train_env = SubprocVecEnv(
        [prepare_env(i) for i in range(args.n_train_envs)])
    eval_env = SubprocVecEnv(
        [prepare_env(i, is_eval=True) for i in range(args.n_eval_envs)])

    model = PPO('MlpPolicy',
                train_env,
                n_steps=max(1, args.n_steps // args.n_train_envs),
                batch_size=args.batch_size,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                ent_coef=args.ent_coef,
                clip_range=args.clip_range,
                n_epochs=args.n_epochs,
                gae_lambda=args.gae_lambda,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                policy_kwargs=args.policy_kwargs,
                seed=args.seed)
    model.set_logger(Logger.get_sb3_logger())

    time_cb = TimeLoggerCallback()

    if trial:
        from rl_zoo3.callbacks import TrialEvalCallback
        eval_cb = TrialEvalCallback(eval_env,
                                    trial,
                                    log_path=f'runs/{args.name}',
                                    eval_freq=args.total_steps * args.eval_freq // args.n_train_envs,
                                    n_eval_episodes=args.n_eval_envs)
    else:
        eval_cb = EvalCallback(eval_env,
                               best_model_save_path=f'runs/{args.name}',
                               log_path=f'runs/{args.name}',
                               eval_freq=args.total_steps * args.eval_freq // args.n_train_envs,
                               n_eval_episodes=args.n_eval_envs)

    try:
        model.learn(args.total_steps,
                    callback=[time_cb, eval_cb],
                    progress_bar=True)
    except (AssertionError, ValueError) as e:
        if trial:
            import optuna
            raise optuna.TrialPruned() from e
        else:
            raise e

    train_env.close()
    eval_env.close()

    if trial:
        import optuna
        if eval_cb.is_pruned:  # type: ignore
            raise optuna.TrialPruned()
    else:
        model.save(f'runs/{args.name}/final')

    hparams = {k: v if isinstance(v, (int, float, str, bool)) else str(v)
               for k, v in vars(args).items()}
    Logger.log_hyperparams(
        hparams, {'last_mean_reward': eval_cb.last_mean_reward})  # type: ignore
    return eval_cb.last_mean_reward  # type: ignore


if __name__ == '__main__':
    train()


# pip install --upgrade wandb[service]
# wandb.require("service")
