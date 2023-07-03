import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from sailboat_drl import prepare_env, args, TimeLoggerCallback


def train():
    print('Training with the following arguments:')
    for k, v in vars(args).items():
        print(f'{k} = {v}')

    group = wandb.util.generate_id()
    wandb_args = {
        'project': 'sailboat-drl',
        'sync_tensorboard': True,
        'config': vars(args),
        'group': group,
    }

    run = wandb.init(**wandb_args)

    train_env = SubprocVecEnv(
        [prepare_env(f'train-{i}', wandb_args=wandb_args, eval=False) for i in range(args.n_train_envs)])
    eval_env = SubprocVecEnv(
        [prepare_env(f'eval-{i}', wandb_args=wandb_args, eval=True) for i in range(args.n_eval_envs)])

    model = PPO('MlpPolicy',
                train_env,
                n_steps=args.n_steps_per_rollout // args.n_train_envs,
                batch_size=args.batch_size,
                # gae_lambda=0.95,
                # gamma=0.9,
                # n_epochs=10,
                # ent_coef=0.0,
                # learning_rate=1e-3,
                # clip_range=0.2,
                # use_sde=True,
                # sde_sample_freq=4,
                verbose=1,
                tensorboard_log=f'runs/{run.id}')

    eval_cb = EvalCallback(eval_env,
                           best_model_save_path=f'models/{run.id}',
                           log_path=f'runs/{run.id}',
                           eval_freq=args.n_steps_per_rollout * args.eval_every_n_rollout // args.n_train_envs,
                           verbose=1,
                           n_eval_episodes=1)
    wandb_cb = WandbCallback(model_save_path=f'models/{run.id}')
    time_cb = TimeLoggerCallback()

    model.learn(args.total_steps,
                callback=[eval_cb, wandb_cb, time_cb],
                progress_bar=True)

    model.save(f'models/{run.id}/final')
    run.finish()


if __name__ == '__main__':
    train()
