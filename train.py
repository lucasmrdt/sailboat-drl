import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback
from sailboat_gym import SailboatLSAEnv

from sailboat_drl import prepare_env, args, TimeLoggerCallback, extract_env_instance


def train():
    print('Training with the following arguments:')
    for k, v in vars(args).items():
        print(f'{k} = {v}')

    group_id = wandb.util.generate_id()
    wandb_args = {
        'project': 'sailboat-drl',
        'config': vars(args),
        'group': group_id,
    }
    wandb.init(**wandb_args, name='train.py', sync_tensorboard=True)

    train_env = SubprocVecEnv(
        [prepare_env(f'train-{i}', wandb_args=wandb_args) for i in range(args.n_train_envs)])
    eval_env = SubprocVecEnv(
        [prepare_env(f'eval-{i}', wandb_args=wandb_args, eval=True, record=(i == 0)) for i in range(args.n_eval_envs)])

    # base_env = extract_env_instance(train_env[0], SailboatEnv)

    n_steps_per_rollout = args.n_episode_per_rollout \
        * args.episode_duration \
        * SailboatLSAEnv.NB_STEPS_PER_SECONDS
    model = PPO('MlpPolicy',
                train_env,
                n_steps=n_steps_per_rollout // args.n_train_envs,
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
                tensorboard_log=f'runs/{group_id}')

    eval_cb = EvalCallback(eval_env,
                           best_model_save_path=f'models/{group_id}',
                           log_path=f'runs/{group_id}',
                           eval_freq=n_steps_per_rollout *
                           args.eval_every_n_rollout // args.n_train_envs,
                           verbose=1,
                           n_eval_episodes=1)
    wandb_cb = WandbCallback(model_save_path=f'models/{group_id}')
    time_cb = TimeLoggerCallback()

    model.learn(args.total_steps,
                callback=[eval_cb, wandb_cb, time_cb],
                progress_bar=True)

    model.save(f'models/{group_id}/final')


if __name__ == '__main__':
    train()
