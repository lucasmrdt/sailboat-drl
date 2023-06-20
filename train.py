import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from sailboat_drl import prepare_env, cli


def train():
    run = wandb.init(
        project='sailboat-drl',
        sync_tensorboard=True,
        monitor_gym=True,
        config=vars(cli.args),
    )

    train_env = SubprocVecEnv(
        [prepare_env(f'train-{i}') for i in range(1)])
    eval_env = SubprocVecEnv(
        [prepare_env(f'eval-{i}', record=True, run_id=run.id) for i in range(1)])

    model = PPO('MlpPolicy', train_env, verbose=1,
                tensorboard_log=f'runs/{run.id}')

    eval_cb = EvalCallback(eval_env, best_model_save_path=f'models/{run.id}',
                           log_path=f'runs/{run.id}', eval_freq=600, deterministic=True, render=True, verbose=1, n_eval_episodes=1)
    wandb_cb = WandbCallback(gradient_save_freq=100,
                             model_save_path=f'models/{run.id}',
                             verbose=2)

    model.learn(total_timesteps=600 * 10,
                callback=[eval_cb, wandb_cb], progress_bar=True, log_interval=1)

    model.save('./output/models/ppo_sailboat')
    run.finish()


if __name__ == '__main__':
    train()
