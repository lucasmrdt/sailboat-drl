import wandb
from sailboat_drl import prepare_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

def prepare_env():
    

def eval():
    env = DummyVecEnv([prepare_env()])
    env = VecVideoRecorder(
        env, f'videos/{run.id}', record_video_trigger=lambda x: x == 0, video_length=600, name_prefix='sailboat-')

    model.learn(total_timesteps=600 * 10,
                callback=[eval_cb, wandb_cb], progress_bar=True, log_interval=1)

    model.save('./output/models/ppo_sailboat')
    run.finish()


if __name__ == '__main__':
    train()
