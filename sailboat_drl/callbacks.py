import time
from stable_baselines3.common.callbacks import BaseCallback
from sailboat_gym import SailboatLSAEnv

from .cli import args
from .utils import extract_env_instance


class TimeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None

    def _on_rollout_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.start_time is not None
        base_env = extract_env_instance(self.training_env, SailboatLSAEnv)
        rollout_time = time.time() - self.start_time
        n_steps_per_episode = args.episode_duration * SailboatLSAEnv.NB_STEPS_PER_SECONDS
        n_steps_per_rollout = args.n_episode_per_rollout * n_steps_per_episode
        average_step_time = rollout_time / n_steps_per_rollout
        self.logger.record("time/step",
                           average_step_time)
        self.logger.record("time/episode",
                           average_step_time * n_steps_per_episode)
        self.logger.record("time/factor",
                           (1/SailboatLSAEnv.NB_STEPS_PER_SECONDS) / average_step_time)
