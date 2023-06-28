import time
from stable_baselines3.common.callbacks import BaseCallback
from sailboat_gym import EPISODE_LENGTH, SailboatLSAEnv

from .cli import args


class TimeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None

    def _on_rollout_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        rollout_time = time.time() - self.start_time
        average_step_time = rollout_time / args.n_steps_per_rollout
        self.logger.record("time/step",
                           average_step_time)
        self.logger.record("time/episode",
                           average_step_time * EPISODE_LENGTH)
        self.logger.record("time/factor",
                           (1/SailboatLSAEnv.SIM_RATE) / average_step_time)
