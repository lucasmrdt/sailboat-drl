import time
from stable_baselines3.common.callbacks import BaseCallback

from .logger import Logger


class TimeLoggerCallback(BaseCallback):
    def __init__(self, n_steps_by_rollout, n_steps_per_second, verbose=0):
        super().__init__(verbose)
        self.n_steps_by_rollout = n_steps_by_rollout
        self.n_steps_per_second = n_steps_per_second
        self.start_time = None

    def _on_rollout_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.start_time is not None
        rollout_time = time.time() - self.start_time
        average_step_time = rollout_time / self.n_steps_by_rollout
        Logger.record({
            "time/rollout": rollout_time,
            "time/step": average_step_time,
            "time/factor": (1 / self.n_steps_per_second) / average_step_time,
        })
        # Logger.dump(step=self.num_timesteps)
