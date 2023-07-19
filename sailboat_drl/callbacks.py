import time
from stable_baselines3.common.callbacks import BaseCallback

from .cli import runtime_env, args
from .logger import Logger


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
        rollout_time = time.time() - self.start_time
        average_step_time = rollout_time / args.n_steps
        Logger.record({
            "time/rollout": rollout_time,
            "time/step": average_step_time,
            "time/factor": (1/runtime_env.nb_steps_per_second) / average_step_time,
        })
        # Logger.dump(step=self.num_timesteps)
