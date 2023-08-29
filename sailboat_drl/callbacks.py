import time
import pickle
from stable_baselines3.common.callbacks import BaseCallback

from .logger import Logger


class Callback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.step_count = 1

    def _on_step(self) -> bool:
        t = self.num_timesteps
        i = self.step_count
        if (i) * self.save_freq <= t <= (i + 1) * self.save_freq:
            print(f'Saving model at step {t}')
            self.model.save(
                f'{self.save_path}/step{self.step_count}_model.zip')
            self.env.save(
                f'{self.save_path}/step{self.step_count}_envstats.pkl')
            pickle.dump(args, open(
                f'{self.save_path}/step{self.step_count}_args.pkl', 'wb'))
            self.step_count += 1
        return True
