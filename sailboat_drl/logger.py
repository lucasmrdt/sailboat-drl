import time
from gymnasium import Wrapper
from stable_baselines3.common.logger import configure
from collections.abc import Iterable

_sb3_logger = None
_logger_name = None


class Logger:
    @staticmethod
    def configure(name: str):
        global _sb3_logger, _logger_name
        _logger_name = name
        _sb3_logger = configure(f'runs/{name}/0',
                                ['tensorboard', 'csv'])

    @staticmethod
    def re_configure(postfix):
        global _sb3_logger
        _sb3_logger = configure(
            f'runs/{_logger_name}/{postfix}', ['tensorboard', 'csv'])

    @staticmethod
    def record(data={}, exclude=None):
        global _sb3_logger
        assert _sb3_logger is not None, 'Logger not configured, call Logger.configure() first'
        for k, v in data.items():
            if isinstance(v, Iterable) and not isinstance(v, str):
                for i, x in enumerate(v):
                    _sb3_logger.record(f'{k}/{i}', x, exclude=exclude)
            else:
                _sb3_logger.record(k, v, exclude=exclude)

    @staticmethod
    def dump(step: int):
        global _sb3_logger
        assert _sb3_logger is not None, 'Logger not configured, call Logger.configure() first'
        _sb3_logger.dump(step)

    @staticmethod
    def get_sb3_logger():
        global _sb3_logger
        assert _sb3_logger is not None, 'Logger not configured, call Logger.configure() first'
        return _sb3_logger

    @staticmethod
    def log_hyperparams(hyperparams={}, exclude=None):
        global _sb3_logger
        assert _sb3_logger is not None, 'Logger not configured, call Logger.configure() first'
        for k, v in hyperparams.items():
            if isinstance(v, Iterable):
                v = str(v)
            _sb3_logger.record(f'hyperparams/{k}', v, exclude=exclude)
        Logger.dump(0)


class LoggerWrapper(Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = 0
        self.n_episodes = 0
        self.n_steps_per_second = self.env.unwrapped.NB_STEPS_PER_SECONDS
        self.is_reset = False
        self.start_time = None

    def reset(self, **kwargs):
        self.is_reset = True
        self.n_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        if self.is_reset:
            Logger.re_configure(str(self.n_episodes))
            self.n_episodes += 1
            self.is_reset = False
            self.start_time = time.time()
        self.n_steps += 1

        observation, reward, terminated, truncated, info = super().step(action)

        Logger.record({
            "relative_time": time.time() - self.start_time,
            "reward": reward,
        })

        Logger.dump(step=self.n_steps)
        return observation, reward, terminated, truncated, info
