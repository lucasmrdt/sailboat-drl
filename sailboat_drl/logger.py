from gymnasium import Wrapper
from stable_baselines3.common.logger import configure, HParam
from collections.abc import Iterable

from .cli import args

_sb3_logger = None
_logger_name = None


class Logger:
    @staticmethod
    def configure(name: str):
        global _sb3_logger, _logger_name
        _logger_name = name
        _sb3_logger = configure(
            f'runs/{args.name}/{name}/0', ['tensorboard', 'csv'])

    @staticmethod
    def re_configure(postfix):
        global _sb3_logger
        _sb3_logger = configure(
            f'runs/{args.name}/{_logger_name}/{postfix}', ['tensorboard', 'csv'])

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
    def log_hyperparams(hyperparams={}, metrics={}):
        global _sb3_logger
        assert _sb3_logger is not None, 'Logger not configured, call Logger.configure() first'
        hyperparams = HParam(hyperparams, metrics)
        _sb3_logger.record('hyperparams', hyperparams, exclude=('csv',))
        Logger.dump(0)


class LoggerDumpWrapper(Wrapper):
    def __init__(self, eval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval = eval
        self.n_steps = 0
        self.n_episodes = 0
        self.is_reset = False

    def reset(self, **kwargs):
        self.is_reset = True
        self.n_steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        if self.is_reset:
            Logger.re_configure(str(self.n_episodes))
            self.n_episodes += 1
            self.is_reset = False
        self.n_steps += 1
        observation, reward, terminated, truncated, info = super().step(action)
        Logger.dump(step=self.n_steps)
        return observation, reward, terminated, truncated, info
