import gymnasium as gym
from gymnasium.core import Env


class CbWrapper(gym.Wrapper):
    def __init__(self, env: Env, on_reset=None, on_step=None, on_close=None, on_render=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.on_reset = on_reset
        self.on_step = on_step
        self.on_close = on_close
        self.on_render = on_render

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        if self.on_reset is not None:
            self.on_reset(obs, info)
        return obs, info

    def step(self, *args, **kwargs):
        obs, reward, done, truncated, info = super().step(*args, **kwargs)
        if self.on_step is not None:
            self.on_step(obs, reward, done, truncated, info)
        return obs, reward, done, truncated, info

    def render(self, *args, **kwargs):
        if self.on_render is not None:
            self.on_render()
        return super().render(*args, **kwargs)

    def close(self):
        if self.on_close is not None:
            self.on_close()
        return super().close()
