from sailboat_drl import prepare_env
from sailboat_gym import EPISODE_LENGTH
import time
import numpy as np

env = prepare_env('0', eval=True)()
obs, _ = env.reset(seed=0)

for _ in range(EPISODE_LENGTH):
    # act = env.action_space.sample()
    act = 0
    obs, reward, terminated, truncated, info = env.step(act)

env.close()
