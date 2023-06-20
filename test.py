from sailboat_drl import prepare_env
import time

env = prepare_env('0', record=True)()
obs, _ = env.reset(seed=0)

time.sleep(20)

for _ in range(100):
    act = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(act)

env.close()
