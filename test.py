import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import sailboat_gym

env = gym.make('SailboatLSAEnv-v0',
               renderer=sailboat_gym.CV2DRenderer(),
               container_tag='realtime')
env = RecordVideo(env, video_folder='./output/videos/')

env.reset(seed=10)

while True:
    act = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(act)
    if truncated:
        break
    env.render()

env.close()
