import allow_local_package_imports

import numpy as np
from sailboat_gym import Observation, get_best_sail

from sailboat_drl.env import create_env

theta_wind = np.deg2rad(90)


def ctrl(obs: Observation):
    wanted_heading = np.deg2rad(30)
    rudder_angle = obs['theta_boat'][2] - wanted_heading
    return [rudder_angle]


def run():
    env = create_env(is_eval=True, prepare_env_for_nn=False, wind_speed=4)

    obs, info = env.reset(seed=10)
    env.render()
    while True:
        obs, reward, terminated, truncated, info = env.step(ctrl(obs))
        if terminated:
            print('Terminated')
            break
        if truncated:
            print('Truncated')
            break
        env.render()

    env.close()


if __name__ == '__main__':
    run()
