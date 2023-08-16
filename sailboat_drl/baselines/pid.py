from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import type_aliases
from simple_pid import PID
from torch import nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sailboat_gym import env_by_name
import numpy as np

from ..env import create_env, available_act_wrappers, available_obs_wrappers, available_rewards
from ..logger import Logger
from ..utils import rotate_vector

def smallest_signed_angle(angle):
    """Transform an angle to be between -pi and pi"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class TAEModel(type_aliases.PolicyPredictor):
  def __init__(self, Kp, Ki, Kd, dt):
    self.pid = PID(Kp, Ki, Kd, setpoint=0)
    self.dt = dt

  def predict(self, obs, **_):
    assert 'tae' in obs, f'tae not in obs: {obs}'
    tae = obs['tae'][0]
    control = self.pid(tae, dt=self.dt)
    print(f'tae: {tae}, xte: {obs["xte"][0]} control: {control}')
    if control is None:
      action = 1 # noop
    elif control < 0:
      action = 0 # turn left
    else:
      action = 2 # turn right
    return [action], None # noop

class LOSModel(type_aliases.PolicyPredictor):
  def __init__(self, Kp, Ki, Kd, dt, path, radius):
    self.pid = PID(Kp, Ki, Kd, setpoint=0)
    self.dt = dt
    self.path = path
    self.radius = radius

  def compute_los(self, pos):
    p1, p2 = self.path - pos # center on pos
    dx, dy = p2 - p1 + 1e-8 # avoid div by 0
    dr = np.sqrt(dx**2 + dy**2)
    D = p1[0]*p2[1] - p2[0]*p1[1]
    delta = self.radius**2 * dr**2 - D**2

    def compute_intersect(sign=1):
      return np.array([
        D*dy + sign * np.sign(dy)*dx*np.sqrt(delta),
        -D*dx + sign * np.abs(dy)*np.sqrt(delta),
      ]) / dr**2

    if delta < 0:
      return []
    elif delta == 0:
      return [compute_intersect()]
    else:
      return [compute_intersect(-1), compute_intersect(1)]

  def predict(self, obs, **_):
    assert 'p_boat' in obs, f'p_boat not in obs: {obs}'
    assert 'xte' in obs, f'xte not in obs: {obs}'
    assert 'theta_boat' in obs, f'theta_boat not in obs: {obs}'
    pos = obs['p_boat'][0,0:2]
    vel = obs['dt_p_boat'][0,0:2]
    xte = obs['xte'][0,0]
    theta_boat = obs['theta_boat'][0,2] # z axis

    # transform relative velocity to world frame
    vel = rotate_vector(vel, theta_boat)

    los = np.array(self.compute_los(pos))

    if len(los) > 0:
      path_vec = self.path[1] - self.path[0]
      los = los[np.dot(los, path_vec) > 0] # keep only points in the path direction

    los_angles = np.arctan2(los[:,1], los[:,0]) if len(los) > 0 else np.array([-np.sign(xte)*np.pi/2])

    vel_angle = np.arctan2(vel[1], vel[0])
    vel_norm = np.linalg.norm(vel)
    ref_angle = vel_angle if vel_norm > 0.1 else theta_boat # if vel is too small, use boat angle
    angle_diffs = smallest_signed_angle(los_angles - ref_angle)

    angle_error = min(angle_diffs, key=abs)
    # angle_to_follow *= np.sign(xte)
    # angle_error = smallest_signed_angle(angle_to_follow - theta_boat)

    control = self.pid(angle_error, dt=self.dt)
    print(f'vel_angle: {vel_angle}, vel_norm: {vel_norm}, angle_error: {angle_error}, xte: {xte}, control: {control}')
    action = np.array([control])
    # if control is None:
    #   action = 1 # noop
    # elif control < 0:
    #   action = 0 # turn left
    # else:
    #   action = 2 # turn right
    return [action], None # noop

def get_args():
    def extended_eval(s):
        return eval(s, {'pi': np.pi, 'nn': nn})

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name',
                        type=str, required=True, help='experiment name')
    parser.add_argument('--seed',
                        type=int, default=0, help='random seed')
    parser.add_argument('--env-name', choices=list(env_by_name.keys()),
                        default=list(env_by_name.keys())[0], help='environment name')
    parser.add_argument('--act', choices=list(available_act_wrappers.keys()),
                        default=list(available_act_wrappers.keys())[0], help='action used by the agent')
    parser.add_argument('--reward', choices=list(available_rewards.keys()),
                        default=list(available_rewards.keys())[0], help='reward function')
    parser.add_argument('--reward-kwargs', type=extended_eval,
                        default={}, help='reward function arguments')
    parser.add_argument('--episode-duration', type=int,
                        default=100, help='episode duration (in seconds)')
    parser.add_argument('--n-envs', type=int, default=30,
                        help='number of environments')
    parser.add_argument('--wind-speed', type=float, default=1,
                        help='wind speed')
    args = parser.parse_args()

    return args

def prepare_env(args, env_idx=0, is_eval=False, theta_wind=np.pi/2):
    def _init():
        thetas = np.linspace(0 + 30, 360 - 30, args.n_envs, endpoint=True)
        thetas = np.deg2rad(thetas)
        return create_env(env_idx=env_idx,
                          is_eval=is_eval,
                          wind_speed=args.wind_speed,
                          theta_wind=theta_wind,
                          reward=args.reward,
                          reward_kwargs=args.reward_kwargs,
                          act=args.act,
                          env_name=args.env_name,
                          seed=args.seed,
                          keep_sim_running=True,
                          episode_duration=args.episode_duration,
                          prepare_env_for_nn=False,
                          logger_prefix=args.name)
    return _init

#  Info: {'map_bounds': array([[   0., -100.,    0.],
#        [ 200.,  100.,    1.]], dtype=float32)}
def run_pid(theta_wind, Kp, Ki, Kd):
    args = get_args()

    Logger.configure(f'{args.name}/eval.py')

    env = prepare_env(args, env_idx=0, is_eval=True, theta_wind=theta_wind)()

    nb_steps_per_seconds = env_by_name[args.env_name].NB_STEPS_PER_SECONDS
    dt = 1/nb_steps_per_seconds
    # model = TAEModel(Kp, Ki, Kd, dt)
    path = np.array(args.reward_kwargs['path'], dtype=np.float32)
    # map_bounds = np.array(args.reward_kwargs['map_bounds'])
    # path = path * (map_bounds[1] - map_bounds[0])[None,:] + map_bounds[0][None,:]
    model = LOSModel(Kp, Ki, Kd, dt, path, 5)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.n_envs)

    print(f'mean_reward = {mean_reward}')
    print(f'std_reward = {std_reward}')

    env.close()


def main():
    run_pid(np.pi/2, .1, 0.01, 0.005)