import numpy as np
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.flatten_observation import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from sailboat_gym import env_by_name

from .weather_conditions import ConstantWindGenerator, NoWaterCurrentGenerator
from .rewards import RewardRenderer, MaxDistReward, MaxVMCReward
from .wrappers import CustomRecordVideo, Basic2DObs, Basic2DObs_V2, Basic2DObs_V3, RudderAngleAction, RudderForceAction, RawObs
from .logger import Logger, LoggerWrapper

available_rewards = {
    'max_dist': MaxDistReward,
    'max_vmc': MaxVMCReward,
}

available_act_wrappers = {
    'rudder_angle_act': RudderAngleAction,
    'rudder_force_act': RudderForceAction,
}

available_obs_wrappers = {
    'basic_2d_obs': Basic2DObs,
    'basic_2d_obs_v2': Basic2DObs_V2,
    'basic_2d_obs_v3': Basic2DObs_V3,
    'raw_obs': RawObs,
}

available_envs = list(env_by_name.keys())

available_wind_generators = {
    'constant': ConstantWindGenerator,
}

available_water_current_generators = {
    'none': NoWaterCurrentGenerator,
}


def create_env(env_id='0', is_eval=False, wind_speed=2, wind_dir=np.pi / 2, water_current_dir=np.pi / 2, water_current_speed=.01, reward='max_dist', reward_kwargs={'path': [[0, 0], [200, 0]], 'full_obs': True}, obs='raw_obs', act='rudder_angle_act', env_name='SailboatLSAEnv-v0', seed=None, episode_duration=100, prepare_env_for_nn=True, logger_prefix=None, keep_sim_running=False, wind_generator='constant', water_current_generator='none', container_tag='mss1-ode'):
    nb_steps_per_second = env_by_name[env_name].NB_STEPS_PER_SECONDS

    assert reward in available_rewards, f'unknown reward {reward} in {available_rewards.keys()}'
    assert act in available_act_wrappers, f'unknown act wrapper {act} in {available_act_wrappers.keys()}'
    assert obs in available_obs_wrappers, f'unknown obs wrapper {obs} in {available_obs_wrappers.keys()}'
    assert env_name in env_by_name, f'unknown env {env_name} in {env_by_name.keys()}'
    assert wind_generator in available_wind_generators, f'unknown wind generator {wind_generator} in {available_wind_generators.keys()}'
    assert water_current_generator in available_water_current_generators, f'unknown water current generator {water_current_generator} in {available_water_current_generators.keys()}'
    assert isinstance(wind_dir, float), \
        f'wind_dir must be a float, got {type(wind_dir)}'
    assert isinstance(water_current_dir, float), \
        f'water_current_dir must be a float, got {type(water_current_dir)}'

    name = f'{"eval" if is_eval else "train"}-{env_id}'
    log_name = f'{logger_prefix}/{name}'

    Logger.configure(log_name)

    Reward = available_rewards[reward]
    ObsWrapper = available_obs_wrappers[obs]
    ActWrapper = available_act_wrappers[act]
    WindGenerator = available_wind_generators[wind_generator]
    WaterCurrentGenerator = available_water_current_generators[water_current_generator]

    wind_generator_fn = WindGenerator(wind_dir, wind_speed)
    water_current_generator_fn = WaterCurrentGenerator(water_current_dir,
                                                       water_current_speed)
    reward = Reward(**reward_kwargs)

    env = gym.make(env_name,
                   renderer=RewardRenderer(reward, padding=30),
                   reward_fn=reward.reward_fn,
                   stop_condition_fn=reward.stop_condition_fn,
                   wind_generator_fn=wind_generator_fn,
                   water_generator_fn=water_current_generator_fn,
                   container_tag=container_tag,
                   video_speed=10,
                   map_scale=1,
                   keep_sim_alive=keep_sim_running,
                   name=name)
    env = TimeLimit(env,
                    max_episode_steps=episode_duration * nb_steps_per_second)

    if is_eval:
        env = CustomRecordVideo(env,
                                video_folder=f'runs/{log_name}/videos',
                                episode_trigger=lambda _: True,
                                video_length=0)

    env = ActWrapper(env)
    env = ObsWrapper(env, reward)

    if prepare_env_for_nn:
        env = FlattenObservation(env)

    env = Monitor(env)
    env = LoggerWrapper(env)

    return env
