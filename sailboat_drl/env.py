import numpy as np
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.flatten_observation import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from sailboat_gym import env_by_name

from .rewards import PFMaxVMC, PFMaxVMCContinuity, PFCircularCamille, PFRenderer, PPSparseReward, PPDistToTargetReward, PPGainedDistToTargetReward, PPVelocityReward, PPRenderer
from .wrappers import CustomRecordVideo, Basic2DObs, Basic2DObs_V2, RudderAngleAction, RudderForceAction, PersistentNormalizeObservation
from .logger import Logger, LoggerDumpWrapper

available_path_planning_rewards = {
    'pp_sparse': PPSparseReward,
    'pp_dist_to_target': PPDistToTargetReward,
    'pp_gained_dist_to_target': PPGainedDistToTargetReward,
    'pp_velocity': PPVelocityReward,
}

available_path_following_rewards = {
    'pf_max_vmc': PFMaxVMC,
    'pf_max_vmc_continuity': PFMaxVMCContinuity,
    'pf_circular_camille': PFCircularCamille,
}

available_rewards = {
    **available_path_planning_rewards,
    **available_path_following_rewards,
}

available_renderer = {
    **{k: PPRenderer for k in available_path_planning_rewards},
    **{k: PFRenderer for k in available_path_following_rewards},
}

available_act_wrappers = {
    'rudder_angle_act': RudderAngleAction,
    'rudder_force_act': RudderForceAction,
}

available_obs_wrappers = {
    'basic_2d_obs': Basic2DObs,
    'basic_2d_obs_v2': Basic2DObs_V2,
}


def create_env(env_idx=0, is_eval=False, wind_speed=2, n_envs=1, reward='pf_max_vmc', reward_kwargs={'path': [[0, .5], [1, .5]]}, obs='basic_2d_obs_v2', act='rudder_force_act', env_name=list(env_by_name.keys())[0], seed=None, episode_duration=100, prepare_env_for_nn=True, logger_prefix=None, keep_sim_running=False):
    nb_steps_per_second = env_by_name[env_name].NB_STEPS_PER_SECONDS

    assert reward in available_rewards, f'unknown reward {reward} in {available_rewards.keys()}'
    assert act in available_act_wrappers, f'unknown act wrapper {act} in {available_act_wrappers.keys()}'
    assert obs in available_obs_wrappers, f'unknown obs wrapper {obs} in {available_obs_wrappers.keys()}'

    def wind_generator_fn(_):
        # if seed is not None:
        #     np.random.seed(seed) # use global seed
        # if is_eval:
        #     thetas = np.linspace(-np.pi, 0, args.n_eval_envs, endpoint=False)
        # else:
        #     thetas = np.linspace(-np.pi, np.pi,
        #                          args.n_train_envs, endpoint=False)
        #     rand_translate = np.random.uniform(0, 2*np.pi)
        #     thetas += rand_translate  # add a random translation
        #     thetas = (thetas + np.pi) % (2*np.pi) - np.pi  # wrap to [-pi, pi]
        thetas = np.linspace(0 + 30, 360 - 30, n_envs, endpoint=True)
        thetas = np.deg2rad(thetas)
        theta_wind = thetas[env_idx]
        return np.array([np.cos(theta_wind), np.sin(theta_wind)])*wind_speed

    name = f'{"eval" if is_eval else "train"}-{env_idx}'
    log_name = f'{logger_prefix}/{name}'

    Logger.configure(log_name)

    Reward = available_rewards[reward]
    Renderer = available_renderer[reward]
    ObsWrapper = available_obs_wrappers[obs]
    ActWrapper = available_act_wrappers[act]

    reward = Reward(**reward_kwargs)

    env = gym.make(env_name,
                   renderer=Renderer(reward, padding=0),
                   reward_fn=reward,
                   wind_generator_fn=wind_generator_fn,
                   container_tag='mss1',
                   video_speed=20,
                   map_scale=.5,
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
        env = PersistentNormalizeObservation(env)

    env = Monitor(env)
    env = LoggerDumpWrapper(is_eval, env)

    return env


def save_env_wrappers(env, path):
    while hasattr(env, 'env'):
        if hasattr(env, 'save'):
            env.save(path)
        env = env.env


def load_env_wrappers(env, path):
    while hasattr(env, 'env'):
        if hasattr(env, 'load'):
            env.load(path)
        env = env.env
