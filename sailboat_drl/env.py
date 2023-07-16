import wandb
import numpy as np
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward
from stable_baselines3.common.monitor import Monitor

from .rewards import PFMaxVMC, PFCircularCamille, PFRenderer, PPSparseReward, PPDistToTargetReward, PPGainedDistToTargetReward, PPVelocityReward, PPRenderer
from .wrappers import CbWrapper, CustomRecordVideo, Basic2DObs, RudderAngleAction, RudderForceAction
from .cli import args, runtime_env
from .logger import Logger, LoggerDumpWrapper

available_path_planning_rewards = {
    'pp_sparse': PPSparseReward,
    'pp_dist_to_target': PPDistToTargetReward,
    'pp_gained_dist_to_target': PPGainedDistToTargetReward,
    'pp_velocity': PPVelocityReward,
}

available_path_following_rewards = {
    'pf_max_vmc': PFMaxVMC,
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
}

assert args.reward in available_rewards, f'unknown reward {args.reward} in {available_rewards.keys()}'
assert args.act in available_act_wrappers, f'unknown act wrapper {args.act} in {available_act_wrappers.keys()}'
assert args.obs in available_obs_wrappers, f'unknown obs wrapper {args.obs} in {available_obs_wrappers.keys()}'


def prepare_env(env_idx=0, eval=False, record=False):
    def wind_generator_fn(seed: int | None):
        if eval:
            thetas = np.linspace(-np.pi, 0, args.n_eval_envs, endpoint=False)
        else:
            thetas = np.linspace(-np.pi, np.pi, args.n_train_envs, endpoint=False)
        theta_wind = thetas[env_idx]
        wind_speed = 2
        return np.array([np.cos(theta_wind), np.sin(theta_wind)])*wind_speed

    def __init():

        name = f'{"eval" if eval else "train"}-{env_idx}'

        Logger.configure(name)

        Reward = available_rewards[args.reward]
        Renderer = available_renderer[args.reward]
        ObsWrapper = available_obs_wrappers[args.obs]
        ActWrapper = available_act_wrappers[args.act]

        reward = Reward(**args.reward_kwargs)

        env = gym.make(args.env_name,
                       renderer=Renderer(reward, padding=0),
                       reward_fn=reward,
                       wind_generator_fn=wind_generator_fn,
                       container_tag='mss1',
                       video_speed=20,
                       map_scale=.5,
                       name=f'{env_idx}' if args.use_same_sim else name)
        
        episode_duration = args.eval_episode_duration if eval else args.train_episode_duration
        env = TimeLimit(env,
                        max_episode_steps=episode_duration * runtime_env.nb_steps_per_second)

        if record:
            env = CustomRecordVideo(env,
                                   video_folder=f'runs/{args.name}/{name}/videos',
                                   episode_trigger=lambda _: True,
                                   video_length=0)

        env = ActWrapper(env)
        env = ObsWrapper(env, reward)
        env = FlattenObservation(env)
        env = NormalizeObservation(env)
        # if not eval:
        #     env = NormalizeReward(env)

        # env = Monitor(env)
        env = LoggerDumpWrapper(env)

        return env
    return __init
