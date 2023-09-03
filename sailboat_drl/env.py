import numpy as np
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.flatten_observation import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from sailboat_gym import env_by_name
from functools import partial

from .weather_conditions import WindConstantGenerator, WindScenario1Generator, WaterCurrentNoneGenerator, WaterCurrentScenario1Generator, WindScenario2Generator, WaterCurrentScenario2Generator, WindScenario3Generator, WaterCurrentScenario3Generator
from .rewards import EvalReward, RewardRenderer, MaxDistReward, MaxDistRewardWithPenalty, MaxDistRewardWithPenaltyOnDerivative, MaxVMCWithPenality, MaxVMCWithPenalityAndDelta, MaxVMCWith2PenalityAndDelta, MaxVMCMinXTE, MaxVMCMinXTEPenalizeXTE, MaxVMCMinXTEMinDtRudder, MaxVMCPenalizeXTEMPenalizeDtRudder, MaxVMCPenalizeXTEMPenalizeDeltaRudder, MaxVMCCustomShape, MaxVMCCustomShapeV2, MaxVMCExpShape, MaxVMCLinearShape
from .wrappers import CustomRecordVideo, Basic2DObs, Basic2DObs_V2, Basic2DObs_V3, Basic2DObs_V4, Basic2DObs_V5, Basic2DObs_V6, RudderAngleAction, RudderForceAction, RawObs, CbWrapper
from .logger import Logger, LoggerWrapper

available_rewards = {
    'default': MaxDistReward,
    'max_dist': MaxDistReward,
    'max_dist_v1': MaxDistReward,
    'max_dist_v2': partial(MaxDistRewardWithPenalty, rudder_change_penalty=.1),
    # 0.05 is the max gain dist, 5 is the max derivative of the rudder angle
    'max_dist_v3': partial(MaxDistRewardWithPenaltyOnDerivative, rudder_change_penalty=.1 * 0.05 / (5**2)),
    # 0.4 is the max velocity
    'max_vmc_v1': partial(MaxVMCWithPenality, rudder_change_penalty=.001 * .4),
    'max_vmc_v2': partial(MaxVMCWithPenality, rudder_change_penalty=.005 * .4),
    'max_vmc_v3': partial(MaxVMCWithPenality, rudder_change_penalty=.01 * .4),
    'max_vmc_v4': partial(MaxVMCWithPenality, rudder_change_penalty=.05 * .4),
    'max_vmc_v5': partial(MaxVMCWithPenality, rudder_change_penalty=.1 * .4),
    'max_vmc_v6': partial(MaxVMCWithPenality, rudder_change_penalty=.5 * .4),

    # v3 seems to be the best
    'max_vmc_v7': partial(MaxVMCWithPenalityAndDelta, rudder_change_penalty=.01 * .4),

    'max_vmc_v8': partial(MaxVMCWith2PenalityAndDelta, rudder_change_penalty=.01 * .4, xte_penality=.01 * .4),
    'max_vmc_v9': partial(MaxVMCWith2PenalityAndDelta, rudder_change_penalty=.01 * .4, xte_penality=.005 * .4),
    'max_vmc_v10': partial(MaxVMCWith2PenalityAndDelta, rudder_change_penalty=.01 * .4, xte_penality=.001 * .4),

    'max_vmc_v11': partial(MaxVMCWithPenalityAndDelta, rudder_change_penalty=.05 * .4),
    'max_vmc_v12': partial(MaxVMCWithPenalityAndDelta, rudder_change_penalty=.1 * .4),
    'max_vmc_v13': partial(MaxVMCWithPenalityAndDelta, rudder_change_penalty=.5 * .4),

    'max_vmc_min_xte_v1': partial(MaxVMCMinXTE, vmc_coef=1, xte_coef=1),
    'max_vmc_min_xte_v2': partial(MaxVMCMinXTE, vmc_coef=1, xte_coef=.5),
    'max_vmc_min_xte_v3': partial(MaxVMCMinXTE, vmc_coef=1, xte_coef=.1),

    'max_vmc_min_xte_penalize_xte_v1': partial(MaxVMCMinXTEPenalizeXTE, vmc_coef=1, xte_coef=1),
    'max_vmc_min_xte_penalize_xte_v2': partial(MaxVMCMinXTEPenalizeXTE, vmc_coef=1, xte_coef=.5),
    'max_vmc_min_xte_penalize_xte_v3': partial(MaxVMCMinXTEPenalizeXTE, vmc_coef=1, xte_coef=.1),

    'max_vmc_min_xte_min_dt_rudder_v1': partial(MaxVMCMinXTEMinDtRudder, rudder_coef=1, vmc_coef=1, xte_coef=1),
    'max_vmc_min_xte_min_dt_rudder_v2': partial(MaxVMCMinXTEMinDtRudder, rudder_coef=.1, vmc_coef=1, xte_coef=1),
    'max_vmc_min_xte_min_dt_rudder_v3': partial(MaxVMCMinXTEMinDtRudder, rudder_coef=.01, vmc_coef=1, xte_coef=1),

    'max_vmc_penalize_xte_and_dt_rudder_v1': partial(MaxVMCPenalizeXTEMPenalizeDtRudder, rudder_coef=1, vmc_coef=1, xte_coef=1),
    'max_vmc_penalize_xte_and_dt_rudder_v2': partial(MaxVMCPenalizeXTEMPenalizeDtRudder, rudder_coef=.5, vmc_coef=1, xte_coef=1),
    'max_vmc_penalize_xte_and_dt_rudder_v3': partial(MaxVMCPenalizeXTEMPenalizeDtRudder, rudder_coef=.1, vmc_coef=1, xte_coef=1),
    'max_vmc_penalize_xte_and_dt_rudder_v4': partial(MaxVMCPenalizeXTEMPenalizeDtRudder, rudder_coef=2, vmc_coef=1, xte_coef=1),
    'max_vmc_penalize_xte_and_dt_rudder_v5': partial(MaxVMCPenalizeXTEMPenalizeDtRudder, rudder_coef=5, vmc_coef=1, xte_coef=1),
    'max_vmc_penalize_xte_and_dt_rudder_v6': partial(MaxVMCPenalizeXTEMPenalizeDtRudder, rudder_coef=10, vmc_coef=1, xte_coef=1),

    'max_vmc_penalize_xte_and_delta_rudder_v1': partial(MaxVMCPenalizeXTEMPenalizeDeltaRudder, rudder_coef=1, vmc_coef=1, xte_coef=1),
    'max_vmc_penalize_xte_and_delta_rudder_v2': partial(MaxVMCPenalizeXTEMPenalizeDeltaRudder, rudder_coef=.5, vmc_coef=1, xte_coef=1),
    'max_vmc_penalize_xte_and_delta_rudder_v3': partial(MaxVMCPenalizeXTEMPenalizeDeltaRudder, rudder_coef=.1, vmc_coef=1, xte_coef=1),

    'max_vmc_custom_shape_v1': partial(MaxVMCCustomShape, rudder_coef=.1, vmc_coef=1, vmc_a=4, xte_coef=1, xte_a=15),
    'max_vmc_custom_shape_v2': partial(MaxVMCCustomShape, rudder_coef=.1, vmc_coef=1, vmc_a=4, xte_coef=1, xte_a=10),
    'max_vmc_custom_shape_v3': partial(MaxVMCCustomShape, rudder_coef=.1, vmc_coef=1, vmc_a=1, xte_coef=1, xte_a=15),

    # > 78
    'max_vmc_custom_shape_v4': partial(MaxVMCCustomShape, rudder_coef=.2, vmc_coef=1, vmc_a=4, xte_coef=1, xte_a=15),
    'max_vmc_custom_shape_v5': partial(MaxVMCCustomShape, rudder_coef=.2, vmc_coef=1, vmc_a=4, xte_coef=1, xte_a=10),
    'max_vmc_custom_shape_v6': partial(MaxVMCCustomShape, rudder_coef=.2, vmc_coef=1, vmc_a=1, xte_coef=1, xte_a=15),

    'max_vmc_penalize_xte_and_delta_rudder_v4': MaxVMCPenalizeXTEMPenalizeDeltaRudder,

    'max_vmc_custom_shape_v7': partial(
        MaxVMCCustomShapeV2,
        rudder_coef=.2,
        vmc_coef=1,
        xte_coef=1,
        xte_params=dict(
            steepness=6,
        ),
        vmc_params=dict(
            start_penality=.5,
            steepness=6,
        )
    ),
    'max_vmc_custom_shape_tuning': MaxVMCCustomShapeV2,

    'max_vmc_custom_shape_s0_best_k5': partial(
        MaxVMCCustomShapeV2,
        rudder_coef=.75,
        vmc_coef=1,
        xte_coef=1,
        xte_params=dict(
            steepness=19,
        ),
        vmc_params=dict(
            start_penality=.5,
            steepness=15,
        )
    ),
    'max_vmc_custom_shape_s1_best_k5': partial(
        MaxVMCCustomShapeV2,
        rudder_coef=.5,
        vmc_coef=1,
        xte_coef=1,
        xte_params=dict(
            steepness=5.5,
        ),
        vmc_params=dict(
            start_penality=.5,
            steepness=3.8,
        )
    ),
    'max_vmc_custom_shape_s2_best_k5': partial(
        MaxVMCCustomShapeV2,
        rudder_coef=.1,
        vmc_coef=1,
        xte_coef=1,
        xte_params=dict(
            steepness=4.6,
        ),
        vmc_params=dict(
            start_penality=.7,
            steepness=1,
        )
    ),

    'max_vmc_custom_shape_s1_best_k5_1': partial(
        MaxVMCCustomShapeV2,
        rudder_coef=.0595,
        vmc_coef=.568,
        xte_coef=.00254,
        xte_params=dict(
            steepness=2.56,
        ),
        vmc_params=dict(
            start_penality=.384,
            steepness=1.01,
        )
    ),
    'max_vmc_custom_shape_s1_best_k5_2': partial(
        MaxVMCCustomShapeV2,
        rudder_coef=.1,
        vmc_coef=1,
        xte_coef=1,
        xte_params=dict(
            steepness=2.56,
        ),
        vmc_params=dict(
            start_penality=.384,
            steepness=1.01,
        )
    ),
    'max_vmc_custom_shape_s1_best_k5_3': partial(
        MaxVMCCustomShapeV2,
        rudder_coef=.0514,
        vmc_coef=1,
        xte_coef=1,
        xte_params=dict(
            steepness=5.48,
        ),
        vmc_params=dict(
            start_penality=.841,
            steepness=3.82,
        )
    ),

    'max_vmc_exp_shape_s1_best_k5_1': partial(
        MaxVMCExpShape,
        rudder_coef=.00769,
        vmc_coef=.198,
        xte_coef=.135,
    ),
    'max_vmc_exp_shape_s1_best_k5_2': partial(
        MaxVMCExpShape,
        rudder_coef=.00379,
        vmc_coef=.0692,
        xte_coef=.0015,
    ),

    'max_vmc_linear_shape_s1_best_k5_1': partial(
        MaxVMCLinearShape,
        rudder_coef=.00554,
        vmc_coef=.0526,
        xte_coef=.00234,
    ),
    'max_vmc_linear_shape_s1_best_k5_2': partial(
        MaxVMCLinearShape,
        rudder_coef=.00147,
        vmc_coef=.0436,
        xte_coef=.00141,
    ),

    'max_vmc_exp_shape_full_tuning': MaxVMCExpShape,
    'max_vmc_linear_shape_full_tuning': MaxVMCLinearShape,
    'max_vmc_custom_shape_full_tuning': MaxVMCCustomShapeV2,
}

available_act_wrappers = {
    'default': RudderAngleAction,
    'rudder_angle_act': RudderAngleAction,
    'rudder_force_act': RudderForceAction,
}

available_obs_wrappers = {
    'default': RawObs,
    'basic_2d_obs': Basic2DObs,
    'basic_2d_obs_v2': Basic2DObs_V2,
    'basic_2d_obs_v3': Basic2DObs_V3,
    'basic_2d_obs_v4': Basic2DObs_V4,
    'basic_2d_obs_v5': Basic2DObs_V5,
    'basic_2d_obs_v6': Basic2DObs_V6,
    'raw_obs': RawObs,
}

available_envs = list(env_by_name.keys())

available_wind_generators = {
    'default': WindConstantGenerator,
    'constant': WindConstantGenerator,
    'scenario_1': WindScenario1Generator,
    'scenario_2': WindScenario2Generator,
    'scenario_3': WindScenario3Generator,
}

available_water_current_generators = {
    'default': WaterCurrentNoneGenerator,
    'none': WaterCurrentNoneGenerator,
    'scenario_1': WaterCurrentScenario1Generator,
    'scenario_2': WaterCurrentScenario2Generator,
    'scenario_3': WaterCurrentScenario3Generator,
}


def create_env(env_id='0', is_eval=False, wind_dir=np.pi / 2, water_current_dir=np.pi / 2, reward='default', reward_kwargs={}, obs='default', act='default', env_name='SailboatLSAEnv-v0', seed=None, episode_duration=100, prepare_env_for_nn=True, logger_prefix='default', keep_sim_running=False, wind_generator='default', water_current_generator='default', container_tag='mss1-ode'):
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
    np.random.seed(seed)

    Reward = available_rewards[reward]
    ObsWrapper = available_obs_wrappers[obs]
    ActWrapper = available_act_wrappers[act]
    WindGenerator = available_wind_generators[wind_generator]
    WaterCurrentGenerator = available_water_current_generators[water_current_generator]

    # where the wind is coming from
    wind_theta = (wind_dir + np.pi) % (2 * np.pi)

    wind_generator = WindGenerator(wind_theta)
    water_current_generator = WaterCurrentGenerator(water_current_dir)

    if is_eval:
        reward = EvalReward(Reward, **reward_kwargs)
    else:
        reward = Reward(**reward_kwargs)

    env = gym.make(env_name,
                   renderer=RewardRenderer(reward, padding=30),
                   reward_fn=reward.reward_fn,
                   stop_condition_fn=reward.stop_condition_fn,
                   wind_generator_fn=wind_generator.get_force,
                   water_generator_fn=water_current_generator.get_force,
                   container_tag=container_tag,
                   video_speed=10,
                   map_scale=.5,
                   keep_sim_alive=keep_sim_running,
                   name=name)
    env = CbWrapper(env,
                    on_reset=reward.on_reset,
                    on_step=reward.on_step)
    env = TimeLimit(env,
                    max_episode_steps=episode_duration * nb_steps_per_second)

    if is_eval:
        env = CustomRecordVideo(env,
                                video_folder=f'runs/{log_name}/videos',
                                episode_trigger=lambda _: True,
                                video_length=0)

    env = ActWrapper(env, wind_theta)
    env = ObsWrapper(env, reward)

    if prepare_env_for_nn:
        env = FlattenObservation(env)

    env = Monitor(env)
    env = LoggerWrapper(env)

    return env
