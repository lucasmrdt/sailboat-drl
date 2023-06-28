from .act_wrappers import *
from .obs_wrappers import *


available_obs_wrappers = {
    'only_reward_obs': OnlyRewardObs,
    'basic_2d_obs': Basic2DObs,
}

available_act_wrappers = {
    'rudder_angle_act': RudderAngleAction,
    'rudder_force_act': RudderForceAction,
}
