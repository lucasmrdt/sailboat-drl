from .path_planning_rewards import *
from .path_following_rewards import *
from .abc_reward import *

available_path_planning_rewards = {
    'pp_sparse': PPSparseReward,
    'pp_dist_to_target': PPDistToTargetReward,
    'pp_gained_dist_to_target': PPGainedDistToTargetReward,
    'pp_velocity': PPVelocityReward,
}

available_path_following_rewards = {
    # 'pf_ours': PFOursReward,
    'pf_max_vmc': PFMaxVMC,
}

available_rewards = {
    **available_path_planning_rewards,
    **available_path_following_rewards,
}

available_renderer = {
    **{k: PPRenderer for k in available_path_planning_rewards},
    **{k: PFRenderer for k in available_path_following_rewards},
}
