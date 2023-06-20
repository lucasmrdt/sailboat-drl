from .path_planning_rewards import *
from .abc_reward import *

available_rewards = {
    'path_planning_sparse_1': PathPlanningSparse1Reward,
    'path_planning_dense_1': PathPlanningDense1Reward,
}

available_renderer = {
    'path_planning_sparse_1': PathPlanningRenderer,
    'path_planning_dense_1': PathPlanningRenderer,
}
