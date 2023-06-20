import numpy as np


def norm(x: np.ndarray) -> float:
    return np.linalg.norm(x, ord=2)


def normalize(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    min_pos, max_pos = bounds
    return (x - min_pos) / (max_pos - min_pos)  # [0, 1]


def is_env_instance(env, env_class):
    while env:
        if isinstance(env, env_class):
            return True
        env = env.env if hasattr(env, 'env') else None
    return False


def extract_env_instance(env, env_class):
    while env:
        if isinstance(env, env_class):
            return env
        env = env.env if hasattr(env, 'env') else None
    return None
