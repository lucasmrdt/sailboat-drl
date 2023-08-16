import numpy as np
from typing import Any


def norm(x: np.ndarray) -> np.floating[Any]:
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

def dict_without_keys(d: dict[Any, Any], keys: set[Any]) -> dict[Any, Any]:
    return {k: v for k, v in d.items() if k not in keys}

def smallest_signed_angle(angle):
    """Transform an angle to be between -pi and pi"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotate_vector(v, theta):
    """Rotate a vector by an angle theta"""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R@v