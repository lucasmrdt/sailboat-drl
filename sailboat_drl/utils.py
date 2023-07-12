import time
import threading
import subprocess
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

def _force_kill_wandb():
    subprocess.run(['pkill', '-9', 'wandb-service'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def timeout_wandb(seconds: float):
    threading.Timer(seconds, _force_kill_wandb).start()