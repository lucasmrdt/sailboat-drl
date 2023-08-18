from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from torch import nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sailboat_gym import env_by_name
import numpy as np
import uuid
import pickle

from .env import create_env, available_act_wrappers, available_obs_wrappers, available_rewards
from .callbacks import TimeLoggerCallback
from .logger import Logger


def parse_args(overwrite_args={}):
    def extended_eval(s):
        return eval(s, {'pi': np.pi, 'nn': nn})

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name',
                        type=str, default=str(uuid.uuid4()), help='experiment name')
    parser.add_argument('--seed',
                        type=int, default=0, help='random seed')
    parser.add_argument('--env-name', choices=list(env_by_name.keys()),
                        default=list(env_by_name.keys())[0], help='environment name')
    parser.add_argument('--obs', choices=list(available_obs_wrappers.keys()),
                        default=list(available_obs_wrappers.keys())[0], help='observation used by the agent')
    parser.add_argument('--act', choices=list(available_act_wrappers.keys()),
                        default=list(available_act_wrappers.keys())[0], help='action used by the agent')
    parser.add_argument('--reward', choices=list(available_rewards.keys()),
                        default=list(available_rewards.keys())[0], help='reward function')
    parser.add_argument('--reward-kwargs', type=extended_eval,
                        default={}, help='reward function arguments')
    parser.add_argument('--episode-duration', type=int,
                        default=100, help='episode duration (in seconds)')
    parser.add_argument('--n-envs', type=int, default=30,
                        help='number of environments')
    parser.add_argument('--wind-speed', type=float, default=1,
                        help='wind speed')
    parser.add_argument('--water-current', type=extended_eval,
                        default=[0, 0], help='water current')
    parser.add_argument('--keep-sim-running', action='store_true',
                        help='keep the simulator running after training')

    # stable-baselines3 arguments
    parser.add_argument('--n-steps', type=int, default=1000,
                        help='number of steps')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='batch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--ent-coef', type=float, default=0.0,
                        help='entropy coefficient')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='clip range')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max grad norm')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help='value function coefficient')
    parser.add_argument('--policy-kwargs', type=extended_eval,
                        default={'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': False}, help='policy kwargs')
    parser.add_argument('--total-steps', type=int, default=1e6,
                        help='total steps')
    args, unknown = parser.parse_known_args()

    args.__dict__ = {k: v for k, v in vars(args).items()
                     if k not in overwrite_args}
    args.__dict__.update(overwrite_args)
    return args


def prepare_env(args, env_idx=0, is_eval=False):
    def _init():
        no_go_zone = np.deg2rad(30)
        thetas = np.linspace(-np.pi + no_go_zone,
                             np.pi - no_go_zone,
                             args.n_envs,
                             endpoint=True)
        return create_env(env_idx=env_idx,
                          is_eval=is_eval,
                          wind_speed=args.wind_speed,
                          water_current=args.water_current,
                          theta_wind=thetas[env_idx],
                          reward=args.reward,
                          reward_kwargs=args.reward_kwargs,
                          obs=args.obs,
                          act=args.act,
                          env_name=args.env_name,
                          seed=args.seed,
                          episode_duration=args.episode_duration,
                          prepare_env_for_nn=True,
                          keep_sim_running=args.keep_sim_running,
                          logger_prefix=args.name)
    return _init


def train_model(overwrite_args={}):
    args = parse_args(overwrite_args)

    print('Training with the following arguments:')
    for k, v in vars(args).items():
        print(f'{k} = {v}')

    Logger.configure(f'{args.name}/train.py')

    env = SubprocVecEnv(
        [prepare_env(args, i) for i in range(args.n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma)

    model = PPO('MlpPolicy',
                env,
                n_steps=max(1, args.n_steps // args.n_envs),
                batch_size=args.batch_size,
                gamma=args.gamma,
                learning_rate=args.learning_rate,
                ent_coef=args.ent_coef,
                clip_range=args.clip_range,
                n_epochs=args.n_epochs,
                gae_lambda=args.gae_lambda,
                max_grad_norm=args.max_grad_norm,
                vf_coef=args.vf_coef,
                policy_kwargs=args.policy_kwargs,
                seed=args.seed)
    model.set_logger(Logger.get_sb3_logger())

    time_cb = TimeLoggerCallback(n_steps_by_rollout=args.n_steps,
                                 n_steps_per_second=env_by_name[args.env_name].NB_STEPS_PER_SECONDS)

    model.learn(args.total_steps,
                callback=[time_cb],
                progress_bar=True)
    model.save(f'runs/{args.name}/final.model.zip')
    env.save(f'runs/{args.name}/final.envstats.pkl')
    pickle.dump(args, open(f'runs/{args.name}/final.args.pkl', 'wb'))

    env.close()

    hparams = {k: v if isinstance(v, (int, float, str, bool)) else str(v)
               for k, v in vars(args).items()}
    Logger.log_hyperparams(hparams, {'last_mean_reward': -1})
