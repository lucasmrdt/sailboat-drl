import allow_local_package_imports

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from torch import nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sailboat_gym import env_by_name
import numpy as np
import uuid
import pickle

from sailboat_drl.env import create_env, available_act_wrappers, available_obs_wrappers, available_rewards, available_water_current_generators, available_wind_generators
from sailboat_drl.logger import Logger


def parse_args(overwrite_args={}):
    def extended_eval(s):
        return eval(s, {'pi': np.pi, 'nn': nn})

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name',
                        type=str, default=str(uuid.uuid4()), help='experiment name')
    parser.add_argument('--seed',
                        type=int, default=0, help='random seed')
    parser.add_argument('--env-name', choices=list(env_by_name.keys()),
                        default='SailboatLSAEnv-v0', help='environment name')
    parser.add_argument('--obs', choices=list(available_obs_wrappers.keys()),
                        default='basic_2d_obs_v3', help='observation used by the agent')
    parser.add_argument('--act', choices=list(available_act_wrappers.keys()),
                        default='rudder_angle_act', help='action used by the agent')
    parser.add_argument('--reward', choices=list(available_rewards.keys()),
                        default='max_dist', help='reward function')
    parser.add_argument('--reward-kwargs', type=extended_eval,
                        default={'path': [[0, 0], [100, 0]], 'xte_threshold': .1}, help='reward function arguments')
    parser.add_argument('--episode-duration', type=int,
                        default=200, help='episode duration (in seconds)')
    parser.add_argument('--n-envs', type=int, default=7,
                        help='number of environments')
    parser.add_argument('--water-current', choices=list(available_water_current_generators.keys()),
                        default='none', help='water current generator')
    parser.add_argument('--wind', choices=list(available_wind_generators.keys()),
                        default='constant', help='wind generator')
    parser.add_argument('--wind-dirs', type=eval, default=[45, 90, 135, 180, 225, 270, 315],
                        help='wind directions (in deg)')
    parser.add_argument('--water-current-dir', type=float, default=90,
                        help='water current direction (in deg)')
    parser.add_argument('--keep-sim-running', action='store_true',
                        help='keep the simulator running after training')
    parser.add_argument('--container-tag', type=str, default='mss1-ode',
                        help='container tag')
    parser.add_argument('--prefix-env-id', type=str,
                        help='prefix environment id')
    parser.add_argument('--disable-reward-normalization', action='store_true',
                        help='disable reward normalization')

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

    print('Unknown arguments:', unknown)

    args.__dict__ = {k: v for k, v in vars(args).items()
                     if k not in overwrite_args}
    args.__dict__.update(overwrite_args)
    return args


def prepare_env(args, env_id=0, is_eval=False):
    def deg2rad(deg):
        return np.deg2rad(deg) % (2 * np.pi)

    assert isinstance(args.wind_dirs, list), 'wind_dirs must be a list'

    def _init():
        wind_dir = args.wind_dirs[env_id % len(args.wind_dirs)]
        env_id_prefix = args.prefix_env_id if args.prefix_env_id is not None else args.name
        return create_env(env_id=f'{env_id_prefix}-{env_id}',
                          is_eval=is_eval,
                          water_current_generator=args.water_current,
                          water_current_dir=deg2rad(args.water_current_dir),
                          wind_generator=args.wind,
                          wind_dir=deg2rad(wind_dir),
                          reward=args.reward,
                          reward_kwargs=args.reward_kwargs,
                          act=args.act,
                          obs=args.obs,
                          container_tag=args.container_tag,
                          env_name=args.env_name,
                          keep_sim_running=args.keep_sim_running,
                          seed=args.seed + env_id,
                          episode_duration=args.episode_duration,
                          prepare_env_for_nn=True,
                          logger_prefix=args.name)
    return _init


def train_model(overwrite_args={}):
    args = parse_args(overwrite_args)
    assert isinstance(args.wind_dirs, list), \
        'wind_dirs must be a list'
    assert args.n_envs % len(args.wind_dirs) == 0, \
        'n_envs must be a multiple of len(wind_dirs)'

    print('Training with the following arguments:')
    for k, v in vars(args).items():
        print(f'{k} = {v}')

    Logger.configure(f'{args.name}/train.py')
    Logger.log_hyperparams(args.__dict__)

    env = SubprocVecEnv(
        [prepare_env(args, i) for i in range(args.n_envs)])
    env = VecNormalize(
        env, norm_obs=True, norm_reward=not args.disable_reward_normalization, gamma=args.gamma)

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
                seed=args.seed,
                tensorboard_log=f'runs/{args.name}')
    # model.set_logger(Logger.get_sb3_logger())

    pickle.dump(args, open(f'runs/{args.name}/model_args.pkl', 'wb'))

    save_freq = args.total_steps // 10
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, save_freq // args.n_envs),
        save_path=f'runs/{args.name}',
        name_prefix='model',
        save_vecnormalize=True,
    )

    model.learn(args.total_steps,
                callback=checkpoint_callback,
                progress_bar=True)
    model.save(f'runs/{args.name}/model_final.zip', )
    env.save(f'runs/{args.name}/model_vecnormalize_final.pkl')

    env.close()


if __name__ == '__main__':
    train_model()
