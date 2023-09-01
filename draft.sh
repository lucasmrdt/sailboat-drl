python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_delta_rudder_v4 --obs=basic_2d_obs_v5 --total=200000 --n-envs=7 --name="09-01-max2xpenalize-v4-1" --reward_kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': False},'rudder_coef':0.1,'xte_coef':"


python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_custom_shape_v1 --obs=basic_2d_obs_v5 --total=10000 --n-envs=7 --name="09-01-custom-shape-1"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_custom_shape_v2 --obs=basic_2d_obs_v5 --total=10000 --n-envs=7 --name="09-01-custom-shape-2"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_custom_shape_v3 --obs=basic_2d_obs_v5 --total=10000 --n-envs=7 --name="09-01-custom-shape-3"





python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_delta_rudder_v3 --obs=basic_2d_obs_v5 --total=200000 --n-envs=7 --name="09-01-max2xpenalize-v3-1"

python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_delta_rudder_v1 --obs=basic_2d_obs_v5 --total=100000 --n-envs=7 --name="08-31-max2xpenalize-delta-1"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_delta_rudder_v2 --obs=basic_2d_obs_v5 --total=100000 --n-envs=7 --name="08-31-max2xpenalize-delta-2"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_delta_rudder_v3 --obs=basic_2d_obs_v5 --total=100000 --n-envs=7 --name="08-31-max2xpenalize-delta-3"

python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_dt_rudder_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-max2xpenalize-1"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_dt_rudder_v2 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-max2xpenalize-2"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_dt_rudder_v3 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-max2xpenalize-3"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_dt_rudder_v4 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-max2xpenalize-4"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_dt_rudder_v5 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-max2xpenalize-5"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_penalize_xte_and_dt_rudder_v6 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-max2xpenalize-6"


python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_min_xte_penalize_xte_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-maxminpenalize-A1"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_min_xte_penalize_xte_v2 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-maxminpenalize-A2"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_min_xte_penalize_xte_v3 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-maxminpenalize-A3"

python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_min_xte_min_dt_rudder_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-maxminminrudder-A1"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_min_xte_min_dt_rudder_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-maxminminrudder-A2"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_min_xte_min_dt_rudder_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-maxminminrudder-A3"

python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_min_xte_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-maxmin-A1"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_min_xte_v2 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-maxmin-A2"
python3 scripts/sb3_train_and_eval.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_min_xte_v3 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-31-maxmin-A3"

screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': True}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v11 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-F0"
screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': True}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v12 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-F1"
screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': True}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v13 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-F2"

screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': True}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v11 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-G0"
screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': True}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v12 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-G1"
screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': True}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v13 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-G2"


screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': True}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v7 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-D0"
screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v7 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-D1"
screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v7 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-D2"

screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}"  --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v8 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-E0"
screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}"  --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v9 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-E1"
screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v10 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-E2"


python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B1" && python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v2 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B2"

python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v3 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B3" && python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v4 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B4"

python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v5 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B5" && python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v6 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B6"


# screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B1"
# screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v2 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B2"

# screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v3 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B3"
# screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v4 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B4"

# screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v5 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B5"
# screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v6 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B6"

zip -r 08-30-D0.zip runs/08-30-D0/model_*
zip -r 08-30-D1.zip runs/08-30-D1/model_*
zip -r 08-30-D2.zip runs/08-30-D2/model_*
zip -r 08-30-E0.zip runs/08-30-E0/model_*
zip -r 08-30-E1.zip runs/08-30-E1/model_*
zip -r 08-30-E2.zip runs/08-30-E2/model_*

08-31-max2xpenalize-1
08-31-max2xpenalize-2
08-31-max2xpenalize-3
zip -r 08-31-max2xpenalize-1.zip runs/08-31-max2xpenalize-1/model_*
zip -r 08-31-max2xpenalize-2.zip runs/08-31-max2xpenalize-2/model_*
zip -r 08-31-max2xpenalize-3.zip runs/08-31-max2xpenalize-3/model_*
unzip saved_models/08-31-max2xpenalize-1.zip
unzip saved_models/08-31-max2xpenalize-2.zip
unzip saved_models/08-31-max2xpenalize-3.zip

99995
199990
299985
399980
499975
599970
699965
799960
899955
999950
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-99995" --checkpoint-step=99995
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-199990" --checkpoint-step=199990
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-299985" --checkpoint-step=299985
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-399980" --checkpoint-step=399980
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-499975" --checkpoint-step=499975
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-599970" --checkpoint-step=599970
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-699965" --checkpoint-step=699965
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-799960" --checkpoint-step=799960
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-899955" --checkpoint-step=899955
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-delta-2" --log-name="08-31-max2xpenalize-1M-999950" --checkpoint-step=999950


python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-1" --log-name="08-31-max2xpenalize-1-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-1" --log-name="08-31-max2xpenalize-1-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-1" --log-name="08-31-max2xpenalize-1-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-1" --log-name="08-31-max2xpenalize-1-5-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-2" --log-name="08-31-max2xpenalize-2-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-2" --log-name="08-31-max2xpenalize-2-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-2" --log-name="08-31-max2xpenalize-2-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-2" --log-name="08-31-max2xpenalize-2-5-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-3" --log-name="08-31-max2xpenalize-3-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-3" --log-name="08-31-max2xpenalize-3-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-3" --log-name="08-31-max2xpenalize-3-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-max2xpenalize-3" --log-name="08-31-max2xpenalize-3-5-99960" --checkpoint-step=99960


08-31-maxminpenalize-A1
08-31-maxminpenalize-A2
08-31-maxminpenalize-A3
08-31-maxminminrudder-A1
08-31-maxminminrudder-A2
08-31-maxminminrudder-A3
zip -r 08-31-maxminpenalize-A1.zip runs/08-31-maxminpenalize-A1/model_*
zip -r 08-31-maxminpenalize-A2.zip runs/08-31-maxminpenalize-A2/model_*
zip -r 08-31-maxminpenalize-A3.zip runs/08-31-maxminpenalize-A3/model_*
zip -r 08-31-maxminminrudder-A1.zip runs/08-31-maxminminrudder-A1/model_*
zip -r 08-31-maxminminrudder-A2.zip runs/08-31-maxminminrudder-A2/model_*
zip -r 08-31-maxminminrudder-A3.zip runs/08-31-maxminminrudder-A3/model_*
mv *.zip saved_models/
unzip saved_models/08-31-maxminpenalize-A1.zip
unzip saved_models/08-31-maxminpenalize-A2.zip
unzip saved_models/08-31-maxminpenalize-A3.zip
unzip saved_models/08-31-maxminminrudder-A1.zip
unzip saved_models/08-31-maxminminrudder-A2.zip
unzip saved_models/08-31-maxminminrudder-A3.zip

python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A1" --log-name="08-31-maxminpenalize-A1-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A1" --log-name="08-31-maxminpenalize-A1-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A1" --log-name="08-31-maxminpenalize-A1-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A1" --log-name="08-31-maxminpenalize-A1-5-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A2" --log-name="08-31-maxminpenalize-A2-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A2" --log-name="08-31-maxminpenalize-A2-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A2" --log-name="08-31-maxminpenalize-A2-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A2" --log-name="08-31-maxminpenalize-A2-5-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A3" --log-name="08-31-maxminpenalize-A3-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A3" --log-name="08-31-maxminpenalize-A3-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A3" --log-name="08-31-maxminpenalize-A3-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminpenalize-A3" --log-name="08-31-maxminpenalize-A3-5-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A1" --log-name="08-31-maxminminrudder-A1-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A1" --log-name="08-31-maxminminrudder-A1-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A1" --log-name="08-31-maxminminrudder-A1-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A1" --log-name="08-31-maxminminrudder-A1-5-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A2" --log-name="08-31-maxminminrudder-A2-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A2" --log-name="08-31-maxminminrudder-A2-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A2" --log-name="08-31-maxminminrudder-A2-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A2" --log-name="08-31-maxminminrudder-A2-5-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A3" --log-name="08-31-maxminminrudder-A3-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A3" --log-name="08-31-maxminminrudder-A3-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A3" --log-name="08-31-maxminminrudder-A3-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxminminrudder-A3" --log-name="08-31-maxminminrudder-A3-5-99960" --checkpoint-step=99960




08-31-maxmin-A1
08-31-maxmin-A2
08-31-maxmin-A3
zip -r 08-31-maxmin-A1.zip runs/08-31-maxmin-A1/model_*
zip -r 08-31-maxmin-A2.zip runs/08-31-maxmin-A2/model_*
zip -r 08-31-maxmin-A3.zip runs/08-31-maxmin-A3/model_*
mv *.zip saved_models/
unzip saved_models/08-31-maxmin-A1.zip
unzip saved_models/08-31-maxmin-A2.zip
unzip saved_models/08-31-maxmin-A3.zip

zip -r 08-30-C3-1.zip runs/08-30-C3-1/model_*
zip -r 08-30-C3-2.zip runs/08-30-C3-2/model_*
mv *.zip saved_models/

unzip saved_models/08-30-C3.zip 
unzip saved_models/08-30-C3-1.zip 
unzip saved_models/08-30-C3-2.zip 


unzip saved_models/08-30-D0.zip
unzip saved_models/08-30-D1.zip
unzip saved_models/08-30-D2.zip
unzip saved_models/08-30-E0.zip
unzip saved_models/08-30-E1.zip
unzip saved_models/08-30-E2.zip

unzip saved_models/08-30-B1.zip
unzip saved_models/08-30-B3.zip
unzip saved_models/08-30-B5.zip

unzip 08-30-B2.zip
unzip 08-30-B4.zip
unzip 08-30-B6.zip

# 994_steps.zip
# 1988_steps.zip
# 2982_steps.zip
# 3976_steps.zip
# 4970_steps.zip
# 5964_steps.zip
# 6958_steps.zip
# 7952_steps.zip
# 8946_steps.zip
# 9940_steps.zip

# 08-30-B1
# 08-30-B2
# 08-30-B3

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D0" --log-name="08-30-D0-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D0" --log-name="08-30-D0-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D0" --log-name="08-30-D0-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D0" --log-name="08-30-D0-99960" --checkpoint-step=99960
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D1" --log-name="08-30-D1-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D1" --log-name="08-30-D1-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D1" --log-name="08-30-D1-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D1" --log-name="08-30-D1-99960" --checkpoint-step=99960
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D2" --log-name="08-30-D2-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D2" --log-name="08-30-D2-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D2" --log-name="08-30-D2-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-D2" --log-name="08-30-D2-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-19992" --checkpoint-step=19992
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-29988" --checkpoint-step=29988
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-49980" --checkpoint-step=49980
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-59976" --checkpoint-step=59976
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-79968" --checkpoint-step=79968
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-89964" --checkpoint-step=89964
python3 scripts/sb3_eval.py --n-envs=7 --name="08-31-maxmin-A1" --log-name="08-31-maxmin-A1-5-99960" --checkpoint-step=99960

 extracting: runs/08-31-maxmin-A2/model_19992_steps.zip
 extracting: runs/08-31-maxmin-A2/model_29988_steps.zip
 extracting: runs/08-31-maxmin-A2/model_39984_steps.zip
 extracting: runs/08-31-maxmin-A2/model_49980_steps.zip
 extracting: runs/08-31-maxmin-A2/model_59976_steps.zip
 extracting: runs/08-31-maxmin-A2/model_69972_steps.zip
 extracting: runs/08-31-maxmin-A2/model_79968_steps.zip
 extracting: runs/08-31-maxmin-A2/model_89964_steps.zip
 extracting: runs/08-31-maxmin-A2/model_9996_steps.zip
 extracting: runs/08-31-maxmin-A2/model_99960_steps.zip

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-9996" --checkpoint-step=9996

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-100000"
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-19992" --checkpoint-step=19992
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-29988" --checkpoint-step=29988
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-49980" --checkpoint-step=49980
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-59976" --checkpoint-step=59976
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-79968" --checkpoint-step=79968
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-89964" --checkpoint-step=89964
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-89964" --checkpoint-step=89964

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E0" --log-name="08-30-E0-99960" --checkpoint-step=99960
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E1" --log-name="08-30-E1-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E1" --log-name="08-30-E1-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E1" --log-name="08-30-E1-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E1" --log-name="08-30-E1-99960" --checkpoint-step=99960
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E2" --log-name="08-30-E2-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E2" --log-name="08-30-E2-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E2" --log-name="08-30-E2-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-E2" --log-name="08-30-E2-99960" --checkpoint-step=99960


python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3" --log-name="08-30-C3-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3" --log-name="08-30-C3-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3" --log-name="08-30-C3-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3" --log-name="08-30-C3-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3-1" --log-name="08-30-C3-1-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3-1" --log-name="08-30-C3-1-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3-1" --log-name="08-30-C3-1-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3-1" --log-name="08-30-C3-1-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3-2" --log-name="08-30-C3-2-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3-2" --log-name="08-30-C3-2-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3-2" --log-name="08-30-C3-2-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-C3-2" --log-name="08-30-C3-2-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B3" --log-name="08-30-B3-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B3" --log-name="08-30-B3-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B3" --log-name="08-30-B3-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B3" --log-name="08-30-B3-99960" --checkpoint-step=99960

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B5" --log-name="08-30-B5-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B5" --log-name="08-30-B5-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B5" --log-name="08-30-B5-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B5" --log-name="08-30-B5-99960" --checkpoint-step=99960

#

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B2" --log-name="08-30-B2-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B4" --log-name="08-30-B4-9996" --checkpoint-step=9996
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B6" --log-name="08-30-B6-9996" --checkpoint-step=9996

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B2" --log-name="08-30-B2-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B4" --log-name="08-30-B4-39984" --checkpoint-step=39984
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B6" --log-name="08-30-B6-39984" --checkpoint-step=39984

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B2" --log-name="08-30-B2-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B4" --log-name="08-30-B4-69972" --checkpoint-step=69972
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B6" --log-name="08-30-B6-69972" --checkpoint-step=69972

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B2" --log-name="08-30-B2-99960" --checkpoint-step=99960
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B4" --log-name="08-30-B4-99960" --checkpoint-step=99960
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-B6" --log-name="08-30-B6-99960" --checkpoint-step=99960
