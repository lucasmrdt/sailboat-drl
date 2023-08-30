python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': True}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v7 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-D0"
python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': False}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v7 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-D1"
python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[64, 64], vf=[64, 64]), 'activation_fn': nn.Tanh, 'ortho_init': True}" --batch-size=16 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v7 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-D2"


python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B1" && python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v2 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B2"

python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v3 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B3" && python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v4 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B4"

python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v5 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B5" && python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v6 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B6"


# screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v1 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B1"
# screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v2 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B2"

# screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v3 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B3"
# screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v4 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B4"

# screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v5 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B5"
# screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_vmc_v6 --obs=basic_2d_obs_v4 --total=100000 --n-envs=7 --name="08-30-B6"

zip -r 08-30-C3.zip runs/08-30-C3/model_*
zip -r 08-30-C3-1.zip runs/08-30-C3-1/model_*
zip -r 08-30-C3-2.zip runs/08-30-C3-2/model_*
mv *.zip saved_models/

unzip saved_models/08-30-C3.zip 
unzip saved_models/08-30-C3-1.zip 
unzip saved_models/08-30-C3-2.zip 

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
