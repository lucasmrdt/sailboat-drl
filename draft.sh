screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_dist_v3 --obs=basic_2d_obs_v4 --total=10000 --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A"
screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_dist_v3 --obs=basic_2d_obs_v4 --total=10000 --n-envs=7 --episode-duration=100 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A"
screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_dist_v3 --obs=basic_2d_obs_v4 --total=10000 --n-envs=7 --episode-duration=50 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A"

screen -d -S drl-0 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_dist_v3 --obs=basic_2d_obs_v4 --total=10000 --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B"
screen -d -S drl-1 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_dist_v3 --obs=basic_2d_obs_v4 --total=10000 --n-envs=7 --episode-duration=100 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B"
screen -d -S drl-2 -m python3 scripts/sb3_train.py --policy-kwargs="{'net_arch': dict(pi=[256, 256], vf=[256, 256]), 'activation_fn': nn.ReLU, 'ortho_init': False}" --batch-size=32 --n-steps=1024 --gamma=0.999 --gae-lambda=.9 --max-grad-norm=0.6 --learning-rate=3e-05 --vf-coef=0.2 --n-epochs=10 --wind=constant --water-current=none --wind-dirs="[45, 90, 135, 180, 225, 270, 315]" --reward=max_dist_v3 --obs=basic_2d_obs_v4 --total=10000 --n-envs=7 --episode-duration=50 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B"

zip -r 08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A.zip runs/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A/model_*
zip -r 08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A.zip runs/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A/model_*
zip -r 08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A.zip runs/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A/model_*
zip -r 08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B.zip runs/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B/model_*
zip -r 08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B.zip runs/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B/model_*
zip -r 08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B.zip runs/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B/model_*

unzip saved_models/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A.zip
unzip saved_models/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B.zip
unzip saved_models/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A.zip
unzip saved_models/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B.zip
unzip saved_models/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A.zip
unzip saved_models/08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B.zip

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

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A-994" --checkpoint-step=994
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A-3976" --checkpoint-step=3976
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A-6958" --checkpoint-step=6958
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A-9940" --checkpoint-step=9940

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A-994" --checkpoint-step=994
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A-3976" --checkpoint-step=3976
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A-6958" --checkpoint-step=6958
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1A-9940" --checkpoint-step=9940

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A-994" --checkpoint-step=994
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A-3976" --checkpoint-step=3976
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A-6958" --checkpoint-step=6958
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2A-9940" --checkpoint-step=9940


python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0A" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B-994" --checkpoint-step=994
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B-3976" --checkpoint-step=3976
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B-6958" --checkpoint-step=6958
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-0B-9940" --checkpoint-step=9940

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B-994" --checkpoint-step=994
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B-3976" --checkpoint-step=3976
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B-6958" --checkpoint-step=6958
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-1B-9940" --checkpoint-step=9940

python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B-994" --checkpoint-step=994
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B-3976" --checkpoint-step=3976
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B-6958" --checkpoint-step=6958
python3 scripts/sb3_eval.py --n-envs=7 --name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B" --log-name="08-30-rew_v3-obs_v4-scen_0-T_10000-env_7-ep-2B-9940" --checkpoint-step=9940
