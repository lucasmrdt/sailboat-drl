python3 scripts/multi_eval_pid.py --wind-dirs="[30, 40, 51, 61, 71, 82, 92, 102, 113, 123, 133, 144, 154, 164, 175, 185, 196, 206, 216, 227, 237, 247, 258, 268, 278, 289, 299, 309, 320, 330]" --pid-algo=tae --Kp=0.5250113231150243 --Kd=0.00512388141150684 --Ki=0.0008589324677228019 --n=10 --name=pid-tae-median-median-top5

python3 scripts/multi_eval_pid.py --wind-dirs="[30, 40, 51, 61, 71, 82, 92, 102, 113, 123, 133, 144, 154, 164, 175, 185, 196, 206, 216, 227, 237, 247, 258, 268, 278, 289, 299, 309, 320, 330]" --pid-algo=tae --Kp=0.6403936503089891 --Kd=0.028641937203193798 --Ki=0.01012959196329757 --n=10 --name=pid-tae-mean-median-top5

`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.008354138390106647 --Kd=0.04048951099437157 --Ki=0.0002843557421585538 --wind-dir=30 --n=10 --name=pid-tae-top1-30deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.008354138390106647 --Kd=0.04048951099437157 --Ki=0.0002843557421585538 --wind-dir=30 --n=10 --name=pid-tae-median-top5-30deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.008354138390106647 --Kd=0.04048951099437157 --Ki=0.0002843557421585538 --wind-dir=30 --n=10 --name=pid-tae-top1-30deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.28014891825687316 --Kd=0.15919109694437725 --Ki=0.00029266585270133034 --wind-dir=40 --n=10 --name=pid-tae-top1-40deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.28014891825687316 --Kd=0.15919109694437725 --Ki=0.00029266585270133034 --wind-dir=40 --n=10 --name=pid-tae-median-top5-40deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.28014891825687316 --Kd=0.15919109694437725 --Ki=0.00029266585270133034 --wind-dir=40 --n=10 --name=pid-tae-top1-40deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.8038737978940386 --Kd=0.004847018513587718 --Ki=0.0008000557004617207 --wind-dir=51 --n=10 --name=pid-tae-top1-51deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.8038737978940386 --Kd=0.004847018513587718 --Ki=0.0008000557004617207 --wind-dir=51 --n=10 --name=pid-tae-median-top5-51deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.8038737978940386 --Kd=0.004847018513587718 --Ki=0.0008000557004617207 --wind-dir=51 --n=10 --name=pid-tae-top1-51deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.1053919066037832 --Kd=0.008062140123473355 --Ki=0.011195211987185049 --wind-dir=61 --n=10 --name=pid-tae-top1-61deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.1053919066037832 --Kd=0.008062140123473355 --Ki=0.011195211987185049 --wind-dir=61 --n=10 --name=pid-tae-median-top5-61deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.1053919066037832 --Kd=0.008062140123473355 --Ki=0.011195211987185049 --wind-dir=61 --n=10 --name=pid-tae-top1-61deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.017817414420222 --Kd=0.0011003170388651444 --Ki=0.002754183411448548 --wind-dir=71 --n=10 --name=pid-tae-top1-71deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.017817414420222 --Kd=0.0011003170388651444 --Ki=0.002754183411448548 --wind-dir=71 --n=10 --name=pid-tae-median-top5-71deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.017817414420222 --Kd=0.0011003170388651444 --Ki=0.002754183411448548 --wind-dir=71 --n=10 --name=pid-tae-top1-71deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.0784449449566653 --Kd=0.0012630674817451187 --Ki=0.04708238693584302 --wind-dir=82 --n=10 --name=pid-tae-top1-82deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.0784449449566653 --Kd=0.0012630674817451187 --Ki=0.04708238693584302 --wind-dir=82 --n=10 --name=pid-tae-median-top5-82deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.0784449449566653 --Kd=0.0012630674817451187 --Ki=0.04708238693584302 --wind-dir=82 --n=10 --name=pid-tae-top1-82deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.6548584707435811 --Kd=0.009553126955552821 --Ki=0.024642267559786036 --wind-dir=92 --n=10 --name=pid-tae-top1-92deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.6548584707435811 --Kd=0.009553126955552821 --Ki=0.024642267559786036 --wind-dir=92 --n=10 --name=pid-tae-median-top5-92deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.6548584707435811 --Kd=0.009553126955552821 --Ki=0.024642267559786036 --wind-dir=92 --n=10 --name=pid-tae-top1-92deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.5872608590514979 --Kd=0.0032846942842955536 --Ki=0.00586151487328055 --wind-dir=102 --n=10 --name=pid-tae-top1-102deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.5872608590514979 --Kd=0.0032846942842955536 --Ki=0.00586151487328055 --wind-dir=102 --n=10 --name=pid-tae-median-top5-102deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.5872608590514979 --Kd=0.0032846942842955536 --Ki=0.00586151487328055 --wind-dir=102 --n=10 --name=pid-tae-top1-102deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.9362848973472313 --Kd=0.005054690443626856 --Ki=0.0003488603961033602 --wind-dir=113 --n=10 --name=pid-tae-top1-113deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.9362848973472313 --Kd=0.005054690443626856 --Ki=0.0003488603961033602 --wind-dir=113 --n=10 --name=pid-tae-median-top5-113deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.9362848973472313 --Kd=0.005054690443626856 --Ki=0.0003488603961033602 --wind-dir=113 --n=10 --name=pid-tae-top1-113deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.39932300673119225 --Kd=0.002873434804611062 --Ki=0.0016744019539390478 --wind-dir=123 --n=10 --name=pid-tae-top1-123deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.39932300673119225 --Kd=0.002873434804611062 --Ki=0.0016744019539390478 --wind-dir=123 --n=10 --name=pid-tae-median-top5-123deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.39932300673119225 --Kd=0.002873434804611062 --Ki=0.0016744019539390478 --wind-dir=123 --n=10 --name=pid-tae-top1-123deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.19244089446502707 --Kd=0.02514010889756403 --Ki=0.0006362721983445181 --wind-dir=133 --n=10 --name=pid-tae-top1-133deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.19244089446502707 --Kd=0.02514010889756403 --Ki=0.0006362721983445181 --wind-dir=133 --n=10 --name=pid-tae-median-top5-133deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.19244089446502707 --Kd=0.02514010889756403 --Ki=0.0006362721983445181 --wind-dir=133 --n=10 --name=pid-tae-top1-133deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.2891792677718655 --Kd=0.001300172556186999 --Ki=0.00020974447495953935 --wind-dir=144 --n=10 --name=pid-tae-top1-144deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.2891792677718655 --Kd=0.001300172556186999 --Ki=0.00020974447495953935 --wind-dir=144 --n=10 --name=pid-tae-median-top5-144deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.2891792677718655 --Kd=0.001300172556186999 --Ki=0.00020974447495953935 --wind-dir=144 --n=10 --name=pid-tae-top1-144deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.24092897430141363 --Kd=0.003275351219606702 --Ki=0.00023149369177768162 --wind-dir=154 --n=10 --name=pid-tae-top1-154deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.24092897430141363 --Kd=0.003275351219606702 --Ki=0.00023149369177768162 --wind-dir=154 --n=10 --name=pid-tae-median-top5-154deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.24092897430141363 --Kd=0.003275351219606702 --Ki=0.00023149369177768162 --wind-dir=154 --n=10 --name=pid-tae-top1-154deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.322947351000719 --Kd=0.0009837245149051902 --Ki=0.0008861201964704737 --wind-dir=164 --n=10 --name=pid-tae-top1-164deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.322947351000719 --Kd=0.0009837245149051902 --Ki=0.0008861201964704737 --wind-dir=164 --n=10 --name=pid-tae-median-top5-164deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.322947351000719 --Kd=0.0009837245149051902 --Ki=0.0008861201964704737 --wind-dir=164 --n=10 --name=pid-tae-top1-164deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.4438070799517836 --Kd=0.07081113338734477 --Ki=0.002154011168538682 --wind-dir=175 --n=10 --name=pid-tae-top1-175deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.4438070799517836 --Kd=0.07081113338734477 --Ki=0.002154011168538682 --wind-dir=175 --n=10 --name=pid-tae-median-top5-175deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.4438070799517836 --Kd=0.07081113338734477 --Ki=0.002154011168538682 --wind-dir=175 --n=10 --name=pid-tae-top1-175deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.5229012843180046 --Kd=0.023885629358116136 --Ki=0.04694255553127333 --wind-dir=185 --n=10 --name=pid-tae-top1-185deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.5229012843180046 --Kd=0.023885629358116136 --Ki=0.04694255553127333 --wind-dir=185 --n=10 --name=pid-tae-median-top5-185deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.5229012843180046 --Kd=0.023885629358116136 --Ki=0.04694255553127333 --wind-dir=185 --n=10 --name=pid-tae-top1-185deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.13813300557112965 --Kd=0.00302492057746148 --Ki=0.0009662387178277916 --wind-dir=196 --n=10 --name=pid-tae-top1-196deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.13813300557112965 --Kd=0.00302492057746148 --Ki=0.0009662387178277916 --wind-dir=196 --n=10 --name=pid-tae-median-top5-196deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.13813300557112965 --Kd=0.00302492057746148 --Ki=0.0009662387178277916 --wind-dir=196 --n=10 --name=pid-tae-top1-196deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.23973290404368797 --Kd=0.040893425683324464 --Ki=0.00044934653798795897 --wind-dir=206 --n=10 --name=pid-tae-top1-206deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.23973290404368797 --Kd=0.040893425683324464 --Ki=0.00044934653798795897 --wind-dir=206 --n=10 --name=pid-tae-median-top5-206deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.23973290404368797 --Kd=0.040893425683324464 --Ki=0.00044934653798795897 --wind-dir=206 --n=10 --name=pid-tae-top1-206deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.14878156548756127 --Kd=0.0013379543061913773 --Ki=0.00023709298071527754 --wind-dir=216 --n=10 --name=pid-tae-top1-216deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.14878156548756127 --Kd=0.0013379543061913773 --Ki=0.00023709298071527754 --wind-dir=216 --n=10 --name=pid-tae-median-top5-216deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.14878156548756127 --Kd=0.0013379543061913773 --Ki=0.00023709298071527754 --wind-dir=216 --n=10 --name=pid-tae-top1-216deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.23373978830113765 --Kd=0.0065904623781573995 --Ki=0.0003829647231417219 --wind-dir=227 --n=10 --name=pid-tae-top1-227deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.23373978830113765 --Kd=0.0065904623781573995 --Ki=0.0003829647231417219 --wind-dir=227 --n=10 --name=pid-tae-median-top5-227deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.23373978830113765 --Kd=0.0065904623781573995 --Ki=0.0003829647231417219 --wind-dir=227 --n=10 --name=pid-tae-top1-227deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.30357955643514944 --Kd=0.08764225651666513 --Ki=0.0010482540507939125 --wind-dir=237 --n=10 --name=pid-tae-top1-237deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.30357955643514944 --Kd=0.08764225651666513 --Ki=0.0010482540507939125 --wind-dir=237 --n=10 --name=pid-tae-median-top5-237deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.30357955643514944 --Kd=0.08764225651666513 --Ki=0.0010482540507939125 --wind-dir=237 --n=10 --name=pid-tae-top1-237deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.7704326783215005 --Kd=0.008805189398862729 --Ki=0.0033413335673649043 --wind-dir=247 --n=10 --name=pid-tae-top1-247deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.7704326783215005 --Kd=0.008805189398862729 --Ki=0.0033413335673649043 --wind-dir=247 --n=10 --name=pid-tae-median-top5-247deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.7704326783215005 --Kd=0.008805189398862729 --Ki=0.0033413335673649043 --wind-dir=247 --n=10 --name=pid-tae-top1-247deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.4045553855594305 --Kd=0.06884793023075561 --Ki=0.0005561138217091762 --wind-dir=258 --n=10 --name=pid-tae-top1-258deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.4045553855594305 --Kd=0.06884793023075561 --Ki=0.0005561138217091762 --wind-dir=258 --n=10 --name=pid-tae-median-top5-258deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.4045553855594305 --Kd=0.06884793023075561 --Ki=0.0005561138217091762 --wind-dir=258 --n=10 --name=pid-tae-top1-258deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.7639406977480421 --Kd=0.012125679155391984 --Ki=0.10361740974491909 --wind-dir=268 --n=10 --name=pid-tae-top1-268deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.7639406977480421 --Kd=0.012125679155391984 --Ki=0.10361740974491909 --wind-dir=268 --n=10 --name=pid-tae-median-top5-268deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.7639406977480421 --Kd=0.012125679155391984 --Ki=0.10361740974491909 --wind-dir=268 --n=10 --name=pid-tae-top1-268deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.9751471592730317 --Kd=0.0008718262793620662 --Ki=0.002473059208641655 --wind-dir=278 --n=10 --name=pid-tae-top1-278deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.9751471592730317 --Kd=0.0008718262793620662 --Ki=0.002473059208641655 --wind-dir=278 --n=10 --name=pid-tae-median-top5-278deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.9751471592730317 --Kd=0.0008718262793620662 --Ki=0.002473059208641655 --wind-dir=278 --n=10 --name=pid-tae-top1-278deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.942330339212066 --Kd=0.0006872622632319778 --Ki=0.0005065060061247566 --wind-dir=289 --n=10 --name=pid-tae-top1-289deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.942330339212066 --Kd=0.0006872622632319778 --Ki=0.0005065060061247566 --wind-dir=289 --n=10 --name=pid-tae-median-top5-289deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.942330339212066 --Kd=0.0006872622632319778 --Ki=0.0005065060061247566 --wind-dir=289 --n=10 --name=pid-tae-top1-289deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.1183706413619943 --Kd=0.030238161568883714 --Ki=0.016277385344023596 --wind-dir=299 --n=10 --name=pid-tae-top1-299deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.1183706413619943 --Kd=0.030238161568883714 --Ki=0.016277385344023596 --wind-dir=299 --n=10 --name=pid-tae-median-top5-299deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=1.1183706413619943 --Kd=0.030238161568883714 --Ki=0.016277385344023596 --wind-dir=299 --n=10 --name=pid-tae-top1-299deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.8676574638586909 --Kd=0.0035448062393233708 --Ki=0.015229537488400887 --wind-dir=309 --n=10 --name=pid-tae-top1-309deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.8676574638586909 --Kd=0.0035448062393233708 --Ki=0.015229537488400887 --wind-dir=309 --n=10 --name=pid-tae-median-top5-309deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.8676574638586909 --Kd=0.0035448062393233708 --Ki=0.015229537488400887 --wind-dir=309 --n=10 --name=pid-tae-top1-309deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.527121361912044 --Kd=0.0011923787748941387 --Ki=0.00029983441974866987 --wind-dir=320 --n=10 --name=pid-tae-top1-320deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.527121361912044 --Kd=0.0011923787748941387 --Ki=0.00029983441974866987 --wind-dir=320 --n=10 --name=pid-tae-median-top5-320deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.527121361912044 --Kd=0.0011923787748941387 --Ki=0.00029983441974866987 --wind-dir=320 --n=10 --name=pid-tae-top1-320deg` &
`python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.028615385092928585 --Kd=0.20429963687171165 --Ki=0.0002002630940374524 --wind-dir=330 --n=10 --name=pid-tae-top1-330deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.028615385092928585 --Kd=0.20429963687171165 --Ki=0.0002002630940374524 --wind-dir=330 --n=10 --name=pid-tae-median-top5-330deg; python3 scripts/eval_pid.py --pid-algo=tae --Kp=0.028615385092928585 --Kd=0.20429963687171165 --Ki=0.0002002630940374524 --wind-dir=330 --n=10 --name=pid-tae-top1-330deg` &
