# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss1-bullet --name=eval-pid-mss1-bullet-n-envs-1 --wind-speed=8
# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss1-simbody --name=eval-pid-mss1-simbody-n-envs-1 --wind-speed=8
# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss1-dart --name=eval-pid-mss1-dart-n-envs-1 --wind-speed=8

python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss1-ode --name=eval-pid-mss1-upnquick-ode-n-envs-1 --wind-speed=8
python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-ode --name=eval-pid-mss1-upnquick-ode-n-envs-7 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7]"
python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-ode --name=eval-pid-mss1-upnquick-ode-n-envs-21 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23]"
python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-ode --name=eval-pid-mss1-upnquick-ode-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"

# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-bullet --name=eval-pid-mss1-bullet-n-envs-7 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-bullet --name=eval-pid-mss1-bullet-n-envs-21 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-bullet --name=eval-pid-mss1-bullet-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"

# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-simbody --name=eval-pid-mss1-simbody-n-envs-7 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-simbody --name=eval-pid-mss1-simbody-n-envs-21 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-simbody --name=eval-pid-mss1-simbody-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"

# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-dart --name=eval-pid-mss1-dart-n-envs-7 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-dart --name=eval-pid-mss1-dart-n-envs-21 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss1-dart --name=eval-pid-mss1-dart-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"


# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss2-ode --name=eval-pid-mss2-ode-n-envs-1 --wind-speed=8
# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss2-bullet --name=eval-pid-mss2-bullet-n-envs-1 --wind-speed=8
# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss2-simbody --name=eval-pid-mss2-simbody-n-envs-1 --wind-speed=8
# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss2-dart --name=eval-pid-mss2-dart-n-envs-1 --wind-speed=8

# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss4-ode --name=eval-pid-mss4-ode-n-envs-1 --wind-speed=8
# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss4-bullet --name=eval-pid-mss4-bullet-n-envs-1 --wind-speed=8
# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss4-simbody --name=eval-pid-mss4-simbody-n-envs-1 --wind-speed=8
# # python3 scripts/eval_pid.py --pid-algo=tae --container-tag=mss4-dart --name=eval-pid-mss4-dart-n-envs-1 --wind-speed=8


# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss2-ode --name=eval-pid-mss2-ode-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss2-bullet --name=eval-pid-mss2-bullet-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss2-simbody --name=eval-pid-mss2-simbody-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss2-dart --name=eval-pid-mss2-dart-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"

# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss4-ode --name=eval-pid-mss4-ode-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss4-bullet --name=eval-pid-mss4-bullet-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss4-simbody --name=eval-pid-mss4-simbody-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"
# python3 scripts/multi_eval_pid.py --pid-algo=tae --container-tag=mss4-dart --name=eval-pid-mss4-dart-n-envs-35 --wind-speed=8 --wind-dirs="[90.1,90.2,90.3,90.4,90.5,90.6,90.7,90.8,90.9,90.11,90.12,90.13,90.14,90.15,90.16,90.17,90.18,90.19,90.21,90.22,90.23,90.24,90.25,90.26,90.27,90.28,90.29,90.31,90.32,90.33,90.34,90.35,90.36,90.37,90.38]"