import numpy as np

from .utils import OscillationForceGenerator


class WindConstantGenerator(OscillationForceGenerator):
    def __init__(self, wind_theta, wind_speed):
        super().__init__(wind_theta,
                         wind_speed,
                         sigma_dir=0,
                         sigma_speed=0)


class WaterCurrentNoneGenerator(OscillationForceGenerator):
    def __init__(self, water_current_theta, water_current_speed):
        super().__init__(force_theta=0,
                         force_speed=0,
                         sigma_dir=0,
                         sigma_speed=0)


class WindScenario1Generator(OscillationForceGenerator):
    # 0% -> 25%:
    def __init__(self, wind_theta, wind_speed):
        super().__init__(wind_theta,
                         force_speed=1.5,
                         sigma_dir=np.deg2rad(5),
                         sigma_speed=0.2)


class WaterCurrentScenario1Generator(OscillationForceGenerator):
    # 0% -> 25%:
    def __init__(self, water_current_theta, water_current_speed):
        super().__init__(water_current_theta,
                         force_speed=0,
                         sigma_dir=0,
                         sigma_speed=0)


class WindScenario2Generator(OscillationForceGenerator):
    # 25% -> 50%:
    def __init__(self, wind_theta, wind_speed):
        super().__init__(wind_theta,
                         force_speed=1.5,
                         sigma_dir=np.deg2rad(10),
                         sigma_speed=0.2,
                         nb_oscilations=4)


class WaterCurrentScenario2Generator(OscillationForceGenerator):
    # 25% -> 50%:
    def __init__(self, water_current_theta, water_current_speed):
        super().__init__(water_current_theta,
                         force_speed=0.15,
                         sigma_dir=np.deg2rad(10),
                         sigma_speed=0.04,
                         nb_oscilations=4)


class WindScenario3Generator(OscillationForceGenerator):
    # 50% -> 75%:
    def __init__(self, wind_theta, wind_speed):
        super().__init__(wind_theta,
                         force_speed=1.5,
                         sigma_dir=np.deg2rad(20),
                         sigma_speed=0.2,
                         nb_oscilations=8)


class WaterCurrentScenario3Generator(OscillationForceGenerator):
    # 50% -> 75%:
    def __init__(self, water_current_theta, water_current_speed):
        super().__init__(water_current_theta,
                         force_speed=0.15,
                         sigma_dir=np.deg2rad(20),
                         sigma_speed=0.04,
                         nb_oscilations=8)
