# class PFSparseReward(AbcPFReward):
#     def __call__(self, obs, act, next_obs):
#         xte = self._compute_xte(next_obs)
#         vmc = self._compute_vmc(next_obs)
#         return -(self.k1 * xte**2 + self.k2 * vmc**2)**.5


# class PFDerajEtAl2022Reward(AbcPFReward):
#     def __init__(self, k1=1, k2=1, **kwargs):
#         super().__init__(**kwargs)
#         self.k1 = k1
#         self.k2 = k2

#     def __call__(self, obs, act, next_obs):
#         xte = self._compute_xte(next_obs)
#         vmc = self._compute_vmc(next_obs)
#         vmc_bar = (vmc - 1) / 2
#         return np.exp(-self.k1 * xte**2) + np.exp(self.k2 * vmc_bar**2)


# class PFMaxVMCContinuity(AbcPFReward):
#     def __call__(self, obs, act, next_obs):
#         vmc = self._compute_vmc(next_obs)
#         delta_theta_rudder = (
#             obs['theta_rudder'].item() - next_obs['theta_rudder'].item())**2
#         return vmc - .1 * delta_theta_rudder


# class PFCircularCamille(AbcPFReward):
#     def __init__(self, max_vmc=.50, k1=5, **kwargs):
#         super().__init__(**kwargs)
#         self.max_vmc = max_vmc
#         self.k1 = k1

#     def __call__(self, obs, act, next_obs):
#         xte = self._compute_xte(next_obs)
#         vmc = self._compute_vmc(next_obs)
#         return -(self.k1 * (self.max_vmc - vmc)**2 + xte**2)**.5
