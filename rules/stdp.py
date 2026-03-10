import numpy as np


class STDP:
    def __init__(
        self,
        num_pre: int,
        num_post: int,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max

        self.t_pre = np.full(num_pre, -1e9)
        self.t_post = np.full(num_post, -1e9)

    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        weights: np.ndarray,
        t: float,
    ):
        delta_t = self.t_post[:, np.newaxis] - self.t_pre[np.newaxis, :]

        post_mask = post_spikes[:, np.newaxis]
        ltp = self.A_plus * np.exp(-delta_t / self.tau_plus) * post_mask
        ltp[delta_t <= 0] = 0.0

        pre_mask = pre_spikes[np.newaxis, :]
        ltd = -self.A_minus * np.exp(delta_t / self.tau_minus) * pre_mask
        ltd[delta_t >= 0] = 0.0

        weights += ltp + ltd
        np.clip(weights, self.w_min, self.w_max, out=weights)

        self.t_pre[pre_spikes > 0] = t
        self.t_post[post_spikes > 0] = t
