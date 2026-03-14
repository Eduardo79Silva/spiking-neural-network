class LIFNeuron:
    def __init__(self, v_rest, v_th, v_reset, tau, refractory_period):
        self.v_th = v_th
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.tau = tau
        self.refractory_period = refractory_period
        self.v = v_rest
        self.spike_time = -1
        self.refractory_end_time = -1

    def update(self, input_current, t):
        if t < self.refractory_end_time:
            return 0

        R_m = 10.0
        dv = (-(self.v - self.v_rest) + R_m * input_current) / self.tau
        self.v = self.v + dv

        if self.v >= self.v_th:
            self.spike_time = t
            self.v = self.v_reset
            self.refractory_end_time = t + self.refractory_period
            return 1
        return 0
