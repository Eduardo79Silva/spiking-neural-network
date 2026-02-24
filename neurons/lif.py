class LIFNeuron:
    def __init__(self, tau, v_rest, v_th, v_reset):
        self.v = v_rest
        self.tau = tau
        self.v_rest = v_rest
        self.v_th = v_th
        self.v_reset = v_reset

    def step(self, current, dt):
        dv = dt * (-(self.v - self.v_rest) / self.tau + current)
        self.v += dv
        spike = 0

        if self.v >= self.v_th:
            spike = 1
            self.v = self.v_reset

        return spike
