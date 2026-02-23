def lif_step(v, current, dt, tau, v_rest, v_th=-55, v_reset=-75):
    dv = dt * (-(v - v_rest) / tau + current)
    v = v + dv
    spike = 0

    if v >= v_th:
        spike = 1
        v = v_reset

    return v, spike
