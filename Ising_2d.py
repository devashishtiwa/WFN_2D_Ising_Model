#2D Ising Model

import numpy as np

################################################################################################

def initial_state(L):
    state = np.random.choice([-1, 1], size=(L, L))
    return state

################################################################################################

def montecarlo(spinstate, L, h, beta):
    for _ in range(L * L):
        a = np.random.randint(0, L)
        b = np.random.randint(0, L)
        spin = spinstate[a, b]
        neighbor = (
            spinstate[(a + 1) % L, b]
            + spinstate[(a - 1) % L, b]
            + spinstate[a, (b + 1) % L]
            + spinstate[a, (b - 1) % L]
        )
        proposed_spin = -spin
        energy_cost = 2 * spin * (neighbor + h)
        
        if energy_cost <= 0 or np.random.random() < np.exp(-beta * energy_cost):
            spinstate[a, b] = proposed_spin
    
    return spinstate

################################################################################################

def energy_calculator(spinstate, L, h):
    energy = 0
    for i in range(L):
        for j in range(L):
            spin = spinstate[i, j]
            neighbor = (
                spinstate[(i + 1) % L, j]
                + spinstate[(i - 1) % L, j]
                + spinstate[i, (j + 1) % L]
                + spinstate[i, (j - 1) % L]
            )
            energy -= (neighbor + h) * spin
    return energy / 2

################################################################################################

def magnetization(spinstate):
    return np.sum(spinstate)

################################################################################################

