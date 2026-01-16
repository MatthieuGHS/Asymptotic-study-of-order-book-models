import numpy as np

# Model 1 simulation - Prop 1.3
from . import model1 as mod1

def simulate_vect_path(M, n_steps, N_agents, seed):
    x2 = mod1.x0*np.ones(M)
    rng = np.random.default_rng(seed)
    lam = N_agents*mod1.f

    for _ in range(n_steps):
        x2 += mod1.b(x2) + rng.poisson(lam, size=M)/N_agents

    return(x2)
