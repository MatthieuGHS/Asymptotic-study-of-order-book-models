import numpy as np

x0 = 500    # init of the limit system
xref = 500  # ref point = x0 -> no transition at the begining
drift = 0.2 # Lambda

# SIGMOID
# beta=0.05      # Beta : sigmoid (f) slope
# alpha=1        # Alpha : fix f(xref)
# M=1            # max for f (the sigmoid)

# def f(x):
#     z = -beta * (x - xref)
#     t = alpha * np.exp(z)
#     return(M * (1 - 1/(1+t)))

# For this model f=cst

# Poisson param
f=0.5           # = f(xref) with M=alpha=1

def b(x):
   return(-drift*(x-xref))

def simulate_limit(n_steps):
    path=np.empty(n_steps+1)
    path[0]=x0

    for i in range(n_steps):
        path[i+1]=path[i] + b(path[i]) + f

    return(path)

# the following is vecterized in simulation.py
def simulate_path(n_steps, N_agents, seed):
    rng = np.random.default_rng(seed)

    path=np.empty(n_steps+1)
    path[0]=x0

    for i in range(n_steps):
        path[i+1]=path[i]+b(path[i])+ rng.poisson(N_agents*f)/N_agents  # the poisson random var are iid

    return(path)
