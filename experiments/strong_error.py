import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from orderBook import model1 as mod1
from orderBook import simulation as sim
from orderBook import estimators as estm

import matplotlib.pyplot as plt

N_lst = [10, 20, 50, 100, 200, 300, 500, 750, 1000, 1300, 1700, 2000,3000, 5000, 10000, 15000, 20000]
M=20000
n=200
xlim = mod1.simulate_limit(n)[-1]
seed = 0
errors = np.array([estm.strong_error_mc(sim.simulate_vect_path(M, n, N, seed),xlim) for N in N_lst])

y,x = np.log(errors), np.log(np.array(N_lst))
slope, _ = np.polyfit(x,y,1)

plt.figure()
plt.plot(x,y)
plt.title("Model 1 - Log of the Mean absolute error")
plt.text(
    0.05, 0.95,
    f"Slope = {slope:.3f}",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)
plt.xlabel("Number of agents (log)")
plt.ylabel("Strong error (log)")
plt.savefig('orderBookStage/figures/mod1_log.jpg')
plt.close()

plt.figure()
plt.plot(np.array(N_lst), errors)
plt.title("Model 1 - Mean absolute error")
plt.xlabel("Number of agents")
plt.ylabel("Strong error")
plt.savefig('orderBookStage/figures/mod1_linear.jpg')
plt.close()
