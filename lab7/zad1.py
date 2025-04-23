import pyswarms as ps
import numpy as np
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

def endurance(x):
  return np.exp(-2*(x[1]-np.sin(x[0]))**2) + np.sin(x[2]*x[3]) + np.cos(x[4]*x[5])

def f(x):
  return np.array([-endurance(x[i]) for i in range(x.shape[0])])

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
x_max = np.ones(6) #[2, 2]
x_min = np.zeros(6) #[1, 1]
my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, 
                               options=options, bounds=my_bounds)

optimizer.optimize(f, iters=1000)
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.savefig('cost_history(endurance).png')