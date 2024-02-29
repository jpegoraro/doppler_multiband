from new_simulation import Simulation
import numpy as np

# sim = Simulation(l=0.0107, T=1e-3, v_max=10, fo_max=2.8e3, n_static=2)
# data = np.load('plots/new_sim/varying_T/fc_28/4_static/f_d_error.npy')
# sim.plot_mae(data,'plots/test/test',['2 static'],np.arange(0.08,0.52,0.02))

times = np.arange(0.08,0.52,0.02)
for t in times:
    t = round(t,2)
    print('period= ' + str(t))