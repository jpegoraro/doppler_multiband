from new_simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt

sim = Simulation(l=0.0107, T=1e-3, v_max=10, fo_max=2.8e3, n_static=2)
data = []
data.append(np.load('plots/new_sim/varying_T/new/fc_28/2_static/f_d_error.npy')[0,:,:])
data.append(np.load('plots/new_sim/varying_T/new/fc_28/4_static/f_d_error.npy')[0,:,:])
data.append(np.load('plots/new_sim/varying_T/new/fc_28/6_static/f_d_error.npy')[2,:,:])

data = np.stack(data)
# d = data[0,1,:]
# plt.plot(np.arange(10000),d)
# plt.show()
sim.plot_mae(data,'plots/test/test',['2 static','4 static','6 static'],np.arange(0.08,0.52,0.02))

