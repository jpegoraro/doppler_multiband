from original_sim import Simulation

path= 'plots/fc_5/var_w_1000/2_static/'
sim = Simulation(l=0.06, T=0.8e-3, v_max=10, fd_max=350, fo_max=500)
sim.simulation(path,relative=True, N=10000)