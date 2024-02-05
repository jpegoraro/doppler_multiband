from original_sim import Simulation

path= 'plots/fc_60/var_w_1000/2_static/'
sim = Simulation(T=0.1e-3, fo_max=6e3)
sim.simulation(path,relative=True)