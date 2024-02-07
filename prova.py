from original_sim import Simulation

path= 'plots/varying_fc/'
# sim = Simulation(T=0.1e-3, fo_max=6e3)
# sim.simulation(path,relative=True)
for f in [ 5]:
    print('carrier frequency: ' + str(f) + ' GHz')
    tot_f_d_error = []
    for ns in [2,4,6,8,10]:
        print('ns= ' + str(ns))
        if f==60:
            sim = Simulation(T=0.1e-3, fo_max=6e3, n_static=ns)
        if f==28:
            sim = Simulation(l=0.0107, T=0.15e-3, v_max=10, fd_max=1900, fo_max=2.8e3, n_static=ns)
        if f==5:
            sim = Simulation(l=0.06, T=0.8e-3, v_max=10, fd_max=350, fo_max=500, n_static=ns)
        tot_f_d_error.append(sim.simulation(path,relative=True,zeta_std=[3],SNR=[10],plot=False))
    sim.boxplot_plot(path, tot_f_d_error,'number of static paths','fD error', [2,4,6,8,10], 'varying fc', 'varyingfc_'+str(f)+'GHz')