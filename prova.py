from original_sim import Simulation

path= 'plots/varying_ns/ambiguity1/'
for f in [5]:
    print('carrier frequency: ' + str(f) + ' GHz')
    tot_f_d_error = []
    for ns in [2,4,6,8,10]:
        print('ns= ' + str(ns))
        if f==60:
            sim = Simulation(fo_max=60e3, n_static=ns, ambiguity=True)
        if f==28:
            sim = Simulation(l=0.0107, T=0.26e-3, v_max=10, fd_max=1900, fo_max=28e3, n_static=ns, ambiguity=True)
        if f==5:
            sim = Simulation(l=0.06, T=1.4e-3, v_max=10, fd_max=350, fo_max=5e3, n_static=ns, ambiguity=True)
        tot_f_d_error.append(sim.simulation(path,relative=True,zeta_std=[3],SNR=[10],plot=False))
    sim.boxplot_plot(path, tot_f_d_error,'number of static paths','fD error', [2,4,6,8,10], 'varying ns', 'varyingns_'+str(f)+'GHz')