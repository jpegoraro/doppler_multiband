from original_sim import Simulation

path= 'plots/varying_k/'
for f in[60,28]:
    print('carrier frequency: ' + str(f) + ' GHz')
    tot_f_d_error = []
    for inter in [2,10,20,30,40,50]:
        print('selcted interval of ' + str(inter) + ' packets')
        if f==60:
            ### fc = 60 GHz ###
            sim = Simulation(T=0.05e-3, fo_max=6e3)
        if f==28:
            ### fc = 28 GHz ### 
            sim = Simulation(l=0.0107, T=0.075e-3, v_max=10, fo_max=2.8e3)
        if f==5:
            ### fc = 5 GHz ###
            sim = Simulation(l=0.06, T=0.09e-3, v_max=10, fo_max=0.5e3)
        tot_f_d_error.append(sim.simulation(path,relative=True,zeta_std=[5],SNR=[10],plot=False,interval=inter))
    sim.boxplot_plot(path, tot_f_d_error,'number of static paths','fD error', [2,4,6,8,10], 'varying ns', 'varyingns_'+str(f)+'GHz')

path= 'plots/k_2/'
for f in [60,28]:
    print('carrier frequency: ' + str(f) + ' GHz')
    tot_f_d_error = []
    for ns in [2,4,6,8,10]:
        print('ns= ' + str(ns))
        if f==60:
            ### fc = 60 GHz ###
            sim = Simulation(T=0.05e-3, fo_max=6e3, n_static=ns)
        if f==28:
            ### fc = 28 GHz ### 
            sim = Simulation(l=0.0107, T=0.075e-3, v_max=10, fo_max=2.8e3, n_static=ns)
        tot_f_d_error.append(sim.simulation(path,relative=True,zeta_std=[5],SNR=[10],plot=False,interval=2))
    sim.boxplot_plot(path+'fc_'+str(f)+'/', tot_f_d_error,'number of static paths','fD error', [2,4,6,8,10], 'varying ns', 'varyingns_'+str(f)+'GHz')

