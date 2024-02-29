from new_simulation import Simulation
import numpy as np

#varying T

for f in [28,60]:
    print('carrier frequency: ' + str(f) + ' GHz')
    tot_f_d_error =[]
    for ns in [2,4,6]:
        print('ns= ' + str(ns))
        path = 'plots/new_sim/varying_T/'
        f_d_error = []
        times = np.arange(0.08,0.52,0.02)
        for t in times:
            t  = round(t,2)
            print('period= ' + str(t))
            if f==60:
                ### fc = 60 GHz ###
                sim = Simulation(T=t*1e-3, fo_max=6e3, n_static=ns)
            if f==28:
                ### fc = 28 GHz ### 
                sim = Simulation(l=0.0107, T=t*1e-3, v_max=10, fo_max=2.8e3, n_static=ns)
            f_d_error.append(sim.simulation(path+'fc_'+str(f)+'/'+str(ns)+'_static/',relative=True,zeta_std=[5],phase_std=[5],save=False,plot=False,N=10000,interval=200))
        p = path+'fc_'+str(f)+'/varying_t'
        tot_f_d_error.append(np.squeeze(np.stack(f_d_error)))
        np.save(path+'fc_'+str(f)+'/'+str(ns)+'_static/f_d_error.npy',tot_f_d_error)
    legend = ['2 static paths', '4 static paths', '6 static paths']
    sim.plot_mae(tot_f_d_error,p,legend,times)