from simulation import Simulation
import numpy as np

#varying T

for f in [60]:
    print('carrier frequency: ' + str(f) + ' GHz')
    tot_f_d_error =[]
    for ns in [2,4,6]:
        print('ns= ' + str(ns))
        #path = 'plots/new_sim/varying_T/new/'
        path = 'plots/new_sim/T/'
        f_d_error = []
        times = np.arange(0.08,0.52,0.02)
        for t in times:
            t  = round(t,2)
            print('period= ' + str(t) + ' ms')
            if f==60:
                ### fc = 60 GHz ###
                sim = Simulation(T=t*1e-3, fo_max=60e3, n_static=ns, fd_min=None)
            if f==28:
                ### fc = 28 GHz ### 
                sim = Simulation(l=0.0107, T=t*1e-3, v_max=10, fo_max=28e3, n_static=ns, fd_min=None)
            error = sim.simulation(path+'fc_'+str(f)+'/'+str(ns)+'_static/',relative=True,zeta_std=[5],phase_std=[5],save=False,plot=False,N=10000,interval=200)
            f_d_error.append(error)
            print('average fD relative error: ' + str(np.mean(error)) + ' std: ' + str(np.std(error)))
        #p = path+'fc_'+str(f)+'/varying_t'
        tot_f_d_error.append(np.squeeze(np.stack(f_d_error)))
    np.save(path+'fc_'+str(f)+'_'+str(ns)+'_static_f_d_error.npy',tot_f_d_error)
    legend = ['2 static paths', '4 static paths', '6 static paths']
    sim.plot_mae(tot_f_d_error,path,legend,times)


# path= 'plots/new_sim/2_static/varying_k/'
# #path='plots/test/'
# for f in[28,60]:
#     print('carrier frequency: ' + str(f) + ' GHz')
#     tot_f_d_error = []
#     for inter in [25,50,100,200,400,600]:
#         print('selected interval of ' + str(inter) + ' packets')
#         if f==60:
#             ### fc = 60 GHz ###
#             sim = Simulation(T=0.08e-3, fo_max=60e3)
#         if f==28:
#             ### fc = 28 GHz ### 
#             sim = Simulation(l=0.0107, T=0.08e-3, v_max=10, fo_max=28e3)
#         if f==5:
#             ### fc = 5 GHz ###
#             sim = Simulation(l=0.06, T=0.09e-3, v_max=10, fo_max=0.5e3)
#         tot_f_d_error.append(sim.simulation(path+'fc_'+str(f),relative=True,zeta_std=[5],phase_std=[5],save=True,plot=False,N=10000,interval=inter))
#     sim.boxplot_plot(path+'fc_'+str(f)+'/', tot_f_d_error,'interval [ms]','fD error', [2,4,8,16,32,48], 'varying k', 'varyingk_'+str(f)+'GHz')

# for inter in [100,200]:
#     path = 'plots/new_sim/varying_path/'
#     path = path + 'interval_'+str(inter)+'/'
#     for f in [28,60]:
#         print('carrier frequency: ' + str(f) + ' GHz')
#         tot_f_d_error = []
#         for ns in [2,4,6,8,10]:
#             print('ns= ' + str(ns))
#             if f==60:
#                 ### fc = 60 GHz ###
#                 sim = Simulation(T=0.08e-3, fo_max=60e3, n_static=ns)
#             if f==28:
#                 ### fc = 28 GHz ### 
#                 sim = Simulation(l=0.0107, T=0.08e-3, v_max=10, fo_max=28e3, n_static=ns)
#             tot_f_d_error.append(sim.simulation(path+'fc_'+str(f)+'/',relative=True,zeta_std=[5],phase_std=[5],save=True,plot=False,N=10000,interval=inter))
#         sim.boxplot_plot(path+'fc_'+str(f)+'/', tot_f_d_error,'M','fD error', [4,6,8,10,12], 'varying ns', 'varyingns_'+str(f)+'GHz')

# p = 'plots/new_sim/'
# for inter in [200]:
#     for f in [60,28]:
#         print('carrier frequency: ' + str(f) + ' GHz')
#         tot_f_d_error = []
#         for i,ns in enumerate([2,4,6]):
#             path = p + str(ns) + '_static/'
#             print('ns= ' + str(ns))
#             if f==60:
#                 ## fc = 60 GHz ###
#                 sim = Simulation(T=0.08e-3, fo_max=60e3, n_static=ns)
#             if f==28:
#                 ## fc = 28 GHz ### 
#                 sim = Simulation(l=0.0107, T=0.08e-3, v_max=10, fo_max=28e3, n_static=ns)
#             tot_f_d_error.append(sim.simulation(path+'fc_'+str(f)+'/interval_'+str(inter)+'/',relative=True,zeta_std=[5],phase_std=[10,5,2.5,1],save=True,plot=True,N=10000,interval=inter))

## fc = 60 GHz ###
sim = Simulation(T=0.08e-3, fo_max=60e3, n_static=2)
path='plots/new_sim/2_static/p_std/'
#path='plots/test/'
sim.simulation(path, relative=True, noise=True, N=10000, zeta_std=[1], interval=200, save=True)