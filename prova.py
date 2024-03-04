from new_simulation import Simulation
import numpy as np

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


        