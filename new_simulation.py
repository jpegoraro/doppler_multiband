import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
from MeanEstimator import MeanEstimator
from scipy.optimize import least_squares
import tikzplotlib as tik

class Simulation():
    def __init__(self, l=0.005, T=0.25e-3, v_max=5, fo_max=6e3, alpha=1, n_static=2, ambiguity=False):
        """
            Default values for a 60 GHz carrier frequency system, which can measure frequency Doppler shift 
            caused by a motion of at most 5 m/s.
        """
        # simulation parameters
        self.l = l
        self.T = T
        self.v_max = v_max
        self.fd_max = None
        self.fo_max = fo_max 
        self.alpha = alpha
        self.ambiguity = ambiguity
        self.std_w = 0

        # simulation unknowns
        self.eta = 0
        self.f_d = 0
        self.v = 0
        self.f_off = 0

        # simulation inputs
        self.phases = np.zeros((n_static+2,2))
        self.zetas = np.zeros(n_static+1)
        self.n_static = n_static

    def add_noise_phase(self, noise, phase_std):
        """
            noise: wether to add noise to phases and angle of arrivals;
            phase_std: standard deviation for the noise variable regarding the measured phase.
        """
        if noise:
            n_phases = self.phases + np.random.normal(0,phase_std,self.phases.shape)
        else:
            n_phases = self.phases
        return n_phases

    def add_noise_aoa(self, noise, zeta_std):
        """
            noise: wether to add noise to phases and angle of arrivals;
            zeta_std: standard deviation for the noise variable regarding the angle of arrivals;
        """
        if noise:
            n_zetas = self.zetas + np.random.normal(0,zeta_std,self.n_static+1)
        else:
            n_zetas = self.zetas
        return n_zetas  


    def solve_system(self,n_phases, n_zetas):
        """
            n_zetas: noisy angle of arrivals;
            n_phases: noisy measured phases;
            new: if True use the new resolution method.
            Solve the system for the received input parameters and returns the variables of the system.
        """       
        alpha = n_phases[2]*(np.cos(n_zetas[1])-1)-(n_phases[1]*(np.cos(n_zetas[2])-1))
        beta = n_phases[1]*np.sin(n_zetas[2])-(n_phases[2]*np.sin(n_zetas[1]))
        if abs(beta)<1e-5:
            eta = np.pi/2 - np.arctan(beta/alpha)
        else:
            eta = np.arctan(alpha/beta)
        A = (np.sin(n_zetas[1])*(1-np.cos(n_zetas[2]))) + (np.sin(n_zetas[2])*(np.cos(n_zetas[1])-1))
        if not (beta)/A>0:
            eta = eta + np.pi
        if eta<0:
            eta = eta + (2*np.pi)
        f_d = (n_phases[0]-((n_phases[1]*(np.cos(n_zetas[0]-eta)-np.cos(eta)))/(np.cos(n_zetas[1]-eta)-np.cos(eta))))/(2*np.pi*self.T)
        
        v = self.l/(2*np.pi*self.T)*n_phases[1]/(np.cos(n_zetas[1]-eta)-np.cos(eta))
        return eta, f_d, v 
        
    
    def system(self, x, phases, n_zetas):
        """
        x = [f_D, v, eta]
        """
        results = []
        #target path
        results.append(phases[0]-(2*np.pi*self.T*(x[0]+(x[1]/self.l*(np.cos(n_zetas[0]-x[2])-np.cos(x[2]))))))
        # loop for each static path, i.e., excluding LoS and Target
        for i in range(len(phases)-2):
            results.append(phases[i+1]-(2*np.pi*self.T*(x[1]/self.l*(np.cos(n_zetas[i+1]-x[2])-np.cos(x[2])))))
        return np.array(results)
    
  
    
    def check(self, n_phases, n_zetas, eta):
        f_d = np.zeros(8)
        off = [[0,0,0],[2*np.pi,0,0],[2*np.pi,2*np.pi,0],[0,0,2*np.pi],[0,2*np.pi,0],[2*np.pi,2*np.pi,2*np.pi],[2*np.pi,0,2*np.pi],[0,2*np.pi,2*np.pi]]
        for i in range(8):
            f_d[i] = (n_phases[0]+off[i][0]-n_phases[3]+off[i][2]-(((n_phases[2]+off[i][1]-n_phases[3]+off[i][2])*(np.cos(n_zetas[0]-eta)-np.cos(eta)))/(np.cos(n_zetas[2]-eta)-np.cos(eta))))/(2*np.pi*self.T)
        f_d = f_d[f_d<2100]
        return f_d[f_d>-2100]

    
    def compute_phases(self, k:int):
        """
            Solve the system for the received input parameters and returned the computed phases.
        """
        self.phases[:,0] = self.phases[:,1]
        self.phases[0,1] = 2*np.pi*self.T*k*(self.f_d+(self.v/self.l*np.cos(self.zetas[0]-self.eta))+self.f_off)
        for i in range(len(self.zetas)-1):
            self.phases[i+1,1] = 2*np.pi*self.T*k*((self.v/self.l*np.cos(self.zetas[i+1]-self.eta))+self.f_off)
        self.phases[-1,1] = 2*np.pi*self.T*k*((self.v/self.l*np.cos(self.eta))+self.f_off)
        if self.ambiguity:
            self.phases = np.mod(self.phases,2*np.pi)
        # else:
        #     for t in range(2):
        #         for i in range(self.phases.shape[0]):
        #             p =self.phases[i,t]
        #             if p>0:
        #                 p = p%(2*np.pi)
        #             while p<-2*np.pi:
        #                 p = p+(2*np.pi)
        #             if p<-np.pi or p>np.pi:
        #                 print('PHASE AMBIGUITY p=' + str(np.rad2deg(p)) + '°') 
        #                 print('delta CFO='+str(self.f_off))
    
    def compute_input(self, k_ind:int):
        """
        Computes realistic values of: \n measured phases, angle of arrivals, and variables. 
        """
        v_max_sim= self.v_max-2
        self.v = np.random.uniform(0,v_max_sim)
        self.fd_max = 2/self.l*v_max_sim
        self.f_d = np.random.uniform(-self.fd_max,self.fd_max)        
        k = int(1e-3/self.T)
        self.std_w = self.fo_max/(3*np.sqrt(k**3))
        self.f_off = np.random.normal(0,self.std_w)
        self.zetas = np.random.uniform(-np.pi/4,np.pi/4,len(self.zetas))
        self.eta = np.random.uniform(0,2*np.pi)
        self.compute_phases(k=k_ind)

    def update_input(self, k):
        self.f_off = self.f_off*self.alpha+np.random.normal(0,self.std_w) # f_off at time (k)
        self.compute_phases(k=k)

    def boxplot_plot(self, path, errors, xlabel, ylabel, xticks, title, name=''):
        plt.figure(figsize=(12,8))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid()
        ax = sns.boxplot(data=errors, orient='v', palette='rocket', showfliers=False)
        plt.xticks(np.arange(len(xticks)), xticks)
        plt.title(title)
        plt.savefig(path+name+'.png')
        #plt.show()
        tik.save(path+name+'.tex')
        plt.close()

    def check_initial_values(self, x0):
        if x0[0]<-self.fd_max or x0[0]>self.fd_max:
            x0[0]=self.fd_max/2
        if x0[1]<0 or x0[1]>self.v_max:
            x0[1]=2
        x0[2] = x0[2]%(2*np.pi)
        return x0

    def plot_ransac(self,x,y_real,y_noisy,y_pred,mask):
        plt.plot(x,y_pred,label='RANSAC predicitons',c='r', linewidth=2)
        plt.plot(x,y_real,label='Real values',c='g',linewidth=2)
        plt.scatter(x,y_noisy,label='Noisy values')
        y_noisy = y_noisy*mask  
        x = x[y_noisy!=0]
        y_noisy = y_noisy[y_noisy!=0]
        plt.scatter(x,y_noisy,label='Selected subset')
        plt.grid()
        plt.legend(loc='upper left')
        plt.show()
    
    def ambiguity_check(self, diff, phases):
        if diff.any()<-np.pi or diff.any()>np.pi:
            print('AMBIGUITY') 
        a = np.mod(diff,2*np.pi)
        b = ((phases[:-1,1]%(2*np.pi)-(phases[-1,1]%(2*np.pi)))-(phases[:-1,0]%(2*np.pi)-(phases[-1,0]%(2*np.pi))))%(2*np.pi)
        if a.any() != b.any():
            print('phase problem')
        
                
    def simulation(self, path, relative=True, interval=100, N=10000, zeta_std = [5,3,1], phase_std = [10,5,2.5,1], save=False, use_ransac=False, sub_interval=None, noise=True, only_fD=True, plot=True):
        # N is the number of simulations
        # interval is the number of samples in which variables can be considered constant 
        equiv_snr = 10*np.log10(1/(2*256*np.power(np.deg2rad(phase_std),2)))
        phase_std = np.deg2rad(phase_std)
        mean_estimator = MeanEstimator()
        if not only_fD:
            ransac = RANSACRegressor(estimator=mean_estimator, min_samples=10, max_trials=400)
        if interval>20:
            ransac_fd = RANSACRegressor(estimator=mean_estimator, min_samples=10, max_trials=200)
        elif interval!=2:
            ransac_fd = RANSACRegressor(estimator=mean_estimator, min_samples=int(interval/3), max_trials=200)
        if interval>100:
            ransac_fd = RANSACRegressor(estimator=mean_estimator, min_samples=int(interval/4), max_trials=200)
        for z_std in np.deg2rad(zeta_std):
            tot_eta_error = []
            tot_f_d_error = []
            tot_v_error = []
            print('zeta std: ' + str(round(np.rad2deg(z_std))))
            for p_std in phase_std:
                eta_error = []
                f_d_error = []
                v_error = []
                counter = 0
                samples = 0
                for j in tqdm(range(N),dynamic_ncols=True):
                    etas_n = []
                    f_ds_n = []
                    vs_n = []
                    phase_diff = []
                    zetas = []
                    self.compute_input(k_ind=1)
                    for i in range(2,interval):
                        self.update_input(k=i)
                        phases = self.add_noise_phase(noise, p_std)
                        n_phases = np.zeros(phases.shape)
                        n_zetas = self.add_noise_aoa(noise, z_std)
                        zetas.append(n_zetas)
                        ### remove LoS from other paths ###
                        for p in range(phases.shape[0]):
                            for t in range(2):
                                n_phases[p,t] = phases[p,t] - phases[-1,t]
                        ### phase difference ###
                        diff = n_phases[:-1,1] - n_phases[:-1,0]
                        phase_diff.append(diff)
                        self.ambiguity_check(diff,phases)
                        if use_ransac:
                            eta,f_d,v = self.solve_system(diff, n_zetas) 
                            x0 = [f_d, v, eta]
                            x0 = self.check_initial_values(x0)
                            # if len(self.phases)==4:
                            #     results = least_squares(self.system, x0, args=(diff, n_zetas), bounds=([-self.fd_max,0,0],[self.fd_max,self.v_max,2*np.pi]))
                            # else:
                            #     results = least_squares(self.system, x0, args=(diff, n_zetas))
                            f_ds_n.append(f_d)#results.x[0])
                        
                    if use_ransac:
                        samples += len(f_ds_n)
                        time = np.arange(len(f_ds_n)).reshape(-1,1)
                        try:
                            ransac_fd.fit(time,f_ds_n)
                            pred_f_d = ransac_fd.predict(time)
                            #self.plot_ransac(time,np.ones(len(time))*self.f_d,f_ds_n,pred_f_d,ransac_fd.inlier_mask_)
                        except:
                            counter +=1 
                            pred_f_d = np.mean(f_ds_n)
                        if relative:
                            f_d_error.append(np.abs((self.f_d-np.mean(pred_f_d))/self.f_d))
                        else:
                            f_d_error.append(np.abs(self.f_d-np.mean(pred_f_d)))
                    else:
                        if sub_interval:
                            f_ds_n = []
                            n_zetas = np.stack(zetas,axis=0)
                            phase_diff = np.stack(phase_diff, axis=0)
                            for ii in range(0,len(phase_diff),sub_interval):
                                p_diff = np.mean(phase_diff[ii:ii+sub_interval,:], axis=0)
                                n_zeta = np.mean(n_zetas[ii:ii+sub_interval,:], axis=0)
                                eta, f_d, v = self.solve_system(p_diff,n_zeta)
                                x0 = [f_d, v, eta]
                                x0 = self.check_initial_values(x0)
                                if len(self.phases)==4:
                                    results = least_squares(self.system, x0, args=(p_diff, n_zeta), bounds=([-self.fd_max,0,0],[self.fd_max,self.v_max,2*np.pi]))
                                else:
                                    results = least_squares(self.system, x0, args=(p_diff, n_zeta))
                                f_ds_n.append(results.x[0])
                            time = np.arange(len(f_ds_n)).reshape(-1,1)
                            try:
                                ransac_fd.fit(time,f_ds_n)
                                pred_f_d = ransac_fd.predict(time)
                                #self.plot_ransac(time,np.ones(len(time))*self.f_d,f_ds_n,pred_f_d,ransac_fd.inlier_mask_)
                            except:
                                counter +=1 
                                pred_f_d = np.mean(f_ds_n)
                            if relative:
                                f_d_error.append(np.abs((self.f_d-np.mean(pred_f_d))/self.f_d))
                            else:
                                f_d_error.append(np.abs(self.f_d-np.mean(pred_f_d)))
                        else:
                            phase_diff = np.stack(phase_diff, axis=0)
                            #plt.plot(phase_diff[:,0])
                            phase_diff = np.mean(phase_diff, axis=0)
                            n_zetas = np.mean(np.stack(zetas,axis=0),axis=0)
                            # plt.plot(np.ones(500)*phase_diff[0])
                            # plt.ylim(-np.pi,np.pi)
                            # plt.show()
                            eta, f_d, v = self.solve_system(phase_diff,n_zetas)
                            # if relative:
                            #     f_d_error.append(np.abs((self.f_d-np.mean(f_d))/self.f_d))
                            #x0 = [self.fd_max/4,2,np.pi/3] # [f_d(0), v(0), eta(0)]
                            x0 = [f_d, v, eta]
                            x0 = self.check_initial_values(x0)
                            if len(self.phases)==4:
                                results = least_squares(self.system, x0, args=(phase_diff, n_zetas), bounds=([-self.fd_max,0,0],[self.fd_max,self.v_max,2*np.pi]))
                            else:
                                results = least_squares(self.system, x0, args=(phase_diff, n_zetas))
                                etas_n.append(results.x[2])
                                #if results.x[0]<(self.fd_max*2) and results.x[0]>-(self.fd_max*2): 
                            if relative:
                                f_d_error.append(np.abs((self.f_d-np.mean(results.x[0]))/self.f_d))
                            else:
                                f_d_error.append(np.abs(self.f_d-np.mean(results.x[0])))
                            vs_n.append(results.x[1])
                    
                print(str(counter) + ' ransac fD exceptions')
                print('average number of considered samples: ' + str(samples/N))    
                if not only_fD:
                    tot_eta_error.append(eta_error)
                    tot_v_error.append(v_error)
                tot_f_d_error.append(f_d_error)
                if save:
                    np.save(path+'fd_k'+str(interval)+'_ns'+str(self.n_static)+'_pstd'+str(np.rad2deg(p_std))+'_zstd'+str(np.rad2deg(z_std))+'.npy',f_d_error)
            if plot:
                if relative:    
                    self.boxplot_plot(path, tot_f_d_error, "phase std [°]", "relative frequency Doppler errors", np.rad2deg(phase_std), "frequency Doppler errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'fd_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                    if not only_fD:
                        self.boxplot_plot(path, tot_eta_error, "phase std [°]", "relative eta errors", np.rad2deg(phase_std), "eta errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'eta_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                        self.boxplot_plot(path, tot_v_error, "phase std [°]", "relative speed error", np.rad2deg(phase_std), "speed errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'speed_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                else:
                    self.boxplot_plot(path, tot_f_d_error, "phase std [°]", "frequency Doppler errors (Hz)", np.rad2deg(phase_std), "frequency Doppler errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'fd_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                    if not only_fD:
                        self.boxplot_plot(path, tot_v_error, "phase std [°]", "speed error (m/s)", np.rad2deg(phase_std), "speed errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'speed_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                        self.boxplot_plot(path, tot_eta_error, "phase std [°]", "eta errors (°)", np.rad2deg(phase_std), "eta errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'eta_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
            else:
                if only_fD:
                    return tot_f_d_error
                else:
                    return tot_f_d_error,tot_eta_error,tot_v_error
    
    
if __name__=='__main__':

    ### fc = 60 GHz ###
    sim = Simulation(T=0.08e-3, fo_max=6e3, n_static=2)
    path='plots/new_sim/2_static/p_std/'
    #path='plots/test/'
    sim.simulation(path, relative=True, noise=True, N=10000, interval=200, save=True)
    
    ### fc = 28 GHz ### 
    # sim = Simulation(l=0.0107, T=t*1e-3, v_max=10, fo_max=2.8e3, n_static=4, ambiguity=True)
    # path='plots/varying_T/fc_28/'
    # sim.simulation(path, relative=True, zeta_std=[5], SNR=[10], interval=50)

    ### fc = 5 GHz ###
    # sim = Simulation(l=0.06, T=0.09e-3, v_max=10, fo_max=0.5e3)
    # path='plots/k_2/fc_5/'
    # sim.simulation(path, relative=True, N=100000, interval=2)



    