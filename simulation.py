import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
from MeanEstimator import MeanEstimator
from scipy.optimize import least_squares
import tikzplotlib as tik

class Simulation():
    def __init__(self, l=0.005, T=0.08e-3, v_max=5, fo_max=6e3, alpha=1, n_static=2, fd_min=None):
        """
            Default values for a 60 GHz carrier frequency system, which can measure frequency Doppler shift 
            caused by a motion of at most 5 m/s.

            l: wavelength;
            T: frame period;
            v_max: max receiver speed;
            fo_max: max fo shift in 1 ms;
            alpha: fo random walk parameter;
            n_static: number of static paths (total paths = n_static+2);
            fd_min: minimum target Doppler frequency. 
        """
        # simulation parameters
        self.l = l
        self.T = T
        self.v_max = v_max
        self.fd_max = None
        self.fo_max = fo_max 
        self.alpha = alpha
        self.std_w = 0
        self.fd_min = fd_min
        self.f_off = 0

        # simulation unknowns
        self.eta = 0
        self.f_d = 0
        self.v = 0

        # simulation inputs
        self.phases = np.zeros((n_static+2,2))
        self.zetas = np.zeros(n_static+1)
        self.n_static = n_static

    def add_noise_aoa(self, noise, zeta_std):
        """
            Add noise to the angle of arrivals.

            noise: wether to add noise to phases and angle of arrivals;
            zeta_std: standard deviation for the noise variable regarding the angle of arrivals;
        """
        if noise:
            n_zetas = self.zetas + np.random.normal(0,zeta_std,self.n_static+1)
        else:
            n_zetas = self.zetas
        return n_zetas

    def add_noise_phase(self, noise, phase_std):
        """
            Add noise to the measured phases.    
        
            noise: wether to add noise to phases and angle of arrivals;
            phase_std: standard deviation for the noise variable regarding the measured phase.
        """
        if noise:
            n_phases = self.phases + np.random.normal(0,phase_std,self.phases.shape)
        else:
            n_phases = self.phases
        return n_phases

    def check_initial_values(self, x0):
        """
            Check if the selected initial values are within the known intervals and change them to default value if not.

            x0 = [f_D(0), v(0), eta(0)]: initial values.
        """
        if x0[0]<-self.fd_max or x0[0]>self.fd_max:
            x0[0]=self.fd_max/2
        if x0[1]<0 or x0[1]>self.v_max:
            x0[1]=2
        x0[2] = x0[2]%(2*np.pi)
        return x0  

    def compute_input(self, k:int):
        """
            Computes realistic values of: \n measured phases, angle of arrivals, and variables. 

            k: time index.
        """
        v_max_sim = self.v_max
        self.v = np.random.uniform(0,v_max_sim)
        self.fd_max = 2/self.l*v_max_sim
        if self.fd_min==None: 
            self.fd_min = 0.2/self.l # below this frequency the target is assumed stationary
        if np.random.uniform()<0.5:
            self.f_d = np.random.uniform(-self.fd_max,-self.fd_min)
        else:
            self.f_d = np.random.uniform(self.fd_min,self.fd_max)
        n = int(1e-3/self.T) # number of samples in 1 ms
        self.std_w = self.fo_max/(3*np.sqrt(n**3)) # std for the fo random walk s.t. its max drift in 1 ms is fo_max
        self.f_off = np.random.normal(0,self.std_w)
        self.zetas = np.random.uniform(-np.pi/4,np.pi/4,len(self.zetas)) # antennas f.o.v. is between -45° and 45°
        self.eta = np.random.uniform(0,2*np.pi)
        self.compute_phases(k=k)

    def compute_phases(self, k:int):
        """
            Solve the system for the received input parameters and returned the computed phases.

            k: time index.
        """
        self.phases[:,0] = self.phases[:,1]
        self.phases[0,1] = 2*np.pi*self.T*k*(self.f_d+(self.v/self.l*np.cos(self.zetas[0]-self.eta))+self.f_off)
        for i in range(len(self.zetas)-1):
            self.phases[i+1,1] = 2*np.pi*self.T*k*((self.v/self.l*np.cos(self.zetas[i+1]-self.eta))+self.f_off)
        self.phases[-1,1] = 2*np.pi*self.T*k*((self.v/self.l*np.cos(self.eta))+self.f_off)

    def my_mod_2pi(self, phases):
        """
            Map an array or a matrix of angles [rad] in [-pi,pi].

            phases: measured phases.
        """
        if phases.ndim>1:
            for j in range(phases.ndim):
                for i,p in enumerate(phases[:,j]):
                    while p <-np.pi:
                        p = p+2*np.pi
                    while p>np.pi:
                        p = p-2*np.pi
                    phases[i,j]=p
        else:
            for i,p in enumerate(phases):
                    while p <-np.pi:
                        p = p+2*np.pi
                    while p>np.pi:
                        p = p-2*np.pi
                    phases[i]=p
        return phases

    def solve_system(self, phases, zetas):
        """
            Solve the system for the received input parameters and returns the unknowns of the system.

            phases: measured phases;
            zetas: angle of arrivals.
        """       
        alpha = phases[2]*(np.cos(zetas[1])-1)-(phases[1]*(np.cos(zetas[2])-1))
        beta = phases[1]*np.sin(zetas[2])-(phases[2]*np.sin(zetas[1]))
        if abs(beta)<1e-5:
            eta = np.pi/2 - np.arctan(beta/alpha)
        else:
            eta = np.arctan(alpha/beta)
        A = (np.sin(zetas[1])*(1-np.cos(zetas[2]))) + (np.sin(zetas[2])*(np.cos(zetas[1])-1))
        if not (beta)/A>0:
            eta = eta + np.pi
        if eta<0:
            eta = eta + (2*np.pi)
        f_d = (phases[0]-((phases[1]*(np.cos(zetas[0]-eta)-np.cos(eta)))/(np.cos(zetas[1]-eta)-np.cos(eta))))/(2*np.pi*self.T)
        
        v = self.l/(2*np.pi*self.T)*phases[1]/(np.cos(zetas[1]-eta)-np.cos(eta))
        return eta, f_d, v 
    
    def system(self, x, phases, n_zetas):
        """
            Define the system to solve it using the non linear least-square method, for at least 4 measured phases (i.e. 2 static paths).
            Returns an array representing the system.

            x = [f_D, v, eta]: system unknowns;
            phases: measured phases;
            zetas: angle of arrivals.
        """
        results = []
        #target path
        results.append(phases[0]-(2*np.pi*self.T*(x[0]+(x[1]/self.l*(np.cos(n_zetas[0]-x[2])-np.cos(x[2]))))))
        # loop for each static path, i.e., excluding LoS and Target
        for i in range(len(phases)-2):
            results.append(phases[i+1]-(2*np.pi*self.T*(x[1]/self.l*(np.cos(n_zetas[i+1]-x[2])-np.cos(x[2])))))
        return np.array(results)

    def plot_boxplot(self, path, errors, xlabel, ylabel, xticks, title, name=''):
        """
            Plot boxplots using seaborn for the given list of errors.
        """
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

    def plot_mae(self, dataa, path, legend, times):
        """
            Plot mean and std of data.
        """
        #print(data.shape)
        m = []
        s = []
        for data in dataa:
            mean = np.mean(data,1)
            m.append(mean)
            s.append(np.std(data,1))
            plt.plot(times, mean)
        plt.legend(legend)
        for mean,std in zip(m,s):    
            low = mean-(std)
            low[low<0] = 0
            plt.fill_between(times, low, mean+std, alpha=0.3)
        plt.grid()
        plt.xlabel('time (ms)')
        plt.ylabel('fD mean relative error')
        tik.save(path + '.tex')
        plt.savefig(path + '.png')
        #plt.show()

    def plot_ransac(self,x,y_real,y_noisy,y_pred,mask):
        """
            Plot ransac prediction and selected subset.
        """
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
                
    def simulation(self, path, relative=True, interval=100, N=10000, zeta_std = [5,3,1], phase_std = [10,5,2.5,1], save=False, use_ransac=False, sub_interval=None, noise=True, only_fD=True, plot=True):
        """
            Perform the simulation.

            path: where outputs are saved;
            relative: wether to consider relative or absolute errors;
            interval: number of frames during which the parameters are considered constant;
            N: number of simulations;
            zeta_std: std of the noise added to the angle of arrivals (one set of simulaations for each std);
            phase_std: std of the noise added to the measured phases (one set of simulaations for each std);
            save: wether to save the estimation errors;
            use_ransac: wether to use ransac;
            sub_interval: length of a sub interval<interval, in which to use ransac (if None sub intervals are not used);
            noise: wether to add noise to angle of arrivals and measured phases;
            only_fD: wether to track only fD estimation errors (i.e. if True v and eta estimation errors are not tracked);
            plot: wether to save the estimation errors boxplots.
        """
        #equiv_snr = 10*np.log10(1/(2*256*np.power(np.deg2rad(phase_std),2)))
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
                        mod_phases = self.my_mod_2pi(phases)
                        n_phases = np.zeros(phases.shape)
                        n_zetas = self.add_noise_aoa(noise, z_std)
                        zetas.append(n_zetas)
                        ### remove LoS from other paths ###
                        for p in range(phases.shape[0]):
                            for t in range(2):
                                n_phases[p,t] = mod_phases[p,t] - mod_phases[-1,t]
                        ### phase difference ###
                        diff = n_phases[:-1,1] - n_phases[:-1,0]
                        diff = self.my_mod_2pi(diff)
                        phase_diff.append(diff)
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
                            phase_diff = np.mean(phase_diff, axis=0)
                            n_zetas = np.mean(np.stack(zetas,axis=0),axis=0)
                            eta, f_d, v = self.solve_system(phase_diff,n_zetas)
                            x0 = [f_d, v, eta]
                            x0 = self.check_initial_values(x0)
                            if len(self.phases)==4 or len(self.phases)==6:
                                results = least_squares(self.system, x0, args=(phase_diff, n_zetas), bounds=([-self.fd_max,0,0],[self.fd_max,self.v_max,2*np.pi]))
                            else:
                                results = least_squares(self.system, x0, args=(phase_diff, n_zetas))
                                #if results.x[0]<(self.fd_max*2) and results.x[0]>-(self.fd_max*2): 
                            if relative:
                                err = np.abs((self.f_d-np.mean(results.x[0]))/self.f_d)
                                if err>250:
                                    print('error: '+str(err))
                                    print('real fD: '+str(self.f_d))
                                    print('est. fD: '+str(np.mean(results.x[0])))
                                f_d_error.append(err)
                            else:
                                f_d_error.append(np.abs(self.f_d-np.mean(results.x[0])))
                                etas_n.append(results.x[2])
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
                    self.plot_boxplot(path, tot_f_d_error, "phase std [°]", "relative frequency Doppler errors", np.rad2deg(phase_std), "frequency Doppler errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'fd_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                    if not only_fD:
                        self.plot_boxplot(path, tot_eta_error, "phase std [°]", "relative eta errors", np.rad2deg(phase_std), "eta errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'eta_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                        self.plot_boxplot(path, tot_v_error, "phase std [°]", "relative speed error", np.rad2deg(phase_std), "speed errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'speed_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                else:
                    self.plot_boxplot(path, tot_f_d_error, "phase std [°]", "frequency Doppler errors (Hz)", np.rad2deg(phase_std), "frequency Doppler errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'fd_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                    if not only_fD:
                        self.plot_boxplot(path, tot_v_error, "phase std [°]", "speed error (m/s)", np.rad2deg(phase_std), "speed errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'speed_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                        self.plot_boxplot(path, tot_eta_error, "phase std [°]", "eta errors (°)", np.rad2deg(phase_std), "eta errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'eta_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
            if only_fD:
                return tot_f_d_error
            else:
                return tot_f_d_error,tot_eta_error,tot_v_error

    def update_input(self, k:int):
        """
            Update the frequency offset at time k>0 and consequently the measured phases.

            k: time index. 
        """
        self.f_off = self.f_off*self.alpha+np.random.normal(0,self.std_w) # f_off at time (k)
        self.compute_phases(k=k)
    
if __name__=='__main__':

    ### fc = 60 GHz ###
    sim = Simulation(T=0.08e-3, fo_max=60e3, n_static=2)
    #path='plots/new_sim/2_static/p_std/'
    path='plots/test/'
    err = sim.simulation(path, relative=True, noise=True,zeta_std=[5], phase_std=[5], N=10000, interval=200, plot=False, save=False)
    print(np.mean(err))

    ### fc = 28 GHz ### 
    # sim = Simulation(l=0.0107, T=t*1e-3, v_max=10, fo_max=2.8e3, n_static=4)
    # path='plots/varying_T/fc_28/'
    # sim.simulation(path, relative=True, zeta_std=[5], SNR=[10], interval=50)

    ### fc = 5 GHz ###
    # sim = Simulation(l=0.06, T=0.09e-3, v_max=10, fo_max=0.5e3)
    # path='plots/k_2/fc_5/'
    # sim.simulation(path, relative=True, N=100000, interval=2)



    