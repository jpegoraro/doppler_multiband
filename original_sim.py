import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
from MeanEstimator import MeanEstimator
from scipy.optimize import least_squares
import tikzplotlib as tik

class Simulation():
    def __init__(self, l=0.005, T=0.25e-3, v_max=5, fo_mean=1e5, fd_max=2000, alpha=0.99, var_w=1e3, n_static=2):
        """
            Default values for a 60 GHz carrier frequency system, which can measure frequency Doppler shift 
            caused by a motion of at most 5 m/s.
        """
        # simulation parameters
        self.l = l
        self.T = T
        self.v_max = v_max
        self.fo_mean = fo_mean 
        self.fd_max = fd_max
        self.alpha = alpha
        self.std_w = np.sqrt(var_w)

        # simulation unknowns
        self.eta = 0
        self.f_d = 0
        self.v = 0
        self.f_off = np.zeros(2)

        # simulation inputs
        self.phases = np.zeros(n_static+2)
        self.zetas = np.zeros(n_static+1)
        self.n_static = n_static

    def add_noise(self, noise, zeta_std, phase_std):
        """
            noise: wether to add noise to phases and angle of arrivals;
            zeta_std: standard deviation for the noise variable regarding the angle of arrivals;
            phase_std: standard deviation for the noise variable regarding the measured phase.
        """
        if noise:
            n_zetas = self.zetas + np.random.normal(0,zeta_std,self.n_static+1)
            n_phases = self.phases + np.random.normal(0,phase_std,self.n_static+2)
        else:
            n_zetas = self.zetas
            n_phases = self.phases
        return n_phases, n_zetas  


    def solve_system(self,n_phases, n_zetas, new=True,):
        """
            n_zetas: noisy angle of arrivals;
            n_phases: noisy measured phases;
            new: if True use the new resolution method.
            Solve the system for the received input parameters and returns the variables of the system.
        """       
        alpha = (n_phases[1]-n_phases[3])*np.cos(n_zetas[2])
        beta = (n_phases[1]-n_phases[3])*np.sin(n_zetas[2])
        delta = n_phases[2]-n_phases[1]
        gamma = (n_phases[3]-n_phases[2])*np.cos(n_zetas[1])
        epsilon = (n_phases[3]-n_phases[2])*np.sin(n_zetas[1])
        if new:
            if abs(beta+epsilon)<1e-5:
                eta = np.pi/2 - np.arctan(-(beta+epsilon)/(alpha+delta+gamma))
            else:
                eta = np.arctan(-(alpha+delta+gamma)/(beta+epsilon))
            A = (np.sin(n_zetas[1])*(1-np.cos(n_zetas[2]))) + (np.sin(n_zetas[2])*(np.cos(n_zetas[1])-1))
            if not (beta+epsilon)/A>0:
                eta = eta + np.pi
            if eta<0:
                eta = eta + (2*np.pi)
            f_d = (n_phases[0]-n_phases[3]-(((n_phases[2]-n_phases[3])*(np.cos(n_zetas[0]-eta)-np.cos(eta)))/(np.cos(n_zetas[2]-eta)-np.cos(eta))))/(2*np.pi*self.T)
            check = self.check(n_phases,n_zetas,eta)
            if len(check)!=0:
                f_d=check[0]
            v = self.l/(2*np.pi*self.T)*(n_phases[2]-n_phases[3])/(np.cos(n_zetas[2]-eta)-np.cos(eta))
            f_off = n_phases[3]/(2*np.pi*self.T)-((n_phases[2]-n_phases[3])*np.cos(eta)/(np.cos(n_zetas[2]-eta)-np.cos(eta)))
            return eta, f_d, v #, f_off, alpha, delta, gamma
        else:
            t = np.roots([-(alpha+delta+gamma),2*(beta+epsilon),(alpha+delta+gamma)])
            f_d = np.zeros(2)
            v = np.zeros(2)
            eta = 2*np.arctan(t)
            f_d[0] = (n_phases[0]-n_phases[3]-(((n_phases[2]-n_phases[3])*(np.cos(n_zetas[0]-eta[0])-np.cos(eta[0])))/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0]))))/(2*np.pi*self.T)
            f_d[1] = (n_phases[0]-n_phases[3]-(((n_phases[2]-n_phases[3])*(np.cos(n_zetas[0]-eta[1])-np.cos(eta[1])))/(np.cos(n_zetas[2]-eta[1])-np.cos(eta[1]))))/(2*np.pi*self.T)
            v[0] = self.l/(2*np.pi*self.T)*(n_phases[2]-n_phases[3])/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0]))
            v[1] = self.l*(n_phases[2]-n_phases[3])/((np.cos(n_zetas[2]-eta[1])-np.cos(eta[1]))*2*np.pi*self.T)
            f_off = n_phases[3]/(2*np.pi*self.T)-((n_phases[2]-n_phases[3])*np.cos(eta[0])/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0])))
            return eta[1], f_d[0], v[1] #, f_off, alpha, delta, gamma, n_zetas
    
    def system(self, x, phases, n_zetas):
        """
        x = [f_D, v, eta, off]
        """
        results = []
        #target path
        results.append(phases[0]-(2*np.pi*self.T*(x[0]+(x[1]/self.l*np.cos(n_zetas[0]-x[2]))+x[3])))
        # loop for each static path, i.e., excluding LoS and Target
        for i in range(len(phases)-2):
            results.append(phases[i+1]-(2*np.pi*self.T*(x[1]/self.l*np.cos(n_zetas[i+1]-x[2])+x[3])))
        #LoS path
        results.append(phases[-1]-(2*np.pi*self.T*(x[1]/self.l*np.cos(x[2])+x[3])))
        return np.array(results)
    
    def check(self, n_phases, n_zetas, eta):
        f_d = np.zeros(8)
        off = [[0,0,0],[2*np.pi,0,0],[2*np.pi,2*np.pi,0],[0,0,2*np.pi],[0,2*np.pi,0],[2*np.pi,2*np.pi,2*np.pi],[2*np.pi,0,2*np.pi],[0,2*np.pi,2*np.pi]]
        for i in range(8):
            f_d[i] = (n_phases[0]+off[i][0]-n_phases[3]+off[i][2]-(((n_phases[2]+off[i][1]-n_phases[3]+off[i][2])*(np.cos(n_zetas[0]-eta)-np.cos(eta)))/(np.cos(n_zetas[2]-eta)-np.cos(eta))))/(2*np.pi*self.T)
        f_d = f_d[f_d<2100]
        return f_d[f_d>-2100]

    
    def compute_phases(self, k):
        """
            Solve the system for the received input parameters and returned the computed phases.
        """
        self.phases[0] = (2*np.pi*self.T*(self.f_d+(self.v/self.l*np.cos(self.zetas[0]-self.eta))+(k*self.f_off[1]-(k-1)*self.f_off[0])))%(2*np.pi)
        for i in range(len(self.zetas)-1):
            self.phases[i+1] = (2*np.pi*self.T*(self.v/self.l*np.cos(self.zetas[i+1]-self.eta)+(k*self.f_off[1]-(k-1)*self.f_off[0])))%(2*np.pi)
        self.phases[-1] = (2*np.pi*self.T*(self.v/self.l*np.cos(self.eta)+(k*self.f_off[1]-(k-1)*self.f_off[0])))%(2*np.pi)
    
    def compute_input(self, k):
        """
        Computes realistic values of: \n measured phases, angle of arrivals, and variables. 
        """
        self.f_d = np.random.uniform(-self.fd_max,self.fd_max)        
        self.f_off[0] = np.random.normal(self.fo_mean,1e6)
        self.f_off[1] = self.f_off[0]*self.alpha+np.random.normal(0,self.std_w)
        self.v = np.random.uniform(0,self.v_max)
        self.zetas = np.random.uniform(-np.pi/4,np.pi/4,len(self.zetas))
        self.eta = np.random.uniform(0,2*np.pi)
        self.compute_phases(k=k)
    
    def update_input(self, k):
        self.f_off[0] = self.f_off[1] # f_off at time (k-1)
        self.f_off[1] = self.f_off[0]*self.alpha+np.random.normal(0,self.std_w) # f_off at time (k)
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

    def check_initial_values(self, v, f_d):
        if v<0 or v>self.v_max:
            v=2
        if f_d<-self.fd_max or f_d>self.fd_max:
            f_d=500
        return v,f_d

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
        

    def simulation(self, path, relative, noise=True, only_fD=True):
        SNR = np.array([0,5,10,15]) #
        N = 1000 # number of simulations
        interval = 100 # number of samples in which variables can be considered constant 
        SNR = np.power(10,SNR/10)
        p_std = np.sqrt(1/(2*256*SNR))
        zeta_std = [1,3,5]#
        mean_estimator = MeanEstimator()
        if not only_fD:
            ransac = RANSACRegressor(estimator=mean_estimator, min_samples=10, max_trials=400)
        ransac_fd = RANSACRegressor(estimator=mean_estimator, min_samples=10, max_trials=400)
        for z_std in np.deg2rad(zeta_std):
            tot_eta_error = []
            tot_f_d_error = []
            tot_v_error = []
            print('zeta std: ' + str(round(np.rad2deg(z_std))))
            for phase_std,snr in zip(p_std,SNR):
                eta_error = []
                f_d_error = []
                v_error = []
                for i in tqdm(range(N)):
                    etas_n = []
                    f_ds_n = []
                    vs_n = []
                    for i in range(1,interval):
                        if i==1:
                            self.compute_input(k=i)
                        else:
                            self.update_input(k=i)
                        n_phases, n_zetas = self.add_noise(noise, z_std, phase_std)
                        if len(self.phases)!=4:
                            x0 = [500,2,np.pi/3,1e5] # [f_d(0), v(0), eta(0), f_off(0)]
                            results = least_squares(self.system, x0, args=(n_phases, n_zetas))
                            etas_n.append(results.x[2])
                            f_ds_n.append(results.x[0])
                            vs_n.append(results.x[1])
                        else:
                            eta_n, f_d_n, v_n = self.solve_system(n_phases, n_zetas, new=True)
                            etas_n.append(eta_n)
                            f_ds_n.append(f_d_n)
                            vs_n.append(v_n)
                    ### RANSAC ###
                    #ransac_fd = RANSACRegressor(estimator=mean_estimator, min_samples=int(interval/2))
                    time = np.arange(len(f_ds_n)).reshape(-1,1)
                    # Doppler frequency
                    try:
                        ransac_fd.fit(time,f_ds_n)
                        pred_f_d = ransac_fd.predict(time)
                        #self.plot_ransac(time,np.ones(len(time))*self.f_d,f_ds_n,pred_f_d,ransac_fd.inlier_mask_)
                    except:
                        #print('ransac exception fD')
                        pred_f_d = np.mean(f_ds_n)
                    if relative:
                        f_d_error.append(np.abs((self.f_d-np.mean(pred_f_d))/self.f_d))
                    else:
                        f_d_error.append(np.abs(self.f_d-np.mean(pred_f_d)))
                    if not only_fD:
                        # eta
                        etas_n = np.mod(etas_n,2*np.pi)
                        try:
                            ransac.fit(time,etas_n)
                            pred_eta = ransac.predict(time)
                        except:
                            #print('ransac exception eta')
                            pred_eta = np.mean(etas_n)
                        if relative:
                            eta_error.append(np.abs((np.rad2deg(self.eta)-np.mean(np.rad2deg(pred_eta)))/np.rad2deg(self.eta)))
                        else:
                            eta_error.append(np.abs((np.rad2deg(self.eta)-np.mean(np.rad2deg(pred_eta)))))
                        # speed
                        try:
                            ransac.fit(time, vs_n)
                            pred_v = ransac.predict(time)
                        except:
                            #print('ransac exception speed')
                            pred_v = np.mean(vs_n)
                        if relative:
                            v_error.append(np.abs((self.v-np.mean(pred_v))/self.v))
                        else:
                            v_error.append(np.abs((self.v-np.mean(pred_v))))
                    ##############
                    
                if not only_fD:
                    tot_eta_error.append(eta_error)
                    tot_v_error.append(v_error)
                tot_f_d_error.append(f_d_error)
            
            if relative:    
                self.boxplot_plot(path, tot_f_d_error, "SNR (dB)", "relative frequency Doppler errors", 10*np.log10(SNR), "frequency Doppler errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'fd_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                if not only_fD:
                    self.boxplot_plot(path, tot_eta_error, "SNR (dB)", "relative eta errors", 10*np.log10(SNR), "eta errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'eta_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                    self.boxplot_plot(path, tot_v_error, "SNR (dB)", "relative speed error", 10*np.log10(SNR), "speed errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'speed_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
            else:
                self.boxplot_plot(path, tot_f_d_error, "SNR (dB)", "frequency Doppler errors (Hz)", 10*np.log10(SNR), "frequency Doppler errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'fd_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                if not only_fD:
                    self.boxplot_plot(path, tot_v_error, "SNR (dB)", "speed error (m/s)", 10*np.log10(SNR), "speed errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'speed_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
                    self.boxplot_plot(path, tot_eta_error, "SNR (dB)", "eta errors (°)", 10*np.log10(SNR), "eta errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'eta_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))

if __name__=='__main__':

    # sim = Simulation(T=0.2e-3)
    # path = 'plots/fc_60/var_w_1/alpha_099/'
    # sim.simulation(path, relative=True, noise=True, only_fD=True)
    
    sim = Simulation(T=0.2e-3,n_static=3,var_w=1e3)
    path = 'plots/fc_60/var_w_1/alpha_099/3_static/'
    sim.simulation(path, relative=True, noise=True, only_fD=True)

    sim = Simulation(T=0.2e-3,n_static=4,var_w=1e3)
    path = 'plots/fc_60/var_w_1/alpha_099/4_static/'
    sim.simulation(path, relative=True, noise=True, only_fD=True)

    sim = Simulation(T=0.2e-3,n_static=5, var_w=1e3)
    path = 'plots/fc_60/var_w_1/alpha_099/5_static/'
    sim.simulation(path, relative=True, noise=True, only_fD=True)

    # sim = Simulation(T=0.2e-3,var_w=10e3)
    # path = 'plots/fc_60/var_w_10/alpha_099/'
    # sim.simulation(path, relative=True, noise=True, only_fD=True)

    # sim = Simulation(T=0.2e-3,var_w=10e3, alpha=1)
    # path = 'plots/fc_60/var_w_10/alpha_1/'
    # sim.simulation(path, relative=True, noise=True, only_fD=True)

    # sim = Simulation(T=0.2e-3,var_w=100e3)
    # path = 'plots/fc_60/var_w_100/alpha_099/'
    # sim.simulation(path, relative=True, noise=True, only_fD=True)

    # sim = Simulation(l=0.06,T=1.45e-3,v_max=10,fo_mean=8.5e3,fd_max=330)
    # path = 'plots/fc_5/'
    # sim.simulation(path, relative=True, noise=True, only_fD=True)

    # sim = Simulation(l=0.011,T=0.2e-3,v_max=10,fo_mean=47e3,fd_max=1900)
    # path = 'plots/fc_28/'
    # sim.simulation(path, relative=True, noise=True, only_fD=True)
