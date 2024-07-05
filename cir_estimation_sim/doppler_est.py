import sys
import os
sys.path.insert(0,os.getcwd())
from cir_estimation_sim.channel_model import ChannelModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import time

class DopplerEst(ChannelModel):

    def __init__(self, l, n_static, snr):
        super().__init__(l=l, n_static=n_static[-1], snr=snr)

    def get_phases(self, h, init=False, from_index= True, plot=False):
        """
            Returns cir phases [LoS,t,s1,...,sn_static].
            h: cir,
            from_index: if True use the correct index to locate peaks, else performs peak detection,
            plot: wether to plot the selected paths from the given cir.
        """
        if from_index:
            ind = np.floor(self.paths['delay']*self.B).astype(int) # from paths delay
        else:
            ind = np.argsort(np.abs(h))[-len(self.paths['delay']):] # from cir peaks
        phases = np.angle(h[ind])
        if init:
            self.phases = np.zeros((self.n_static+2,2))
        self.phases[:,0] = self.phases[:,1]
        self.phases[:,1] = phases
        if plot:
            t = np.zeros(len(h))
            t[ind] = np.abs(h[ind])
            plt.stem(np.abs(self.cir), label='real')
            plt.stem(t, markerfmt='gD', label='selected paths')
            plt.grid()
            plt.legend()
            plt.show()
        return phases 
    
    def solve_system(self, phases, zetas):
        """
            Solves the system for the received input parameters. 
            phases: measured phases,
            zetas: angle of arrivals.

            returns the unknowns of the system.
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
            Defines the system to solve it using the non linear least-square method, for at least 4 measured phases (i.e. 2 static paths).
            x = [f_D, v, eta]: system unknowns,
            phases: measured phases,
            zetas: angle of arrivals.

            returns an array representing the system.
        """
        results = []
        #target path
        results.append(phases[0]-(2*np.pi*self.T*(x[0]+(x[1]/self.l*(np.cos(n_zetas[0]-x[2])-np.cos(x[2]))))))
        # loop for each static path, i.e., excluding LoS and Target
        for i in range(1,len(phases)):
            results.append(phases[i]-(2*np.pi*self.T*(x[1]/self.l*(np.cos(n_zetas[i]-x[2])-np.cos(x[2])))))
        return np.array(results)
    
    def check_initial_values(self, x0):
        """
            Check if the selected initial values are within the known intervals and change them to default value if not.
            x0: [f_D(0), v(0), eta(0)]: initial values.
        """
        if x0[0]<-self.fd_max or x0[0]>self.fd_max:
            x0[0]=self.fd_max/2
        if x0[1]<0 or x0[1]>self.vmax:
            x0[1]=2
        x0[2] = x0[2]%(2*np.pi)
        return x0  
    
    def my_mod_2pi(self, phases):
        """
            Maps an array or a matrix of angles [rad] in [-pi,pi].
            phases: measured phases.

            returns phases mapped in [-pi,pi].
        """
        if phases.ndim==2:
            for j in range(phases.shape[1]):
                for i,p in enumerate(phases[:,j]):
                    while p <-np.pi:
                        p = p+2*np.pi
                    while p>np.pi:
                        p = p-2*np.pi
                    phases[i,j]=p
        elif phases.ndim==1:
            for i,p in enumerate(phases):
                    while p <-np.pi:
                        p = p+2*np.pi
                    while p>np.pi:
                        p = p-2*np.pi
                    phases[i]=p
        else:
            raise Exception("phases number of dimensions must be <= 2.")
        return phases
    
    def get_phase_diff(self, interval):
        phase_diff = []
        self.k = 0
        h = self.get_cir_est(init=True, k=self.k)
        self.get_phases(h, init=True)
        for p in range(1,len(self.phases[:,1])):
                self.phases[p,1] = self.phases[p,1] - self.phases[0,1]
        AoA = [self.paths['AoA'][1:] + np.random.normal(0,self.AoAstd,self.n_static+1)]
        for i in range(1,interval):
            self.k = i
            h = self.get_cir_est(init=False, k=self.k)
            self.get_phases(h,plot=False)
            ### remove LoS from other paths ###
            for p in range(1,len(self.phases[:,1])):
                self.phases[p,1] = self.phases[p,1] - self.phases[0,1]
            ### phase difference ###
            diff = self.phases[:,1] - self.phases[:,0]
            phase_diff.append(diff)
            ### collect noisy AoA measurements ###
            AoA.append(self.paths['AoA'][1:] + np.random.normal(0,self.AoAstd,self.n_static+1))

        phase_diff = np.stack(phase_diff, axis=0)
        phase_diff = self.my_mod_2pi(phase_diff)
        AoA = np.stack(AoA,axis=0)
        return phase_diff, AoA

    def simulation(self, N, snr, interval, n_static, aoa):
        nls_time = []
        for j in range(N):
            print("iteration: ", j, end="\r")
            for s in snr:
                self.SNR = s
                for a in aoa:
                    self.AoAstd = np.deg2rad(a)
                    phase_diff, AoA = self.get_phase_diff(interval[-1])
                    for inter in interval: # interval in decreasing order
                        inter = int(inter*1e-3/self.T)
                        AoA = AoA[:inter]
                        phase_diff = phase_diff[:inter]
                        ### time average ###
                        mean_AoA = np.mean(AoA,axis=0)
                        mean_phase_diff = np.mean(phase_diff, axis=0) 
                        mean_phase_diff = mean_phase_diff[1:]
                        ### check phase diff < pi ###
                        for i,p in enumerate(mean_phase_diff):
                            if p>np.pi:
                                mean_phase_diff[i] = p - 2*np.pi
                        
                        if self.v_rx==0:
                            eta = 0
                            f_d = mean_phase_diff[0]/(2*np.pi*self.T)
                            v = 0
                            x0 = [f_d, v, eta]
                        else:
                            eta, f_d, v = self.solve_system(mean_phase_diff,mean_AoA)
                            x0 = [f_d, v, eta]
                            x0 = self.check_initial_values(x0)

                        npath_err = []
                        nls_t = []
                        for k in n_static:
                            start = time.time()
                            results = least_squares(self.system, x0, args=(phase_diff[:k+2], AoA[:k+2]))    
                            nls_t.append(time.time()-start) 
                            npath_err.append(abs((self.fd-np.mean(results.x[0]))/self.fd))   
                        nls_time.append(nls_t)