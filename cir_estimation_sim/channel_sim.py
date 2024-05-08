import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from commpy.filters import rrcosfilter
from scipy.signal import correlate, correlation_lags
from scipy.optimize import least_squares
import seaborn as sns

class channel_sim():

    def __init__(self,v_rx,fd,T=0.08e-3,SNR=50,G_tx=1,G_rx=1,P_tx=1,l=0.005,n_static=2,B=1.76e9,os=16,vmax=5,tx=None,rx=None):
        """
            v_rx: receiver speed vector (2 components)[m/s]
            fd: Doppler shift caused by target movement [Hz]
            G_tx/G_rx: transmitter/receiver antenna gain
            P_tx: transmitted power
            l: wavelength [m] 
            n_static: number of static paths
            B: bandwidth [Hz]
            os: over sampling rate (t' = t / os)
            tx/rx: transmitter/receiver Cartesian coordinates(default [0,0]/[x_max,y_max]) [m]
        """
        self.vrx = v_rx # speed vector
        self.v_rx = (v_rx[0]**2+v_rx[1]**2)**(0.5) # speed modulus
        self.eta = None # speed direction w.r.t. LoS
        self.fd = fd
        self.f_off = 0
        self.vmax = vmax
        
        self.SNR = SNR # dB
        self.G_tx = G_tx
        self.G_rx = G_rx
        self.P_tx = P_tx
        self.l = l
        self.n_static = n_static
        self.B = B
        self.T = T # interpacket time (cir samples period)
        self.os = os
        self.tx = tx
        self.rx = rx

        self.trn_field = None
        self.cir = None
        self.rx_signal = None
        self.k = 0 # dicrete time index

        self.beta = np.zeros(n_static+1)
        self.positions = np.zeros((self.n_static+3,2)) # positions coordinates [rx,tx,t,s1,...,sn_static]
        self.paths = np.zeros((n_static+2,4)) # [delay, phase, attenuations, AoA] for each path [LoS,t,s1,...,sn_static]
        self.phases = np.zeros((n_static+2,2)) # estimated phases at time k-1 and k
        
        self.h_rrc = rrcosfilter(129, alpha=1, Ts=1/(self.B), Fs=os*self.B)[1] 
        n = int(1e-3/self.T) # number of samples in 1 ms
        fo_max = 3e8/(self.l*10e6) # 0.1 ppm of the carrier frequency 
        self.std_w = fo_max/(3*np.sqrt(n**3)) # std for the fo random walk s.t. its max drift in 1 ms is fo_max        

    def get_positions(self,x_max,y_max,res_min=1,dist_min=0.5,plot=False):
        """
            generate random positions for rx, tx, n_static objects in 2D.
            x_max: maximum x coordinate [m]
            y_max: maximum y coordinate [m]
            res_min: minimum distance resolution [m]
            dist_min: minimum distance between two reflector/scatterer [m]
            returns an array of positions coordinates [rx,tx,t,s1,s2]
        """
        #pos = [[0.5,3],[2,0.25],[4,0.75]]
        self.positions
        beta_max = 2*np.arccos(3e8/(2*self.B*res_min))
        self.positions[0,:] = x_max,y_max
        try:
            if self.tx:
                self.positions[1,:] = self.tx
            else:
                self.tx = self.positions[1,:]
        except:
            self.positions[1,:] = self.tx
        try:
            if self.rx:
                self.positions[0,:] = self.rx
            else:
                self.rx = self.positions[0,:]
        except:
            self.positions[0,:] = self.rx
        self.paths[0,0] = self.dist(self.positions[0,:],self.positions[1,:])/3e8 # LoS delay
        for i in range(2,self.n_static+3):
            x = np.random.uniform(0,x_max)
            y = np.random.uniform(0,y_max)
            beta = np.pi
            while True:
                check = []
                for j in range(i):
                    check.append(self.dist(self.positions[j,:],[x,y])<dist_min)
                if any(check):        
                    x = np.random.uniform(0,x_max)
                    y = np.random.uniform(0,y_max)
                    continue        
                m_1 = (y-self.positions[1,1])/(x-self.positions[1,0])
                q_1 = y-m_1*x
                m_2 = -1/m_1
                q_2 = self.positions[0,1]-m_2*self.positions[0,0]
                x_p = (q_1-q_2)/(m_2-m_1)
                y_p = m_1*x_p+q_1
                d = self.dist([x,y],[x_p,y_p])
                ip = self.dist(self.positions[0,:],[x,y])
                if self.dist(self.positions[1,:],[x_p,y_p])>self.dist(self.positions[1,:],[x,y]):
                    beta = np.pi-np.arccos(d/ip)
                else:
                    beta = np.arccos(d/ip)
                assert int(ip**2)==int(d**2+((self.positions[0,0]-x_p)**2+(self.positions[0,1]-y_p)**2))
                if beta<beta_max:
                    self.beta[i-2] = beta
                    self.positions[i,:] = x,y
                    # path delay for non-LoS
                    self.paths[i-1,0] = (self.dist(self.positions[1,:],self.positions[i,:])+self.dist(self.positions[i,:],self.positions[0,:]))/3e8
                    check = []
                    for j in range(0,i):
                        for k in range(0,i):
                            if k!=j:
                                check.append(abs(self.paths[j,0]-self.paths[k,0])>1/self.B)
                    if all(check):
                        break
                x = np.random.uniform(0,x_max)
                y = np.random.uniform(0,y_max)
            ### fix static objects pos ###
            #x = pos[i-2][0]
            #y = pos[i-2][1]
            ##############################
        alpha = np.arctan(self.vrx[1]/self.vrx[0]) # speed direction w.r.t. positive x-axis
        if self.vrx[0]<0:
            alpha = alpha + np.pi
        beta = np.arctan((self.positions[0,1]-self.positions[1,1])/(self.positions[0,0]-self.positions[1,0])) # LoS path direction w.r.t. positive x-axis
        if (self.positions[0,0]-self.positions[1,0])<0:
            beta = beta + np.pi
        alpha = alpha % (2*np.pi)
        beta = beta % (2*np.pi) 
        if alpha>beta:
            self.eta = alpha - beta 
        else:
            self.eta = 2*np.pi - beta + alpha 
        if plot:
            self.plot_pos() 

    def update_positions(self):    
        # rx constant motion
        self.positions[0,:] += self.v_rx * self.k * self.T
        #target 
        self.positions[2,:] 

    def plot_pos(self):
        """
            plots the environmental disposition.
        """
        self.compute_AoAs()
        print('AoA : LoS, t, s1, s2, ... \n' + str(np.rad2deg(self.paths[:,3])))
        plt.plot(self.positions[0,0],self.positions[0,1],'ro',label='rx')
        plt.plot(self.positions[1,0],self.positions[1,1],'go',label='tx')
        plt.plot(self.positions[2,0],self.positions[2,1],'r+',label='target')
        #plt.plot(self.positions[3:,0],self.positions[3:,1],'bo',label='static objects')
        plt.plot(self.positions[0:2,0],self.positions[0:2,1],label='LoS')
        for i in range(2,self.n_static+3):
            plt.plot([self.positions[1,0],self.positions[i,0],self.positions[0,0]],[self.positions[1,1],self.positions[i,1],self.positions[0,1]])
            if i != 2:
                plt.plot(self.positions[i,0],self.positions[i,1],'o',label='static object %s' % (i-2))
        plt.plot([self.positions[0,0],self.positions[0,0]+self.vrx[0]],[self.positions[0,1],self.positions[0,1]+self.vrx[1]], label='v_rx')
        plt.legend()
        print('beta angles: t, s1, s2, ... \n' + str(np.rad2deg(self.beta)))
        print('eta: ' + str(np.rad2deg(self.eta)))
        plt.show()
    
    def dist(self, p1, p2):
        """
            return distance between two points in 2D.
            p1,p2: [x,y] Cartesian coordinates
        """
        return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**(.5)
    
    def radar_eq(self,pos,rcs=1):
        """
            returns the attenuation due to a reflector at position pos.
            pos: [x,y] Cartesian coordinates
        """
        ## assume reflection, but no scatter 
        G_tx = 10**(25/10) ## assume tx gain equal 25dB
        return (G_tx*self.G_rx*self.l**2*rcs/((4*np.pi)**3*(self.dist(pos,self.positions[0,:])+self.dist(pos,self.positions[1,:]))**2))**(0.5)
    
    def path_loss(self):
        """
            returns the attenuation due to path loss (LoS).
        """
        return (self.G_tx*self.G_rx*self.l/((4*np.pi*self.dist(self.positions[0,:],self.positions[1,:]))**2))**(0.5)
    
    def get_delays(self):
        """
            compute delays for each path.
        """
        paths=np.zeros(len(self.paths))
        paths[0] = self.dist(self.positions[0,:],self.positions[1,:])/3e8
        for i in range(1,len(self.paths)):
            paths[i] = (self.dist(self.positions[1,:],self.positions[i+1,:])+self.dist(self.positions[i+1,:],self.positions[0,:]))/3e8
        return paths
    
    def compute_AoAs(self):
        """
            compute angles of arrival for each path.
            LoS AoA=0.
        """
        m = (self.positions[0,1]-self.positions[1,1])/(self.positions[0,0]-self.positions[1,0]) # LoS: y = mx+q
        q = self.positions[0,1]-m*self.positions[0,0]
        m1 = -1/m
        for i in range(1,len(self.paths)):
            x,y = self.positions[i+1,:]
            q1 = y-m1*x
            x_p = (q-q1)/(m1-m) 
            y_p = m*x_p+q       # (x_p,y_p) = target/static obj. projection on LoS
            c = self.dist([x_p,y_p],self.positions[0,:])
            ip = self.dist([x,y],self.positions[0,:])
            assert int(ip**2)==int(c**2+((x-x_p)**2+(y-y_p)**2))
            self.paths[i,3] = np.arccos(c/ip)
            # if y_p<y:
            #     self.paths[i,3] = 2*np.pi-self.paths[i,3]
            
    def compute_phases(self):
        """
            compute phases for each path.
            LoS phase initial offset=0.
        """
        self.paths[0,1] = (self.v_rx/self.l*np.cos(self.eta)) # LoS
        self.paths[1,1] = (self.fd + self.v_rx/self.l*np.cos(self.paths[1,3]-self.eta) )#+ np.random.uniform(0,2*np.pi)) ? # target
        for i in range(2,len(self.paths)):
            self.paths[i,1] = (self.v_rx/self.l*np.cos(self.paths[i,3]-self.eta) )#+ np.random.uniform(0,2*np.pi)) ? # static
    
    def compute_attenuations(self):
        """
            compute attenuations for each path.
        """
        self.paths[0,2] = self.path_loss()
        for i in range(1,len(self.paths)):
            self.paths[i,2] = self.radar_eq(self.positions[i+1])
    
    def load_trn_field(self):
        """
            load TRN field adding upsampling with rate self.os.
            t' = t / os
        """
        trn_unit = loadmat('cir_estimation_sim/TRN_unit.mat')['TRN_FIELD'].squeeze()
        self.trn_field = np.zeros(len(trn_unit)*self.os, dtype=complex)
        self.trn_field[::self.os] = trn_unit
        
        
    def compute_cir(self, plot=False):
        """
            compute channel impulse response
        """
        up_cir = np.zeros(256*16).astype(complex)
        cir = np.zeros(256).astype(complex)
        assert all(self.paths[:,0]==self.get_delays())
        self.compute_AoAs()
        self.compute_phases()
        self.compute_attenuations()
        delays = np.floor(self.paths[:,0]*self.B*self.os)
        delays_1 = np.floor(self.paths[:,0]*self.B)
        for i,(d,d1) in enumerate(zip(delays.astype(int),delays_1.astype(int))):
            up_cir[d] = up_cir[d] + self.paths[i,2] * np.exp(1j * 2 * np.pi * self.T * self.k * self.paths[i,1])
            cir[d1] = cir[d1] + self.paths[i,2] * np.exp(1j * 2 * np.pi * self.T * self.k * self.paths[i,1])
        assert np.count_nonzero(cir)==len(self.paths[:,0]) # check that all paths are separable
        if plot:
            plt.title('upsampled cir')
            plt.grid()
            plt.stem(abs(up_cir),markerfmt='D')#/(max(abs(cir))))
            plt.show()
        self.up_cir = up_cir
        self.cir = cir

    def gen_noise(self, len):
        """
            returns noise of specified length with power computed from the desired SNR level.
            len: noise length.
        """
        snr_lin = 10 ** (self.SNR / 10)
        noise_std = np.sqrt(1 ** 2 / snr_lin) # assuming signal amplitude=1
        noise = np.random.normal(0, noise_std / np.sqrt(2), size=len) + 1j * np.random.normal(
            0, noise_std / np.sqrt(2), size=len
        )
        return noise

    def get_rxsignal(self, plot=False):
        """
            returns the received signal after: transmission, channel, and reception with additive noise.
            plot: wether to plot the received signal.
        """
        filt = np.convolve(self.trn_field.real,self.h_rrc) + 1j*np.convolve(self.trn_field.imag,self.h_rrc)
        rx_signal = np.convolve(filt,self.up_cir)
        noise = self.gen_noise(len(rx_signal))
        #rx_signal += noise
        rx_signal = np.convolve(rx_signal.real,np.flip(self.h_rrc)) + 1j*np.convolve(rx_signal.imag,np.flip(self.h_rrc))
        # g = np.convolve(self.h_rrc, self.h_rrc)
        # plt.plot(self.h_rrc)
        # plt.plot(np.flip(self.h_rrc))
        # plt.grid()
        # plt.show()
        if plot:
            #plt.plot(filt_real)
            #plt.show()
            plt.title('rx signal')
            plt.grid()
            plt.plot(rx_signal.real)
            plt.show()
        return rx_signal

    def sampling(self, up_rx_signal, plot=False):
        """
            compute the sampled received signal after a synchronization performed exploiting the first Golay sequence.
            up_rx_signal: signal to sample;
            plot: wether to plot the the selected samples of the original signal.
        """
        Ga = self.load_Golay_seqs()[0]
        up_Ga = np.zeros(len(Ga)*self.os, dtype=complex)
        up_Ga[::self.os] = Ga
        rrcos_up_Ga = np.convolve(up_Ga.real,self.h_rrc) + 1j*np.convolve(up_Ga.imag,self.h_rrc)
        rcos_up_Ga = np.convolve(rrcos_up_Ga.real,np.flip(self.h_rrc)) + 1j*np.convolve(rrcos_up_Ga.imag,np.flip(self.h_rrc))
        xcorr = np.correlate(up_rx_signal, rcos_up_Ga)[0:256*self.os] # check just for the first peak 
        #plt.plot(xcorr.real)
        #plt.show()
        start = np.argmax(xcorr)%self.os
        up_rx_signal = up_rx_signal[start:]
        self.rx_signal = up_rx_signal[::self.os]
        if plot:
            plt.title('Sampling')
            x = np.linspace(0, len(self.rx_signal), 16*len(self.rx_signal))
            y = up_rx_signal[:16*len(self.rx_signal)].real
            plt.plot(x[:-len(x)+len(y)],y)
            plt.stem(self.rx_signal.real, 'r')
            plt.show()
        self.rx_signal = self.rx_signal[int(np.floor(len(self.h_rrc)/2)/self.os)*2:] #compensate for the filters time shift 

    def add_cfo(self, h_est):
        """
            return the cir estimate after adding the channel frequency offset shift.
            h_est: signal (cir estimate).
        """
        self.f_off += np.random.normal(0,self.std_w)
        return h_est * np.exp(1j*2*np.pi*self.f_off*self.k*self.T)       
        
    def load_Golay_seqs(self):
        Ga = loadmat("cir_estimation_sim/Ga128_rot_2sps.mat")["Ga128_rot_2sps"].squeeze()
        Gb = loadmat("cir_estimation_sim/Gb128_rot_2sps.mat")["Gb128_rot_2sps"].squeeze()
        return Ga, Gb

    def estimate_CIR(self, signal, plot=False):
        """
            returns the estimated channel impulse response performing correlation with the TRN field Golay sequences.
            signal: received, sampled signal.
        """
        Ga,Gb = self.load_Golay_seqs()
        Gacor = correlate(signal, Ga)
        Gbcor = correlate(signal, Gb)
        # get index of the 0 lag correlation
        lags = correlation_lags(len(signal), len(Ga))
        start = np.argwhere(lags == 0)[0][0]
        # cut starting at 0 lag
        Gacor2 = Gacor[start:]
        Gbcor2 = Gbcor[start:]
        # align the a and b sequences
        Gacor3 = Gacor2[: -2 * len(Ga)]
        Gbcor3 = Gbcor2[len(Ga) : -len(Ga)]
        # extract the 3 Golay subsequences
        Gacor4 = np.stack(
            [
                Gacor3[: len(Ga) * 2],
                Gacor3[len(Ga) * 2 : len(Ga) * 4],
                Gacor3[len(Ga) * 4 : len(Ga) * 6],
            ],
            axis=1,
        )
        Gbcor4 = np.stack(
            [
                Gbcor3[: len(Gb) * 2],
                Gbcor3[len(Gb) * 2 : len(Gb) * 4],
                Gbcor3[len(Gb) * 4 : len(Gb) * 6],
            ],
            axis=1,
        )

        # pair complementary sequences
        # +Ga128, -Gb128, +Ga128. +Gb128, +Ga128, -Gb128
        a_part = Gacor4 * np.array([[1, 1, 1]])
        b_part = Gbcor4 * np.array([[-1, 1, -1]])
        ind_h = a_part + b_part

        # add individual results
        h_128 = ind_h.sum(axis=1)

        # add cfo 
        #h_128 = self.add_cfo(h_128)

        if plot:
            plt.stem(np.abs(h_128)/max(abs(h_128)), linefmt='r', markerfmt='rD', label='estimate')
            plt.stem(np.linspace(0,256,16*256),abs(self.up_cir)/max(abs(self.up_cir)), linefmt='g', markerfmt='gD', label='real upsampled')
            plt.stem(np.abs(self.cir)/max(abs(self.cir)), label='real sampled')
            plt.legend()
            plt.show()

        return h_128
    
    def get_phases(self, h, plot=False):
        """
            return cir phases [LoS,t,s1,...,sn_static].
            h: cir;
            plot: wether to plot the selected paths from the given cir.
        """
        ind = np.argsort(np.abs(h))[-len(self.paths[:,0]):] # from cir peaks
        ind = np.floor(self.paths[:,0]*self.B).astype(int) # from paths delay
        phases = np.angle(h[ind])
        self.phases[:,0] = self.phases[:,1]
        self.phases[:,1] = phases
        if plot:
            t = np.zeros(len(h))
            t[ind] = np.abs(h[ind])
            plt.stem(np.abs(h)/max(abs(h)), linefmt='r', markerfmt='rD', label='estimate')
            plt.stem(np.abs(ch_sim.cir)/max(abs(ch_sim.cir)), label='real')
            plt.stem(t/max(t), markerfmt='gD', label='selected paths')
            plt.grid()
            plt.legend()
            plt.show()
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
        for i in range(1,len(phases)):
            results.append(phases[i]-(2*np.pi*self.T*(x[1]/self.l*(np.cos(n_zetas[i]-x[2])-np.cos(x[2])))))
        return np.array(results)
    
    def check_initial_values(self, x0):
        """
            Check if the selected initial values are within the known intervals and change them to default value if not.

            x0 = [f_D(0), v(0), eta(0)]: initial values.
        """
        fd_max = 2/self.l*self.vmax
        if x0[0]<-fd_max or x0[0]>fd_max:
            x0[0]=fd_max/2
        if x0[1]<0 or x0[1]>self.vmax:
            x0[1]=2
        x0[2] = x0[2]%(2*np.pi)
        return x0  
    
    def simulation(self, x_max, y_max, N, interval, path, relative=True, save=True):
        f_d_error = []
        self.load_trn_field()
        for j in tqdm(range(N),dynamic_ncols=True):
            phase_diff = []
            self.k = 1
            self.get_positions(x_max,y_max,plot=False)
            self.compute_cir(plot=False)
            up_rx_signal = self.get_rxsignal()
            self.sampling(up_rx_signal)
            h = self.estimate_CIR(self.rx_signal)
            h_test = self.add_cfo(self.cir)
            self.get_phases(h_test)
            for p in range(1,len(self.phases[:,1])):
                    self.phases[p,1] = self.phases[p,1] - self.phases[0,1]
            for i in range(1,interval):
                self.k += 1
                self.compute_cir()
                self.sampling(self.get_rxsignal(),plot=False)
                h = self.estimate_CIR(self.rx_signal,plot=False)
                h_test = self.add_cfo(self.cir)
                self.get_phases(h_test,plot=False)
                ### remove LoS from other paths ###
                for p in range(1,len(self.phases[:,1])):
                    self.phases[p,1] = self.phases[p,1] - self.phases[0,1]
                ### phase difference ###
                diff = self.phases[:,1] - self.phases[:,0]
                phase_diff.append(diff)

            ### time average ###
            phase_diff = np.stack(phase_diff, axis=0)
            phase_diff = phase_diff%(2*np.pi)
            phase_diff = np.mean(phase_diff, axis=0) 
            phase_diff = phase_diff[1:]
            ### check phase diff < pi ###
            for i,p in enumerate(phase_diff):
                if p>np.pi:
                    phase_diff[i] = p - 2*np.pi
            # AoA add noise or realistic estimation?
            AoA = self.paths[1:,3]
            eta, f_d, v = self.solve_system(phase_diff,AoA)
            x0 = [f_d, v, eta]
            x0 = self.check_initial_values(x0)
            results = least_squares(self.system, x0, args=(phase_diff, AoA))
            if relative:
                err = np.abs((self.fd-np.mean(results.x[0]))/self.fd)
                if err>0.01:
                    a=0
                else:
                    a=1
                if err>250:
                    print('error: '+str(err))
                    print('real fD: '+str(self.fd))
                    print('est. fD: '+str(np.mean(results.x[0])))
                f_d_error.append(err)
            else:
                f_d_error.append(np.abs(self.fd-np.mean(results.x[0])))
        if save:
                np.save(path+'fd_k'+str(interval)+'_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',f_d_error)
        return f_d_error
    
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
        #plt.savefig(path+name+'.png')
        plt.show()
        #tik.save(path+name+'.tex')
        plt.close()



if __name__=='__main__':

    ch_sim = channel_sim(v_rx=[1,2],fd=1000, SNR=100)
   
    # ch_sim.get_positions(5,5, plot=False)

    # ch_sim.compute_cir(plot=False)

    # up_rx_signal = ch_sim.get_rxsignal(plot=False)

    # ch_sim.sampling(up_rx_signal, plot=False)

    # h_128 = ch_sim.estimate_CIR(ch_sim.rx_signal, plot=False)

    # phases = ch_sim.get_phases(h_128, plot=True)

    f_d_error = ch_sim.simulation(5,5,10,50,'',save=False)
    print(np.mean(f_d_error))
    #ch_sim.plot_boxplot('',[f_d_error],'','relative fD error',[1],'')