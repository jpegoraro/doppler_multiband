from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate, correlation_lags, deconvolve
from scipy.optimize import least_squares
import time
import tikzplotlib as tik
#from tqdm import tqdm

class channel_sim():

    def __init__(self, vmax, SNR=20, AoAstd=np.deg2rad(5), G_tx=1, G_rx=1, P_tx=1, l=0.005, 
                 n_static=2, us=16, static_rx=False, tx=None, rx=None, T=None):
        """
            vmax: maximum receiver speed [m/s], (60 GHz-->vmax=5 m/s, 28 GHz-->vmax=10 m/s, 5 GHz-->vmax=20 m/s) these are just reasonable values
            SNR: signal to noise ratio [dB],
            AoAstd: std of the noise added to the angles of arrivals measurements [rad],
            G_tx/G_rx: transmitter/receiver antenna gain,
            P_tx: transmitted power,
            l: wavelength [m],
            n_static: number of static paths,
            us: up sampling rate (ts = t / us),
            tx/rx: transmitter/receiver Cartesian coordinates(default [0,0]/[x_max,y_max]) [m].
        """
        self.eta = None # speed direction w.r.t. LoS
        vmin = 0.5 # if RX moves with a speed below 0.5 m/s it is considered static
        self.v_rx = np.random.uniform(vmin,vmax) # speed modulus
        if static_rx:
            self.v_rx = 0
        self.vrx = None # speed vector
        self.fd_max = vmax/l # max achievable Doppler frequency
        #fd_min = vmin/l # if target moves with a speed below 0.5 m/s it is considered static
        self.fd_min = 100
        self.fd = np.random.uniform(self.fd_min,self.fd_max)
        if np.random.rand()>0.5:
            self.fd = - self.fd
        self.f_off = 0
        self.vmax = vmax
        
        self.SNR = SNR # dB
        self.AoAstd = AoAstd
        self.G_tx = G_tx
        self.G_rx = G_rx
        self.P_tx = P_tx
        self.l = l
        self.n_static = n_static
        if not T:
            self.T = 1/(6*self.fd_max) # interpacket time (cir samples period)
        else:
            self.T = T*1e-3 # [s]
        self.us = us
        self.tx = tx
        self.rx = rx

        # select single carrier modulation for 60 GHz carrier frequency
        if l==0.005:
            self.B = 1.76e9 # bandwidth [Hz]
            self.tx_signal = self.load_trn_field()
        # select OFDM with 16QAM modulation for 28 GHz carrier frequency
        if l==0.0107:
            # 5G-NR parameters
            self.delta_f = 120e3 # subcarrier spacing [Hz]
            self.n_sc = 3332 # number of subcarriers
            self.B = self.n_sc*self.delta_f # bandwidth [Hz] (almost 400 MHz)
            #self.tx_signal = self.generate_16QAMsymbols(self.n_sc)
            #np.save('cir_estimation_sim/28_TXsignal.npy',self.tx_signal)
            #self.tx_signal = np.load('cir_estimation_sim/28_TXsignal.npy')
            self.tx_signal = self.generate_bpsk(self.n_sc)
        if l==0.06:
            # 802.11ax parameters
            self.delta_f = 78.125e3 # subcarrier spacing [Hz]
            self.n_sc = 2048 # number of subcarriers
            self.B =  self.n_sc*self.delta_f # bandwidth [Hz] (160 MHz)
            #self.tx_signal = self.generate_16QAMsymbols(self.n_sc)
            #self.tx_signal = np.load('cir_estimation_sim/5_TXsignal.npy')
            self.tx_signal = self.generate_bpsk(self.n_sc)
        self.cir = None
        self.rx_signal = None
        self.k = 0 # dicrete time index

        self.beta = np.zeros(n_static+1)
        self.positions = np.zeros((self.n_static+3,2)) # positions coordinates [rx,tx,t,s1,...,sn_static]
        self.paths = np.zeros((n_static+2,4)) # [delay, phase, attenuations, AoA] for each path [LoS,t,s1,...,sn_static]
        self.paths = {
            'delay': np.zeros(n_static+2),
            'phase': np.zeros(n_static+2),
            'AoA'  : np.zeros(n_static+2),
            'gain' : np.zeros(n_static+2, dtype=complex)
        }
        self.phases = np.zeros((n_static+2,2)) # estimated phases at time k-1 and k
        
        self.h_rrc = self.rrcos(129,self.us,1)
        n = int(1e-3/self.T) # number of samples in 1 ms
        fo_max = 3e8/(self.l*10e6) # 0.1 ppm of the carrier frequency 
        self.std_w =  fo_max/(6*np.pi*self.T*(n+1))# std for independent samples of the cfo s.t. its max drift in 1 ms is fo_max fo_max/(3*np.sqrt(n**3)) #--->  std for the fo random walk s.t. its max drift in 1 ms is fo_max #  
        a = 0
        
    def generate_16QAMsymbols(self, n_sc, unitAveragePower=True):
        txsymbols = np.random.randint(0,16,n_sc)
        QAM_mapper = []
        for i in [-1,-1/3,1/3,1]:
            for j in [-1,-1/3,1/3,1]:
                QAM_mapper.append(i+1j*j)
        QAM_mapper  = np.reshape(QAM_mapper, len(QAM_mapper))
        QAM_symbols = QAM_mapper[txsymbols]
        if unitAveragePower:
            power = np.mean(abs(QAM_symbols)**2)
            QAM_symbols = QAM_symbols/power
        return QAM_symbols
    
    def generate_bpsk(self, n_sc):
        txsym = np.random.randint(0,2,n_sc)
        txsym[txsym==0] = -1
        return txsym

    def rrcos(self, n, T, beta):
        """
        Root raised cosine filter.
        n: number of filter taps,
        T: upsampling factor = t/ts where t is the 'real period' and ts is the sampling period,
        beta: roll-off factor.

        returns the filter impulsive response.
        """
        h = np.zeros(n)
        start = int((n-1)/2)
        for t in range(int((n+1)/2)):
            if t==0:
                h[start+t] = 1/T*(1+(beta*((4/np.pi)-1)))
            elif t==T/(4*beta):
                h[start+t] = beta/(T*2**(0.5)) * ( (1+(2/np.pi))*np.sin(np.pi/(4*beta)) + (1-(2/np.pi))*np.cos(np.pi/(4*beta)))
            else:
                h[start+t] = 1/T *( (np.sin(np.pi*t/T*(1-beta)) + (4*beta*t/T*np.cos(np.pi*t/T*(1+beta)))) / (np.pi*t/T*(1-(4*beta*t/T)**2)) )
        h[:start] = np.flip(h[start+1:])
        # normalization 
        g = np.convolve(h, h)
        g = g/max(g)
        h1 = deconvolve(g,h)[0]
        return h1
    
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
    
    def get_positions(self, x_max, y_max, res_min=1, dist_min=1, plot=False):
            """
                Generates random positions for rx, tx, n_static objects in 2D.
                x_max: maximum x coordinate [m],
                y_max: maximum y coordinate [m],
                res_min: minimum distance resolution [m],
                dist_min: minimum distance between two reflector/scatterer [m].

                returns an array of positions coordinates [rx,tx,t,s1,s2]
            """
            if self.l==0.06 or self.l==0.0107:
                res_min = 5
            #     x_max = 4*x_max
            #     y_max = 4*y_max
            alpha = np.random.uniform(0,2*np.pi) # speed direction w.r.t. positive x-axis
            self.vrx = [self.v_rx*np.cos(alpha),self.v_rx*np.sin(alpha)] # speed vector
            th = np.arccos(3e8/(2*self.B*res_min)) # threshold to check the system minimum distance resolution 
            assert 2*th<np.pi and 2*(2*np.pi-th)>3*np.pi 
            self.positions[0,:] = [x_max,y_max]
            self.positions[1,:] = [0,0]

            self.paths['delay'][0] = self.dist(self.positions[0,:],self.positions[1,:])/3e8 # LoS delay
            for i in range(2,self.n_static+3):
                x = np.random.uniform(0,x_max)
                y = np.random.uniform(0,y_max)
                beta = np.pi
                start = time.time()
                while True:
                    if time.time()-start>1:
                        self.get_positions(x_max,y_max)
                        break
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
                    if beta<2*th or (beta>np.pi and beta<3*np.pi) or beta>2*(2*np.pi-th):
                        self.beta[i-2] = beta
                        self.positions[i,:] = x,y
                        # path delay for non-LoS
                        self.paths['delay'][i-1] = (self.dist(self.positions[1,:],self.positions[i,:])+self.dist(self.positions[i,:],self.positions[0,:]))/3e8
                        check = []
                        for j in range(0,i):
                            for k in range(0,i):
                                if k!=j:
                                    check.append(abs(self.paths['delay'][j]-self.paths['delay'][k])>1/self.B) # check all path are separable
                        if all(check):
                            check = []
                            self.compute_AoA(ind=i-1)
                            for k,j in combinations(range(i),2):
                                check.append(abs(self.paths['AoA'][k]-self.paths['AoA'][j])>0.05) # AoAs must be different between them
                            if all(check):
                                break
                    x = np.random.uniform(0,x_max)
                    y = np.random.uniform(0,y_max)
            #while True:
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
            # if all(abs(self.paths['AoA'][1:]-(2*self.eta))>0.1): # check second existence condition (AoA!=2eta) 
            #     break
            # self.v_rx = np.random.uniform(0.5,self.vmax)
            # self.eta = np.random.uniform(0,2*np.pi)
            # self.vrx = [self.v_rx*np.cos(self.eta),self.v_rx*np.sin(self.eta)]
            if plot:
                self.plot_pos() 

    def compute_AoA(self, ind):
        """
            Computes angles of arrival for each path.
            ind: path index whose AoA has to be computed.
            LoS AoA=0.
        """
        m = (self.positions[0,1]-self.positions[1,1])/(self.positions[0,0]-self.positions[1,0]) # LoS: y = mx+q
        q = self.positions[0,1]-m*self.positions[0,0]
        m1 = -1/m
        
        x,y = self.positions[ind+1,:]
        q1 = y-m1*x
        x_p = (q-q1)/(m1-m) 
        y_p = m*x_p+q       # (x_p,y_p) = target/static obj. projection on LoS
        c = self.dist([x_p,y_p],self.positions[0,:])
        ip = self.dist([x,y],self.positions[0,:])
        assert int(ip**2)==int(c**2+((x-x_p)**2+(y-y_p)**2))
        self.paths['AoA'][ind] = np.arccos(c/ip)

    def update_positions(self):    
        # rx constant motion
        self.positions[0,:] += self.v_rx * self.k * self.T
        #target 
        self.positions[2,:] 

    def plot_pos(self):
        """
            Plots the environmental disposition.
        """
        print('AoA : LoS, t, s1, s2, ... \n' + str(np.rad2deg(self.paths['AoA'])))
        plt.plot(self.positions[0,0],self.positions[0,1],'ro',label='rx')
        plt.plot(self.positions[1,0],self.positions[1,1],'go',label='tx')
        plt.plot(self.positions[2,0],self.positions[2,1],'r+',label='target')
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
            Returns distance between two points in 2D.
            p1,p2: [x,y] Cartesian coordinates.
        """
        return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**(.5)
    
    def radar_eq(self, pos, rcs=1, scatter=False):
        """
            Returns the attenuation due to a reflector at position pos.
            pos: [x,y] Cartesian coordinates.
        """
        G_tx = 1#10**(20/10) ## assume tx gain for reflectors
        if scatter:
            a = (G_tx*self.G_rx*self.l**2*rcs/((4*np.pi)**3*(self.dist(pos,self.positions[0,:])*self.dist(pos,self.positions[1,:]))**2))**(0.5)
        else:
            a = (G_tx*self.G_rx*self.l**2*rcs/((4*np.pi)**3*(self.dist(pos,self.positions[0,:])+self.dist(pos,self.positions[1,:]))**2))**(0.5)
        return a
    
    def path_loss(self):
        """
            Returns the attenuation due to path loss (LoS).
        """
        return (self.G_tx*self.G_rx*self.l/((4*np.pi*self.dist(self.positions[0,:],self.positions[1,:]))**2))**(0.5)
    
    def get_delays(self):
        """
            Computes delays for each path.
        """
        paths=np.zeros(len(self.paths['delay']))
        paths[0] = self.dist(self.positions[0,:],self.positions[1,:])/3e8
        for i in range(1,len(paths)):
            paths[i] = (self.dist(self.positions[1,:],self.positions[i+1,:])+self.dist(self.positions[i+1,:],self.positions[0,:]))/3e8
        return paths
            
    def compute_phases(self):
        """
            Computes phases for each path.
            LoS phase initial offset=0.
        """
        self.paths['phase'][0] = (self.v_rx/self.l*np.cos(self.eta)) # LoS
        self.paths['phase'][1] = (self.fd + self.v_rx/self.l*np.cos(self.paths['AoA'][1]-self.eta) ) # target
        for i in range(2,len(self.paths['phase'])):
            self.paths['phase'][i] = (self.v_rx/self.l*np.cos(self.paths['AoA'][i]-self.eta) ) # static

    
    def compute_attenuations(self):
        """
            Computes attenuations for each path,
            all paths besides the LoS have also a random phase offset due to reflections.
        """
        self.paths['gain'][0] = self.path_loss()
        for i in range(1,len(self.paths['gain'])):
            self.paths['gain'][i] = self.radar_eq(self.positions[i+1]) * np.exp(1j * np.random.uniform(0,2*np.pi))
    
    def load_trn_field(self):
        """
            Loads TRN field adding upsampling with rate self.us
            ts = t / us
        """
        trn_unit = loadmat('cir_estimation_sim/TRN_unit.mat')['TRN_SUBFIELD'].squeeze()
        trn_field = np.zeros(len(trn_unit)*self.us)
        trn_field[::self.us] = trn_unit
        return trn_field
        
        
    def compute_cir(self, init, plot=False):
        """
            Computes channel impulse response.
            init: if it is the first cir sample compute phases and attenuations.
        """
        if self.l==0.005:
            up_cir = np.zeros(256*self.us).astype(complex)
            cir = np.zeros(256).astype(complex)
        else:
           cir = np.zeros(len(self.tx_signal)).astype(complex) 
        if init:
            assert all(self.paths['delay']==self.get_delays())
            self.compute_phases()
            self.compute_attenuations()
        delays = np.floor(self.paths['delay']*self.B)
        for i,d in enumerate(delays.astype(int)):
            if self.l==0.005:
                up_cir[d*self.us] = up_cir[d*self.us] + self.paths['gain'][i] * np.exp(1j * 2 * np.pi * self.T * self.k * self.paths['phase'][i])
            cir[d] = cir[d] + self.paths['gain'][i] * np.exp(1j * 2 * np.pi * self.T * self.k * self.paths['phase'][i])
        assert np.count_nonzero(cir)==len(self.paths['delay']) # check that all paths are separable
        if plot:
            plt.title('upsampled cir')
            plt.grid()
            plt.stem(abs(up_cir),markerfmt='D')
            plt.show()
        # assuming signal amplitude=1
        if self.l==0.005:
            self.up_cir = up_cir/abs(self.paths['gain'][0])
        self.cir = cir/abs(self.paths['gain'][0])

    def gen_noise(self, len):
        """
            Returns noise of specified length with power computed from the desired SNR level.
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
            Returns the received signal after: transmission, channel, and reception with additive noise.
            plot: wether to plot the received signal.
        """
        ### tx rrcos filter ###
        filt = np.convolve(self.tx_signal,self.h_rrc)
        filt_delay = int((len(self.h_rrc)-1)/2)
        filt = filt[filt_delay:]#compensate for the filters time shift 
        ### physical channel ###
        rx_signal = np.convolve(filt,self.up_cir) 
        ### add noise ###
        noise = self.gen_noise(len(rx_signal))
        rx_signal = rx_signal + noise
        ### rx rrcos filter ###
        rx_signal = np.convolve(rx_signal,self.h_rrc)
        rx_signal = rx_signal[filt_delay:]#compensate for the filters time shift 
        if plot:
            plt.plot(filt.real)
            plt.show()
            plt.title('rx signal')
            plt.grid()
            plt.plot(rx_signal.real)
            plt.show()
        return rx_signal

    def sampling(self, up_rx_signal, plot=False):
        """
            Computes the sampled received signal after a synchronization performed exploiting the first Golay sequence.
            up_rx_signal: signal to sample,
            plot: wether to plot the the selected samples of the original signal.
        """
        # synchronization by correlating the first Golay sequence 
        Ga = self.load_Golay_seqs()[0]
        up_Ga = np.zeros(len(Ga)*self.us)
        up_Ga[::self.us] = Ga
        rrcos_up_Ga = np.convolve(up_Ga,self.h_rrc) 
        rcos_up_Ga = np.convolve(rrcos_up_Ga,self.h_rrc) 
        xcorr = np.correlate(up_rx_signal, rcos_up_Ga)[0:256*self.us] # check just for the first peak 
        #plt.plot(xcorr.real)
        #plt.plot(abs(xcorr))
        #plt.show()
        start = np.argmax(abs(xcorr))%self.us
        # sampling from start to sample on the peaks
        up_rx_signal = up_rx_signal[start:]
        self.rx_signal = up_rx_signal[::self.us]
        if plot:
            plt.title('Sampling')
            x = np.linspace(0, len(self.rx_signal), self.us*len(self.rx_signal))
            y = up_rx_signal[:self.us*len(self.rx_signal)].real
            if len(x)==len(y):
                plt.plot(x,y)
            else:
                plt.plot(x[:-len(x)+len(y)],y)
            plt.stem(self.rx_signal.real, markerfmt='r')
            plt.show()

    def get_rx_ofdm(self, ofdm_symbol):
        H  = np.fft.fft(self.cir)
        Y = H*ofdm_symbol
        noise = self.gen_noise(len(Y))
        return Y + noise

    def add_cfo(self, signal):
        """
            Returns the signal after adding the channel frequency offset shift.
            signal: signal (cir estimate).
        """
        self.f_off = np.random.normal(0,self.std_w)
        return signal * np.exp(1j*2*np.pi*self.f_off*self.k*self.T)       
    
    def add_po(self, signal):
        """
            Returns the signal after adding the phase noise shift.
            signal: signal (cir estimate).
        """
        return signal * np.exp(1j*np.random.normal(0,self.std_w))
        
    def load_Golay_seqs(self):
        Ga = loadmat("cir_estimation_sim/Ga128.mat")["Ga"].squeeze()
        Gb = loadmat("cir_estimation_sim/Gb128.mat")["Gb"].squeeze()
        return Ga, Gb

    def estimate_CIR(self, signal, plot=False):
        """
            Returns the estimated channel impulse response performing correlation with the TRN field Golay sequences.
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

        if plot:
            plt.stem(np.abs(h_128)/max(abs(h_128)), linefmt='r', markerfmt='rD', label='estimate')
            #plt.stem(np.linspace(0,256,16*256),abs(self.up_cir)/max(abs(self.up_cir)), linefmt='g', markerfmt='gD', label='real upsampled')
            plt.stem(np.abs(self.cir)/max(abs(self.cir)), label='real sampled')
            plt.legend()
            plt.show()
            plt.stem(np.angle(h_128), linefmt='r', markerfmt='rD', label='estimate')
            plt.stem(np.angle(self.cir), label='real sampled')
            plt.legend()
            plt.show()

        return h_128
    
    def estimate_ofdm_CIR(self, Y, plot=False):
        G = Y/self.tx_signal
        g = np.fft.ifft(G)
        if plot:
            plt.stem(np.abs(g)/max(abs(g)), linefmt='r', markerfmt='rD', label='estimate')
            #plt.stem(np.linspace(0,256,16*256),abs(self.up_cir)/max(abs(self.up_cir)), linefmt='g', markerfmt='gD', label='real upsampled')
            plt.stem(np.abs(self.cir)/max(abs(self.cir)), label='real sampled')
            plt.legend()
            plt.show()
            plt.stem(np.angle(g), linefmt='r', markerfmt='rD', label='estimate')
            plt.stem(np.angle(self.cir), label='real sampled')
            plt.legend()
            plt.show()
        return g
    
    def get_phases(self, h, from_index= True, plot=False):
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
    
    def simulation(self, x_max, y_max, N, interval, path, relative=True, save=True, save_time=False, save_all=False):
        """
            Performs the simulation.
            x_max: max x Cartesian coordinate,
            y_max: max y Cartesian coordinate,
            N: number of simulation iterations,
            interval: number of frames during which parameters are considered constant,
            path: path to save output errors,
            relative: wether to compute relative errors (if False computes absolute errors),
            save: wether to save the output errors.

            returns target Doppler frequency estimate relative error.
        """
        f_d_error = []
        nls_time = []
        eta_abs_error = []
        eta_rel_error = []
        v_abs_error = []
        v_rel_error = []
        npath_error = []
        
        #for j in tqdm(range(N),dynamic_ncols=True):
        for j in range(N):
            print("iteration: ", j, end="\r")
            phase_diff = []
            self.k = 0
            self.get_positions(x_max,y_max,plot=False)
            self.compute_cir(init=True,plot=False)
            if self.l==0.005:
                up_rx_signal = self.get_rxsignal(plot=False)
                self.sampling(up_rx_signal,plot=False)
                h = self.estimate_CIR(self.rx_signal,plot=False)
            else:
                Y = self.get_rx_ofdm(self.tx_signal)
                h = self.estimate_ofdm_CIR(Y, plot=False)
            ### add cfo ###
            h = self.add_po(self.add_cfo(h))
            self.get_phases(h)
            for p in range(1,len(self.phases[:,1])):
                    self.phases[p,1] = self.phases[p,1] - self.phases[0,1]
            AoA = [self.paths['AoA'][1:] + np.random.normal(0,self.AoAstd,self.n_static+1)]
            for i in range(1,interval):
                self.k = i
                self.compute_cir(init=False)
                if self.l==0.005:
                    up_rx_signal = self.get_rxsignal(plot=False)
                    self.sampling(up_rx_signal,plot=False)
                    h = self.estimate_CIR(self.rx_signal,plot=False)
                else:
                    Y = self.get_rx_ofdm(self.tx_signal)
                    h = self.estimate_ofdm_CIR(Y, plot=False)
                ### add cfo ###
                h = self.add_po(self.add_cfo(h))
                self.get_phases(h,plot=False)
                ### remove LoS from other paths ###
                for p in range(1,len(self.phases[:,1])):
                   self.phases[p,1] = self.phases[p,1] - self.phases[0,1]
                ### phase difference ###
                diff = self.phases[:,1] - self.phases[:,0]
                phase_diff.append(diff)
                ### collect noisy AoA measurements ###
                AoA.append(self.paths['AoA'][1:] + np.random.normal(0,self.AoAstd,self.n_static+1))
            ### time average ###
            AoA = np.mean(np.stack(AoA,axis=0),axis=0)
            phase_diff = np.stack(phase_diff, axis=0)
            phase_diff = self.my_mod_2pi(phase_diff)
            phase_diff = np.mean(phase_diff, axis=0) 
            phase_diff = phase_diff[1:]
            ### check phase diff < pi ###
            for i,p in enumerate(phase_diff):
                if p>np.pi:
                    phase_diff[i] = p - 2*np.pi
            
            if self.v_rx==0:
                eta = 0
                f_d = phase_diff[0]/(2*np.pi*self.T)
                v = 0
                x0 = [f_d, v, eta]
            else:
                eta, f_d, v = self.solve_system(phase_diff,AoA)
                x0 = [f_d, v, eta]
                x0 = self.check_initial_values(x0)
            
            #if len(self.phases)<7:
            #results = least_squares(self.system, x0, args=(phase_diff, AoA), bounds=([-self.fd_max,0,0],[self.fd_max,self.vmax,2*np.pi]))
            #else:
            npath_err = []
            nls_t = []
            if self.n_static==8:
                for j in [2,4,6,8]:
                    start = time.time()
                    results = least_squares(self.system, x0, args=(phase_diff[:j+2], AoA[:j+2]))    
                    nls_t.append(time.time()-start) 
                    npath_err.append(abs((self.fd-np.mean(results.x[0]))/self.fd))   
                nls_time.append(nls_t)
            else:
                start = time.time()
                results = least_squares(self.system, x0, args=(phase_diff, AoA))
                nls_time.append(time.time()-start)

            if relative:
                err = abs((self.fd-np.mean(results.x[0]))/self.fd)
                #print('simulation %s/%s :\nfd estimate relative error: %s' %(j,N,err), end="\r")
                if err>10:
                    dc=1
                #    print('simulation %s/%s :\nfd estimate relative error: %s' %(j,N,err))
                #    print('AoAs:\n%s \nAoA condition 2:\n%s' %(self.paths['AoA'][1:],self.paths['AoA'][1:]-self.eta))
                #    print('fd='+str(self.fd))
                if npath_err:
                    npath_error.append(npath_err)
                f_d_error.append(err)
            else:
                f_d_error.append(abs(self.fd-np.mean(results.x[0])))
            if save_all:
                eta_abs_error.append(abs(np.rad2deg(self.eta)-np.rad2deg(np.mean(results.x[2]))))
                eta_rel_error.append(abs((self.eta-np.mean(results.x[2]))/self.eta))
                v_abs_error.append(abs(self.v_rx-np.mean(results.x[1])))
                v_rel_error.append(abs((self.v_rx-np.mean(results.x[1]))/self.v_rx))
        if save:
                if self.l==0.005:
                    np.save(path+'fd_k'+str(interval)+'_fc60_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',f_d_error)
                    if save_all:
                        np.save(path+'eta/eta_abs_k'+str(interval)+'_fc60_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',eta_abs_error)
                        np.save(path+'eta/eta_rel_k'+str(interval)+'_fc60_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',eta_rel_error)
                        np.save(path+'speed/v_abs_k'+str(interval)+'_fc60_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',v_abs_error)
                        np.save(path+'speed/v_rel_k'+str(interval)+'_fc60_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',v_rel_error)
                    if save_time:
                        np.save(path+'time/nlsTime_fc5_ns'+str(self.n_static)+'.npy',nls_time)
                elif self.l==0.0107:
                    np.save(path+'fd_k'+str(interval)+'_fc28_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',f_d_error)
                    if save_all:
                        np.save(path+'eta/eta_abs_k'+str(interval)+'_fc28_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',eta_abs_error)
                        np.save(path+'eta/eta_rel_k'+str(interval)+'_fc28_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',eta_rel_error)
                        np.save(path+'speed/v_abs_k'+str(interval)+'_fc28_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',v_abs_error)
                        np.save(path+'speed/v_rel_k'+str(interval)+'_fc28_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',v_rel_error)
                    if save_time:
                        np.save(path+'time/nlsTime_fc5_ns'+str(self.n_static)+'.npy',nls_time)
                else:
                    np.save(path+'fd_k'+str(interval)+'_fc5_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',f_d_error)
                    if save_all:
                        np.save(path+'eta/eta_abs_k'+str(interval)+'_fc5_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',eta_abs_error)
                        np.save(path+'eta/eta_rel_k'+str(interval)+'_fc5_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',eta_rel_error)
                        np.save(path+'speed/v_abs_k'+str(interval)+'_fc5_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',v_abs_error)
                        np.save(path+'speed/v_rel_k'+str(interval)+'_fc5_ns'+str(self.n_static)+'_snr'+str(self.SNR)+'.npy',v_rel_error)
                    if save_time:
                        np.save(path+'time/nlsTime_fc5_ns'+str(self.n_static)+'.npy',nls_time)
        if npath_error:
            if nls_time:
                return npath_error, nls_time
            else:
                npath_error
        return f_d_error
    

def varying_static_paths():
    interval = 16 # interval expressed in [ms]
    for n_static in [8]:
        for l in [0.0107, 0.06]:
            print('number of static paths: ' + str(n_static))
            print('wavelength: ' + str(l) + ' m')
            if l==0.0107:
                vmax = 10
                s = 50
            if l==0.06:
                vmax = 20
                s = 100
            if l==0.005:
                vmax = 5
                s = 20
            ch_sim = channel_sim(vmax=vmax, SNR=5, l=l, n_static=n_static)
            i = int(interval*1e-3/ch_sim.T)
            npath_error, nls_time = ch_sim.simulation(x_max=s, y_max=s, N=10000, interval=i, path='data/varying_n/', save=False)
            np.save('cir_estimation_sim/data/varying_npath/tot_5_fd_error_fc%s.npy'%(int(3e8/l*1e-9)),npath_error) 
            np.save('cir_estimation_sim/data/varying_npath/tot_5_nls_time_fc%s.npy'%(int(3e8/l*1e-9)),nls_time)
            print('average fd estimate relative error: ' + str(np.mean(npath_error, axis=0))+'\n')
            print('median fd estimate relative error: ' + str(np.median(npath_error,axis=0))+'\n')

def varying_snr():
    interval = 16 # interval expressed in [ms]
    for snr in [5]:
        for l in [0.0107,0.06]:
            for a in [5]:
                if l==0.0107:
                    vmax = 10
                    s = 50
                if l==0.06:
                    vmax=20
                    s = 100
                print('SNR: ' + str(snr) + ' dB')
                print('wavelength: ' + str(l) + ' m')
                print('AoA: ' + str(a) + '°')
                ch_sim = channel_sim(vmax=vmax,SNR=snr, l=l, AoAstd=a)
                i = int(interval*1e-3/ch_sim.T)
                fd_error = ch_sim.simulation(x_max=s, y_max=s, N=10000, interval=i, path='cir_estimation_sim/data/varying_snr/aoa' + str(a) + '/', save=False)
                print('average fd estimate relative error: ' + str(np.mean(fd_error))+'\n')
                print('median fd estimate relative error: ' + str(np.median(fd_error))+'\n')

def varying_interval():
    for interval in [2,4,8,16,32]:
        for l in [0.005,0.0107]:
            print('interval: ' + str(interval) + ' ms')
            print('wavelength: ' + str(l) + ' m')
            ch_sim = channel_sim(SNR=20, AoAstd=0, l=l)
            i = int(interval*1e-3/ch_sim.T)
            fd_error = ch_sim.simulation(x_max=10, y_max=10, N=10, interval=i, path='data/varying_interval/', save=False)
            print('average fd estimate relative error: ' + str(np.mean(fd_error))+'\n')
            print('median fd estimate relative error: ' + str(np.median(fd_error))+'\n')

def varying_T():
    for f in [28]:
        print('carrier frequency: ' + str(f) + ' GHz')
        tot_f_d_error =[]
        for ns in [2,4,6]:
            print('ns= ' + str(ns))
            path = 'cir_estimation_sim/plot/T/'
            f_d_error = []
            times = np.arange(0.18,0.72,0.02)
            for t in times:
                t  = round(t,2)
                print('period= ' + str(t) + ' ms')
                if f==60:
                    ### fc = 60 GHz ###
                    ch_sim = channel_sim(vmax=5, SNR=10, l=0.005, n_static=ns, T=t)
                if f==28:
                    ### fc = 28 GHz ### 
                    ch_sim = channel_sim(vmax=10, SNR=20, l=0.0107, AoAstd=0, n_static=ns, T=t)
                error = ch_sim.simulation(x_max=20,y_max=20,path=path+'fc_'+str(f)+'/'+str(ns)+'_static/',save=False,N=10000,interval=200)
                f_d_error.append(error)
                print('average fD relative error: ' + str(np.mean(error)) + ' std: ' + str(np.std(error)))
            #p = path+'fc_'+str(f)+'/varying_t'
            tot_f_d_error.append(np.squeeze(np.stack(f_d_error)))
        np.save(path+'fc_'+str(f)+'_'+str(ns)+'_static_snr0_f_d_error_nb.npy',tot_f_d_error)
        legend = ['2 static paths', '4 static paths', '6 static paths']
        ch_sim.plot_mae(tot_f_d_error,path,legend,times)


if __name__=='__main__':

    # vmax = 20
    # snr = 10
    # l = 0.06

    # ch_sim = channel_sim(vmax=vmax,SNR=snr, l=l, AoAstd=np.deg2rad(3))
    # fd_error = ch_sim.simulation(x_max=10, y_max=10, N=10000, interval=200, path='cir_estimation_sim/data/varying_snr/aoa3/', save=True)
    # print('average fd estimate relative error: ' + str(np.mean(fd_error))+'\n')
    # print('median fd estimate relative error: ' + str(np.median(fd_error))+'\n')
    
    varying_static_paths()
    # times = np.load('cir_estimation_sim/data/varying_npath/tot_5_nls_time_fc5.npy')
    # print('average computational times per No. static paths, fc = 5 ' + str(np.mean(times,0)))
    #times = np.load('cir_estimation_sim/data/varying_npath/tot_5_nls_time_fc28.npy')
    #print('average computational times per No. static paths, fc = 28 ' + str(np.mean(times,0)))
    # times = np.load('cir_estimation_sim/data/varying_npath/tot_5_nls_time_fc60.npy')
    # print('average computational times per No. static paths, fc = 60 ' + str(np.mean(times,0)))
    
    # ch_sim = channel_sim(vmax=10, SNR=snr, AoAstd=np.deg2rad(5), l=0.0107, static_rx=False)
    # fd_error = ch_sim.simulation(x_max=10, y_max=10, N=1000, interval=200, path='', save=False)
    # print('average fd estimate relative error: ' + str(np.mean(fd_error)))
    # print('median fd estimate relative error: ' + str(np.median(fd_error))+'\n')
    # ch_sim = channel_sim(vmax=20, SNR=snr, AoAstd=np.deg2rad(5), l=0.06, static_rx=False)
    # fd_error = ch_sim.simulation(x_max=10, y_max=10, N=1000, interval=200, path='', save=False)
    # print('average fd estimate relative error: ' + str(np.mean(fd_error)))
    # print('median fd estimate relative error: ' + str(np.median(fd_error))+'\n')
    # ch_sim = channel_sim(vmax=5, SNR=20, AoAstd=np.deg2rad(5), l=0.005, static_rx=True)
    # fd_error = ch_sim.simulation(x_max=10, y_max=10, N=100, interval=100, path='', save=False)
    # print('average fd estimate relative error: ' + str(np.mean(fd_error)))
    # print('median fd estimate relative error: ' + str(np.median(fd_error))+'\n')