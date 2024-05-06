import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import scipy
from commpy.filters import rrcosfilter
from scipy.signal import correlate, correlation_lags

class channel_sim():

    def __init__(self,v_rx,fd,SNR=50,G_tx=1,G_rx=1,P_tx=1,l=0.005,n_static=2,B=1.76e9,os=16,tx=None,rx=None):
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
        self.v_rx = (v_rx[0]**2+v_rx[1]**2)**(0.5) # speed modulus
        self.eta = np.arctan(v_rx[1]/v_rx[0]) # speed direction w.r.t. positive x-axis
        self.fd = fd
        self.SNR = SNR # dB
        self.G_tx = G_tx
        self.G_rx = G_rx
        self.P_tx = P_tx
        self.l = l
        self.n_static = n_static
        self.B = B
        self.os = os
        self.tx = tx
        self.rx = rx
        self.trn_field = None
        self.cir = None
        self.rx_signal = None
        self.beta = np.zeros(n_static+1)
        self.positions = np.zeros((self.n_static+3,2)) # positions coordinates [rx,tx,t,s1,...,sn_static]
        self.paths = np.zeros((n_static+2,4)) # [delay, phase, attenuations, AoA] for each path [LoS,t,s1,...,sn_static]
        self.h_rrc = rrcosfilter(129, alpha=1, Ts=1/(self.B), Fs=os*self.B)[1] 
        n = int(1e-3*self.B) # number of samples in 1 ms
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
        if plot:
            self.plot_pos()     

    def plot_pos(self):
        """
            plots the environmental disposition.
        """
        print('AoA : LoS, t, s1, s2, ... \n' + str(np.rad2deg(self.paths[:,3])))
        plt.plot(self.positions[0,0],self.positions[0,1],'ro',label='rx')
        plt.plot(self.positions[1,0],self.positions[1,1],'go',label='tx')
        plt.plot(self.positions[2,0],self.positions[2,1],'r+',label='target')
        plt.plot(self.positions[3:,0],self.positions[3:,1],'bo',label='static objects')
        plt.plot(self.positions[0:2,0],self.positions[0:2,1],label='LoS')
        for i in range(2,self.n_static+3):
            plt.plot([self.positions[1,0],self.positions[i,0],self.positions[0,0]],[self.positions[1,1],self.positions[i,1],self.positions[0,1]])
        plt.legend()
        print('beta angles: t, s1, s2, ... \n' + str(np.rad2deg(self.beta)))
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
        m = (self.positions[0,1]-self.positions[1,1])/(self.positions[0,0]-self.positions[1,0])
        q = self.positions[0,1]-m*self.positions[0,0]
        m1 = -1/m
        for i in range(1,len(self.paths)):
            x,y = self.positions[i+1,:]
            q1 = y-m1*x
            x_p = (q-q1)/(m1-m)
            y_p = m*x_p+q
            c = self.dist([x_p,y_p],self.positions[0,:])
            ip = self.dist([x,y],self.positions[0,:])
            assert int(ip**2)==int(c**2+((x-x_p)**2+(y-y_p)**2))
            self.paths[i,3] = np.arccos(c/ip)
            
    def compute_phases(self):
        """
            compute phases for each path.
            LoS phase initial offset=0.
        """
        self.paths[0,1] = self.v_rx/self.l*np.cos(self.eta) # LoS
        self.paths[1,1] = self.fd + self.v_rx/self.l*np.cos(self.paths[1,3]-self.eta) + np.random.uniform(0,2*np.pi) # target
        for i in range(2,len(self.paths)):
            self.paths[i,1] = self.v_rx/self.l*np.cos(self.paths[i,3]-self.eta) + np.random.uniform(0,2*np.pi) # static
    
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
        if self.trn_field==None:
            self.load_trn_field()
        up_cir = np.zeros(256*16).astype(complex)
        cir = np.zeros(256).astype(complex)
        t = 1/(self.B*(self.os))
        assert all(self.paths[:,0]==self.get_delays())
        self.compute_AoAs()
        self.compute_phases()
        self.compute_attenuations()
        delays = np.floor(self.paths[:,0]*self.B*self.os)
        delays_1 = np.floor(self.paths[:,0]*self.B)
        for i,(d,d1) in enumerate(zip(delays.astype(int),delays_1.astype(int))):
            up_cir[d] = up_cir[d] + self.paths[i,2] * np.exp(1j * 2 * np.pi * self.paths[i,1])
            cir[d1] = cir[d1] + self.paths[i,2] * np.exp(1j * 2 * np.pi * self.paths[i,1])
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
        rx_signal += noise
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
        self.f_off = np.random.normal(0,self.std_w)
        return h_est * np.exp(1j*2*np.pi*self.f_off)       
        
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
        h_128 = self.add_cfo(h_128)

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
        ind = np.floor(self.paths[:,0]*self.B).astype(int) # from the paths delay
        phases = np.angle(h[ind])
        if plot:
            t = np.zeros(len(h))
            t[ind] = np.abs(h[ind])
            plt.stem(np.abs(h_128)/max(abs(h_128)), linefmt='r', markerfmt='rD', label='estimate')
            plt.stem(np.abs(ch_sim.cir)/max(abs(ch_sim.cir)), label='real')
            plt.stem(t/max(t), markerfmt='gD', label='selected paths')
            plt.grid()
            plt.legend()
            plt.show()
        return phases 


if __name__=='__main__':

    ch_sim = channel_sim(v_rx=[2,1],fd=500)
   
    ch_sim.get_positions(5,5, plot=False)

    ch_sim.compute_cir(plot=False)

    up_rx_signal = ch_sim.get_rxsignal(plot=False)

    ch_sim.sampling(up_rx_signal, plot=False)

    h_128 = ch_sim.estimate_CIR(ch_sim.rx_signal, plot=False)

    phases = ch_sim.get_phases(h_128, plot=True)
