from scipy.signal import correlate, correlation_lags, deconvolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import os
sys.path.insert(0,os.getcwd())
from cir_estimation_sim.ray_tracing import RayTracing

class ChannelModel(RayTracing):
    
    def __init__(self, l, n_static, snr):
        super().__init__(l=l, n_static=n_static)
        self.SNR = snr
        self.rx_signal = None
        if self.l==0.005:
            self.h_rrc = self.rrcos(129,self.us,1)
        self.f_off = 0
        n = int(1e-3/self.T) # number of samples in 1 ms
        fo_max = 3e8/(self.l*10e6) # 0.1 ppm of the carrier frequency 
        self.std_w = fo_max/(6*np.pi*self.T*(2*n**2+1)**0.5) #--->  std for the fo random walk s.t. its max drift in 1 ms is fo_max #  fo_max/(6*np.pi*self.T*(n+1))# std for independent samples of the cfo s.t. its max drift in 1 ms is fo_max

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
    
    def add_cfo(self, signal, k):
        """
            Returns the signal after adding the channel frequency offset shift.
            signal: signal (cir estimate).
        """
        self.f_off += np.random.normal(0,self.std_w)
        return signal * np.exp(1j*2*np.pi*self.f_off*k*self.T)       
    
    def add_po(self, signal):
        """
            Returns the signal after adding the phase noise shift.
            signal: signal (cir estimate).
        """
        return signal * np.exp(1j*np.random.normal(0,self.std_w))
    
    def get_cir_est(self, init, k):
        self.get_positions(self.x_max,self.y_max,plot=False)
        self.compute_cir(init=init, k=k, plot=False)
        if self.l==0.005:
            up_rx_signal = self.get_rxsignal(plot=False)
            self.sampling(up_rx_signal,plot=False)
            h = self.estimate_CIR(self.rx_signal,plot=False)
        else:
            Y = self.get_rx_ofdm(self.tx_signal)
            h = self.estimate_ofdm_CIR(Y, plot=False)
        ### add cfo ###
        h = self.add_po(self.add_cfo(h, k=k))
        return h

if __name__=='__main__':
    ch = ChannelModel(l=0.005, n_static=2, snr=10)
    ch.get_cir_est(init=True, k=1)