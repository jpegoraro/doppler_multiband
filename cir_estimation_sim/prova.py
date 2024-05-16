import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import time
class channel_sim():

    def __init__(self,T=0.08e-3,SNR=20,AoAstd=np.deg2rad(5),G_tx=1,G_rx=1,P_tx=1,l=0.005,n_static=2,B=1.76e9,us=16,vmax=5):
        """
            v_rx: receiver speed vector (2 components)[m/s]
            fd: Doppler shift caused by target movement [Hz]
            T: interpacket time (cir samples period) [s]
            SNR: signal to noise ratio [dB]
            AoAstd: std of the noise added to the angles of arrivals measurements [rad]
            G_tx/G_rx: transmitter/receiver antenna gain
            P_tx: transmitted power
            l: wavelength [m] 
            n_static: number of static paths
            B: bandwidth [Hz]
            us: up sampling rate (t' = t / us)
            vmax: maximum receiver speed [m/s]
            tx/rx: transmitter/receiver Cartesian coordinates(default [0,0]/[x_max,y_max]) [m]
        """
        self.vrx = None # speed vector
        self.v_rx = None # speed modulus
        self.eta = None # speed direction w.r.t. LoS
        self.fd = None
        self.f_off = 0
        self.vmax = vmax
        
        self.SNR = SNR # dB
        self.AoAstd = AoAstd
        self.G_tx = G_tx
        self.G_rx = G_rx
        self.P_tx = P_tx
        self.l = l
        self.n_static = n_static
        self.B = B
        self.T = T # interpacket time (cir samples period)
        self.us = us

        self.trn_field = None
        self.cir = None
        self.rx_signal = None
        self.k = 0 # dicrete time index

        self.beta = np.zeros(n_static+1)
        self.positions = np.zeros((self.n_static+3,2)) # positions coordinates [rx,tx,t,s1,...,sn_static]
        self.paths = np.zeros((n_static+2,4)) # [delay, phase, attenuations, AoA] for each path [LoS,t,s1,...,sn_static]
        self.phases = np.zeros((n_static+2,2)) # estimated phases at time k-1 and k
        
        #self.h_rrc = rrcosfilter(129, alpha=0.8, Ts=1/(self.B), Fs=us*self.B)[1] 
        #self.h_rrc = self.rrcos(129,self.us,1)
        n = int(1e-3/self.T) # number of samples in 1 ms
        fo_max = 3e8/(self.l*10e6) # 0.1 ppm of the carrier frequency 
        self.std_w = fo_max/(3*np.sqrt(n**3)) # std for the fo random walk s.t. its max drift in 1 ms is fo_max        

    def dist(self, p1, p2):
        """
            return distance between two points in 2D.
            p1,p2: [x,y] Cartesian coordinates
        """
        return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**(.5)
    
    def get_positions(self,x_max,y_max,res_min=1,dist_min=1,plot=False):
            """
                generate random positions for rx, tx, n_static objects in 2D.
                x_max: maximum x coordinate [m]
                y_max: maximum y coordinate [m]
                res_min: minimum distance resolution [m]
                dist_min: minimum distance between two reflector/scatterer [m]
                returns an array of positions coordinates [rx,tx,t,s1,s2]
            """
            self.vrx = np.random.uniform(0.5,self.vmax,2) # speed vector
            self.v_rx = (self.vrx[0]**2+self.vrx[1]**2)**(0.5) # speed modulus
            self.fd = np.random.uniform(200,2000)
            if np.random.rand()>0.5:
                self.fd = - self.fd
            beta_max = 2*np.arccos(3e8/(2*self.B*res_min))
            self.positions[0,:] = [x_max,y_max]
            self.positions[1,:] = [0,0]

            self.paths[0,0] = self.dist(self.positions[0,:],self.positions[1,:])/3e8 # LoS delay
            for i in range(2,self.n_static+3):
                x = np.random.uniform(0,x_max)
                y = np.random.uniform(0,y_max)
                beta = np.pi
                start = time.time()
                while True:
                    if time.time()-start>2:
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
                    if beta<beta_max:
                        self.beta[i-2] = beta
                        self.positions[i,:] = x,y
                        # path delay for non-LoS
                        self.paths[i-1,0] = (self.dist(self.positions[1,:],self.positions[i,:])+self.dist(self.positions[i,:],self.positions[0,:]))/3e8
                        check = []
                        for j in range(0,i):
                            for k in range(0,i):
                                if k!=j:
                                    check.append(abs(self.paths[j,0]-self.paths[k,0])>1/self.B) # check all path are separable
                        if all(check):
                            check = []
                            self.compute_AoA(ind=i-1)
                            for k,j in combinations(range(i),2):
                                check.append(abs(self.paths[k,3]-self.paths[j,3])>0.05) # AoAs must be different between them
                            if all(check):
                                break
                    x = np.random.uniform(0,x_max)
                    y = np.random.uniform(0,y_max)
            #self.compute_AoAs()
            #while True:
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
            # if all(abs(self.paths[1:,3]-(2*self.eta))>0.1): # check second existence condition (AoA!=2eta) 
            #     break
            # self.v_rx = np.random.uniform(0.5,self.vmax)
            # self.eta = np.random.uniform(0,2*np.pi)
            # self.vrx = [self.v_rx*np.cos(self.eta),self.v_rx*np.sin(self.eta)]
            if plot:
                self.plot_pos() 

    def compute_AoA(self,ind):
        """
            compute angles of arrival for each path.
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
        self.paths[ind,3] = np.arccos(c/ip)

    def plot_pos(self):
        """
            plots the environmental disposition.
        """
        #self.compute_AoAs()
        print('AoA : LoS, t, s1, s2, ... \n' + str(np.rad2deg(self.paths[:,3])))
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

ch_sim = channel_sim(n_static=10)
for i in range(1000):
    ch_sim.get_positions(10,10,plot=False)
    print(i)