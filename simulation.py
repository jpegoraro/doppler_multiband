import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
from MeanEstimator import MeanEstimator
import tikzplotlib as tik

def solve_system(noise, zeta_std, phase_std, zetas, phases, new=True, T=0.27*10**(-3), l=0.005):
    """
        noise: wether to add noise to phases and angle of arrivals;
        zeta_std: standard deviation for the noise variable regarding the angle of arrivals;
        phase_std: standard deviation for the noise variable regarding the measured phase;
        zetas: angle of arrivals;
        phases: measured phases;
        new: if True use the new resolution method
        T: sampling period;
        l: wavelength.
        Solve the system for the received input parameters and returns the variables of the system.
    """
    if noise:
        n_zetas = zetas + np.random.normal(0,zeta_std,3)
        n_phases = phases + np.random.normal(0,phase_std,4)
    else:
        n_zetas = zetas
        n_phases = phases         
    alpha = (n_phases[1]-n_phases[3])*np.cos(n_zetas[2])
    beta = (n_phases[1]-n_phases[3])*np.sin(n_zetas[2])
    delta = n_phases[2]-n_phases[1]
    gamma = (n_phases[3]-n_phases[2])*np.cos(n_zetas[1])
    epsilon = (n_phases[3]-n_phases[2])*np.sin(n_zetas[1])
    if new:
        if abs(beta+epsilon)<1e-5==0:
            eta = np.pi/2 - np.arctan(-(beta+epsilon)/(alpha+delta+gamma))
            print('wow')
        else:
            eta = np.arctan(-(alpha+delta+gamma)/(beta+epsilon))
        A = (np.sin(n_zetas[1])*(1-np.cos(n_zetas[2]))) + (np.sin(n_zetas[2])*(np.cos(n_zetas[1])-1))
        if not (beta+epsilon)/A>0:
            eta = eta + np.pi
        if eta<0:
            eta = eta + (2*np.pi)
        f_d = (n_phases[0]-n_phases[3]-(((n_phases[2]-n_phases[3])*(np.cos(n_zetas[0]-eta)-np.cos(eta)))/(np.cos(n_zetas[2]-eta)-np.cos(eta))))/(2*np.pi*T)
        v = l/(2*np.pi*T)*(n_phases[2]-n_phases[3])/(np.cos(n_zetas[2]-eta)-np.cos(eta))
        f_off = n_phases[3]/(2*np.pi*T)-((n_phases[2]-n_phases[3])*np.cos(eta)/(np.cos(n_zetas[2]-eta)-np.cos(eta)))
        return eta, f_d, v, f_off, alpha, delta, gamma, n_zetas
    else:
        t = np.roots([-(alpha+delta+gamma),2*(beta+epsilon),(alpha+delta+gamma)])
        f_d = np.zeros(2)
        v = np.zeros(2)
        eta = 2*np.arctan(t)
        f_d[0] = (n_phases[0]-n_phases[3]-(((n_phases[2]-n_phases[3])*(np.cos(n_zetas[0]-eta[0])-np.cos(eta[0])))/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0]))))/(2*np.pi*T)
        f_d[1] = (n_phases[0]-n_phases[3]-(((n_phases[2]-n_phases[3])*(np.cos(n_zetas[0]-eta[1])-np.cos(eta[1])))/(np.cos(n_zetas[2]-eta[1])-np.cos(eta[1]))))/(2*np.pi*T)
        v[0] = l/(2*np.pi*T)*(n_phases[2]-n_phases[3])/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0]))
        v[1] = l*(n_phases[2]-n_phases[3])/((np.cos(n_zetas[2]-eta[1])-np.cos(eta[1]))*2*np.pi*T)
        f_off = n_phases[3]/(2*np.pi*T)-((n_phases[2]-n_phases[3])*np.cos(eta[0])/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0])))
        return eta[1], f_d[0], v[1], f_off, alpha, delta, gamma, n_zetas

def get_phases(eta, f_d, v, f_off, zetas, T=0.27*10**(-3), l=0.005, k=1):
    """
        Solve the system for the received input parameters and returned the computed phases.
    """
    c_phases = np.zeros(4)
    c_phases[0] = (2*np.pi*T*(f_d+(v/l*np.cos(zetas[0]-eta))+(k*f_off[0]-(k-1)*f_off[1])))#%np.pi
    c_phases[1] = (2*np.pi*T*(v/l*np.cos(zetas[1]-eta)+(k*f_off[0]-(k-1)*f_off[1])))#%np.pi
    c_phases[2] = (2*np.pi*T*(v/l*np.cos(zetas[2]-eta)+(k*f_off[0]-(k-1)*f_off[1])))#%np.pi
    c_phases[3] = (2*np.pi*T*(v/l*np.cos(eta)+(k*f_off[0]-(k-1)*f_off[1])))#%np.pi
    return c_phases

def boxplot_plot(path, errors, xlabel, ylabel, xticks, title, name=''):
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

def check_system():
    for i in range(10000):
        zetas = np.random.uniform(-np.pi/4,np.pi/4,3)
        phases = np.random.uniform(0,2*np.pi,4)
        eta_n, f_d_n, v_n, f_off_n, alpha_n, delta_n, gamma_n, n_zetas = solve_system(False, 0, 0, zetas, phases)
        
        c_phases = get_phases(eta_n[0], f_d_n[0], v_n[0], f_off_n[0], zetas)
        for j in range(4):
            if c_phases[j]-phases[j] > 10**(-10):
                print(c_phases[j]-phases[j])

def plot_ransac(x,y_real,y_noisy,y_pred,mask):
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

def get_input(k=1):
    """
    k=1 represents the discrete time variable, we assume it equal to 1, 
    but it is not involved in the computation regarding the Doppler frequency.
    
    Returns realistic values of: \n measured phases, angle of arrivals, and variables. 
    """
    f_d = np.random.uniform(-1000,1000)
    f_off = np.random.normal(100000,10000,2)
    v = np.random.uniform(0,5)
    count = 0
    while True:
        check = True
        zetas = np.random.uniform(-np.pi/4,np.pi/4,3)
        eta = np.random.uniform(0,2*np.pi)
        phases = get_phases(eta, f_d, v, f_off, zetas, k=k)
        alpha = (phases[1]-phases[3])*np.cos(zetas[2])
        delta = phases[2]-phases[1]
        gamma = (phases[3]-phases[2])*np.cos(zetas[1])
        # if abs(alpha+delta+gamma)<1e-5:
        #     check = False
        if check or count>100000:
            break
        # count += 1
    return phases, zetas, eta, f_d, v, k*f_off[0]-(k-1)*f_off[1]

def simulation(path, relative, only_fD=True):
    SNR = 10#np.array([0,5,10,15])
    N = 10000 # number of simulations
    interval = 100 # number of samples in which variables can be considered constant 
    SNR = np.power(10,SNR/10)
    p_std = np.sqrt(1/(2*256*SNR))
    zeta_std = 3#[1,3,5]
    mean_estimator = MeanEstimator()
    ransac = RANSACRegressor(estimator=mean_estimator, min_samples=10, max_trials=400)
    #ransac_fd = RANSACRegressor(estimator=mean_estimator, min_samples=5,  max_trials=400)
    #for z_std in np.deg2rad(zeta_std):
    z_std = np.deg2rad(zeta_std)
    tot_eta_error = []
    tot_f_d_error = []
    tot_v_error = []
    #print('zeta std: ' + str(round(np.rad2deg(z_std))))
    #for phase_std,snr in zip(p_std,SNR):
    phase_std = p_std
    print('phase std: ' + str(p_std))
    for interval in range(20,120,20):
        eta_error = []
        f_d_error = []
        v_error = []
        for i in tqdm(range(N)):
            phases,zetas,eta,f_d,v,f_off = get_input()
            etas_n = []
            f_ds_n = []
            vs_n = []
            f_offs_n = []
            for i in range(interval):
                eta_n, f_d_n, v_n, f_off_n, alpha_n, delta_n, gamma_n, n_zetas= solve_system(True, z_std, phase_std, zetas, phases, new=True)
                etas_n.append(eta_n)
                f_ds_n.append(f_d_n)
                vs_n.append(v_n)
                f_offs_n.append(f_off_n)
            ### RANSAC ###
            ransac_fd = RANSACRegressor(estimator=mean_estimator, min_samples=int(interval/2))
            time = np.arange(len(f_ds_n)).reshape(-1,1)
            # Doppler frequency
            try:
                ransac_fd.fit(time,f_ds_n)
                pred_f_d = ransac_fd.predict(time)
            except:
                print('ransac exception fD')
                pred_f_d = np.mean(f_ds_n)
            #plot_ransac(time,np.ones(len(time))*f_d,f_ds_n,pred_f_d,ransac.inlier_mask_)
            if relative:
                f_d_error.append(np.abs((f_d-np.mean(pred_f_d))/f_d))
            else:
                f_d_error.append(np.abs(f_d-np.mean(pred_f_d)))
            if not only_fD:
                # eta
                etas_n = np.mod(etas_n,2*np.pi)
                try:
                    ransac.fit(time,etas_n)
                    pred_eta = ransac.predict(time)
                except:
                    print('ransac exception eta')
                    pred_eta = np.mean(etas_n)
                if relative:
                    eta_error.append(np.abs((np.rad2deg(eta)-np.mean(np.rad2deg(pred_eta)))/np.rad2deg(eta)))
                else:
                    eta_error.append(np.abs((np.rad2deg(eta)-np.mean(np.rad2deg(pred_eta)))))
                # speed
                try:
                    ransac.fit(time, vs_n)
                    pred_v = ransac.predict(time)
                except:
                    print('ransac exception speed')
                    pred_v = np.mean(vs_n)
                if relative:
                    v_error.append(np.abs((v-np.mean(pred_v))/v))
                else:
                    v_error.append(np.abs((v-np.mean(pred_v))))
            ##############
            
        if not only_fD:
            tot_eta_error.append(eta_error)
            tot_v_error.append(v_error)
        tot_f_d_error.append(f_d_error)
    
    if relative:    
        #boxplot_plot(path, tot_eta_error, "SNR (dB)", "relative eta errors", 10*np.log10(SNR), "eta errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'eta_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
        boxplot_plot(path, tot_f_d_error, "SNR (dB)", "relative frequency Doppler errors", np.arange(20,120,20), "frequency Doppler errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'fd_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
        #boxplot_plot(path, tot_v_error, "SNR (dB)", "relative speed error", 10*np.log10(SNR), "speed errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'speed_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
    else:
        boxplot_plot(path, tot_eta_error, "SNR (dB)", "eta errors (°)", 10*np.log10(SNR), "eta errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'eta_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
        boxplot_plot(path, tot_f_d_error, "SNR (dB)", "frequency Doppler errors (Hz)", 10*np.log10(SNR), "frequency Doppler errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'fd_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))
        boxplot_plot(path, tot_v_error, "SNR (dB)", "speed error (m/s)", 10*np.log10(SNR), "speed errors with zeta std = " + str(round(np.rad2deg(z_std))) + "°", 'speed_errors_zeta_std' + str(np.round(np.rad2deg(z_std),1)))

if __name__=='__main__':
    path = 'plots/new_solution/ransac_1dim/varying_k/'
    simulation(path,True)
    