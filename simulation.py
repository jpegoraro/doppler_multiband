import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def simplify(angle):
    while True:
        if angle>2*np.pi:
            angle = angle-2*np.pi
        if angle<0:
            angle = 2*np.pi+angle
        if 0<angle<2*np.pi:
            return angle

def solve_system(noise, zeta_std, phase_std, zetas, phases, T=0.57*10**(-9), l=0.005):
    if noise:
        n_zetas = zetas + np.random.normal(0,zeta_std,3)
        n_phases = phases + np.random.normal(0,phase_std,4)
    else:
        n_zetas = zetas
        n_phases = phases         
    alpha = (n_phases[3]-n_phases[1])*np.cos(n_zetas[1])
    beta = (n_phases[3]-n_phases[1])*np.sin(n_zetas[1])
    delta = n_phases[2]-n_phases[3]
    gamma = (n_phases[1]-n_phases[2])*np.cos(n_zetas[2])
    epsilon = (n_phases[1]-n_phases[2])*np.sin(n_zetas[2])
    #pol = np.polynomial.polynomial.Polynomial([-(alpha+delta+gamma),2*(beta+epsilon),(alpha+delta+gamma)])
    #t = pol.roots()
    t = np.roots([-(alpha+delta+gamma),2*(beta+epsilon),(alpha+delta+gamma)])
    #t = np.zeros(2)
    f_d = np.zeros((2))
    f_off = np.zeros(4)
    v = np.zeros(2)
    #t[0] = (-2*(beta+epsilon) + np.sqrt((2*(beta+epsilon))**2+4*(alpha+delta+gamma)**2))/(-2*(alpha+delta+gamma))
    #t[1] = (-2*(beta+epsilon) - np.sqrt((2*(beta+epsilon))**2+4*(alpha+delta+gamma)**2))/(-2*(alpha+delta+gamma))
    eta = 2*np.arctan(t)
    f_d[0] = (n_phases[0]-n_phases[1]-(((n_phases[3]-n_phases[1])*(np.cos(n_zetas[0]-eta[0])-np.cos(eta[0])))/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0]))))/(2*np.pi*T)
    f_d[1] = (n_phases[0]-n_phases[1]-(((n_phases[3]-n_phases[1])*(np.cos(n_zetas[0]-eta[1])-np.cos(eta[1])))/(np.cos(n_zetas[2]-eta[1])-np.cos(eta[1]))))/(2*np.pi*T)
    v[0] = l*(n_phases[3]-n_phases[1])/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0]))
    v[1] = l*(n_phases[3]-n_phases[1])/(np.cos(n_zetas[2]-eta[1])-np.cos(eta[1]))
    f_off[0] = n_phases[1]-(v[0]/l*np.cos(eta[0]))
    f_off[1] = n_phases[1]-(v[0]/l*np.cos(eta[1]))
    f_off[2] = n_phases[1]-(v[1]/l*np.cos(eta[0]))
    f_off[3] = n_phases[1]-(v[1]/l*np.cos(eta[1]))
    return eta, f_d, v, f_off, alpha, delta, gamma, n_zetas

def check_phases(eta, f_d, v, f_off, zetas, l=0.005):
    c_phases = np.zeros(4)
    c_phases[0] = simplify(f_d+f_off+(v/l*np.cos(zetas[0]-eta[0])))
    c_phases[1] = simplify(f_off+(v/l*np.cos(eta[0])))
    c_phases[2] = simplify(f_off+(v/l*np.cos(zetas[1]-eta[0])))
    c_phases[3] = simplify(f_off+(v/l*np.cos(zetas[2]-eta[0])))
    return c_phases

def boxplot_plot(errors, xlabel, ylabel, xticks, title, name=''):
    plt.figure(figsize=(12,8))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    ax = sns.boxplot(data=errors, orient='v', palette='rocket', showfliers=False)
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.title(title)
    plt.savefig('plots/'+name+'.png')
    #plt.show()

zetas = np.random.uniform(-np.pi/4,np.pi/4,2)
phases = np.random.uniform(0,2*np.pi,5)


#for i in range(10000):
zetas = np.random.uniform(-np.pi/4,np.pi/4,3)
phases = np.random.uniform(0,2*np.pi,4)
eta_n, f_d_n, v_n, f_off_n, alpha_n, delta_n, gamma_n, n_zetas = solve_system(False, 0, 0, zetas, phases)
print(f_d_n)
print(v_n)
print(f_off_n)



# c_phases = check_phases(eta_n, f_d_n[0], v_n[0], f_off_n[0], zetas)
# for j in range(4):
#     if c_phases[j]-phases[j] > 10**(-10):
#         print(c_phases[j]-phases[j])

SNR = np.array([-10,-5,0,5,10,15,20])
SNR = np.power(10,SNR/10)
p_std = np.sqrt(1/(2*256*SNR))
zetas_std = [1,3,5,7,10]
threshold = 0.05
for zeta_std in np.deg2rad(zetas_std):
    tot_eta_errors = []
    tot_fd_errors = []    
    for phase_std,snr in zip(p_std,SNR):
        N = 10000 # number of simulations
        eta_errors = []
        f_d_errors = []
        f_off_errors = []
        for ind in range(N):
            zetas = np.random.uniform(0,np.pi/4,3)
            phases = np.random.uniform(0,2*np.pi,4)
            eta_n, f_d_n, v_n, f_off_n, alpha_n, delta_n, gamma_n, n_zetas = solve_system(True, zeta_std, phase_std, zetas, phases) # with added Gaussian noise
            eta, f_d, v, f_off, alpha, delta, gamma, zetas = solve_system(False, zeta_std, phase_std, zetas, phases) # without noise, exact solution
            for i in range(len(n_zetas)):
                n_zetas[i] = simplify(n_zetas[i])
            for i in range(len(eta_n)):
                eta_n[i] = simplify(eta_n[i])
                eta[i] = simplify(eta[i])
            skip = False
            if abs(alpha_n+delta_n+gamma_n) < threshold:
                skip = True
                #eta = [np.pi/2,-np.pi/2]
            for i in range(3):
                for j in range(2):
                    if abs(n_zetas[i]-(2*eta_n[j]))<threshold:
                        skip = True
                    if abs(eta_n[j])<threshold:
                        skip = True
                    if abs(n_zetas[i])<threshold:
                        skip=True
            if skip:
                continue
            eta = eta[0]
            eta_n = eta_n[0]
            f_d = f_d[0]
            f_d_n = f_d_n[0]
            eta_errors.append(np.rad2deg(eta_n-eta))
            f_d_errors.append(f_d_n-f_d)
            f_off_errors.append(f_off_n-f_off)
        tot_eta_errors.append(eta_errors)
        tot_fd_errors.append(f_d_errors)
        print('zeta std: ' + str(np.rad2deg(zeta_std)) + '°')
        print('phase_std: ' + str(np.rad2deg(phase_std)) + '°')
        print('SNR: ' + str(10*np.log10(snr)) + ' dB')
        print('eta errors mean: ' + str(np.mean(eta_errors)) + ', std: ' + str(np.std(eta_errors)))
        print('f_d errors mean: ' + str(np.mean(f_d_errors)) + ', std: ' + str(np.std(f_d_errors)))
        print('f_off errors mean: ' + str(np.mean(f_off_errors)) + ', std: ' + str(np.std(f_off_errors)))
        print('\n')
    boxplot_plot(tot_eta_errors, "SNR (dB)", "eta errors (°)", 10*np.log10(SNR), "eta errors with zeta std = " + str(np.rad2deg(zeta_std)) + "° and threshold= " + str(threshold), 'eta_errors_zeta_std' + str(np.rad2deg(zeta_std)))
    boxplot_plot(tot_fd_errors, "SNR (dB)", "frequency Doppler errors (Hz)", 10*np.log10(SNR), "frequency Doppler errors with zeta std = " + str(np.rad2deg(zeta_std)) + "° and threshold= " + str(threshold), 'fd_errors_zeta_std' + str(np.rad2deg(zeta_std)))