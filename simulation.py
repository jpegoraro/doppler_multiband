import numpy as np

def simplify(angle):
    while True:
        if angle>2*np.pi:
            angle = angle-2*np.pi
        if angle<0:
            angle = 2*np.pi+angle
        if 0<angle<2*np.pi:
            return angle

def solve_system(noise, zeta_std, phase_std, zetas, phases, l=0.005):
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
    t = np.zeros(2)
    t[0] = (-2*(beta+epsilon) + np.sqrt((2*(beta+epsilon))**2+4*(alpha+delta+gamma)**2))/(-2*(alpha+delta+gamma))
    t[1] = (-2*(beta+epsilon) - np.sqrt((2*(beta+epsilon))**2+4*(alpha+delta+gamma)**2))/(-2*(alpha+delta+gamma))
    eta = 2*np.arctan(t)
    f_d = n_phases[0]-n_phases[1]-(((n_phases[3]-n_phases[1])*(np.cos(n_zetas[0]-eta[0])-np.cos(eta[0])))/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0])))
    v = l*(n_phases[3]-n_phases[1])/(np.cos(n_zetas[2]-eta[0])-np.cos(eta[0]))
    f_off = n_phases[1]-(v/l*np.cos(eta[0]))
    return eta, f_d, v, f_off, alpha, delta, gamma, n_zetas

def check_phases(eta, f_d, v, f_off, phases, zetas, l=0.005):
    c_phases = np.zeros(4)
    c_phases[0] = simplify(f_d[0]+f_off[0]+(v[0]/l*np.cos(zetas[0]-eta[0])))
    c_phases[1] = simplify(f_off[0]+(v[0]/l*np.cos(eta[0])))
    c_phases[2] = simplify(f_off[0]+(v[0]/l*np.cos(zetas[1]-eta[0])))
    c_phases[3] = simplify(f_off[0]+(v[0]/l*np.cos(zetas[2]-eta[0])))
    return c_phases

snr = np.array([-10,-5,0,5,10,15,20])
snr = np.power(10,snr/10)
p_std = np.sqrt(1/(2*256*snr))
for phase_std,snr in zip(p_std,snr):
    for zeta_std in np.deg2rad([1,2,3,4,5]):
        N = 10000 # number of simulations
        eta_errors = []
        f_d_errors = []
        f_off_errors = []
        for ind in range(N):
            zetas = np.random.uniform(0,np.pi/4,3)
            phases = np.random.uniform(0,2*np.pi,4)
            eta_n, f_d_n, v_n, f_off_n, alpha_n, delta_n, gamma_n, n_zetas = solve_system(True, zeta_std, phase_std, zetas, phases) # with added Gaussian noise
            eta, f_d, v, f_off, alpha, delta, gamma, zetas = solve_system(False, zeta_std, phase_std,    zetas, phases) # without noise, exact solution
            for i in range(len(n_zetas)):
                n_zetas[i] = simplify(n_zetas[i])
            for i in range(len(eta_n)):
                eta_n[i] = simplify(eta_n[i])
            if abs(alpha_n+delta_n+gamma_n) < 0.5:
                continue
            skip = False
            for i in range(3):
                for j in range(2):
                    if abs(n_zetas[i]-(2*eta_n[j]))<0.5:
                        skip = True
                    if abs(eta_n[j])<0.05:
                        skip = True
                    if abs(n_zetas[i])<0.05:
                        skip=True
            if skip:
                continue
            eta_errors.append(eta_n-eta)
            f_d_errors.append(f_d_n-f_d)
            f_off_errors.append(f_off_n-f_off)
        print('zeta std: ' + str(np.rad2deg(zeta_std)) + '°')
        print('phase_std: ' + str(np.rad2deg(phase_std)) + '°')
        print('SNR: ' + str(10*np.log10(snr)) + ' dB')
        print('eta errors mean: ' + str(np.mean(eta_errors)) + ', std: ' + str(np.std(eta_errors)))
        print('f_d errors mean: ' + str(np.mean(f_d_errors)) + ', std: ' + str(np.std(f_d_errors)))
        print('f_off errors mean: ' + str(np.mean(f_off_errors)) + ', std: ' + str(np.std(f_off_errors)))
        print('\n')