import numpy as np
from simulation import get_phases

def get_input(k=1):
    """
    k=1 it represents the discrete time variable, we assume it equal to 1, 
    but it does not the computation regarding the Doppler frequency.
    
    Returns realistic values of: \n measured phases, angle of arrivals, and variables. 
    """
    f_d = np.random.uniform(-1000,1000)
    f_off = np.random.normal(100000,10000,2)
    v = np.random.uniform(-5,5)
    while True:
        check = True
        zetas = np.random.uniform(-np.pi/4,np.pi/4,3)
        eta = np.random.uniform(0,2*np.pi)
        # for z in zetas:
        #     if z-(2*eta)<1e-10:
        #         check = False
        #         #print('problem 1')
        #     if z<1e-10:
        #         check = False
        #         #print('problem 2')
        if check:
            break
    phases = get_phases(eta, f_d, v, f_off, zetas, k=k)
    return phases, zetas, eta

def solve_system(phases, zetas):      
    alpha = (phases[1]-phases[3])*np.cos(zetas[2])
    delta = phases[2]-phases[1]
    gamma = (phases[3]-phases[2])*np.cos(zetas[1])
    return alpha, delta, gamma

N = 1000000
counter = 0
for i in range(N):
    phases, zetas, eta = get_input()
    alpha, delta, gamma = solve_system(phases,zetas)
    #for z in zetas:
    if abs(alpha+delta+gamma)<1e-5:
        counter += 1
print('The condition is met: ' + str(counter/N))