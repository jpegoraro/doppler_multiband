import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pickle
import os
sym.init_printing()
#define symbolic variables
y1, y2, y3, T, fD, v, l, eta, a, b, c, K, sp = sym.symbols('y1 y2 y3 T fD v l eta a b c K sp')

#define loglikehood
def f1(fD, v, eta, T, l, x):
    return 2*sym.pi*T*(fD+(v/l)*(sym.cos(eta - x) - sym.cos(eta)))

def fm(v, eta, T, l, x):
    return 2*sym.pi*T*(v/l)*(sym.cos(eta - x) - sym.cos(eta))

def get_inv_FI_00(load=True):
    if load:
        for f in os.listdir('CRB'):
            if f=='simp_inv_FI.pkl':
                with open('plots/CRB/simp_inv_FI.pkl', 'rb') as inf:    
                    inv_FI = pickle.loads(inf.read())
                return inv_FI
    loglike = -(y1 - f1(fD, v, eta, T, l, a))**2 / (2*sp) - (y2 - fm(v, eta, T, l, b))**2 / (2*sp) - (y3 - fm(v, eta, T, l, c))**2 / (2*sp)

    # first and second derivatives

    dLdfD = sym.diff(loglike, fD)
    dLdeta = sym.diff(loglike, eta)
    dLdv = sym.diff(loglike, v)

    dLdfD2 = sym.diff(loglike, fD, 2)
    dLdeta2 = sym.diff(loglike, eta, 2)
    dLdv2 = sym.diff(loglike, v, 2)

    dLdfDdeta = sym.diff(dLdfD, eta)
    dLdfDdv = sym.diff(dLdfD, v)
    dLdvdeta = sym.diff(dLdv, eta)

    # dLdetadfD = sym.diff(dLdeta, fD)
    # dLdvdfD = sym.diff(dLdv, fD)
    # dLdetadv = sym.diff(dLdeta, v)

    #define Fisher information matrix

    Ey1 = f1(fD, v, eta, T, l, a)
    Ey2 = fm(v, eta, T, l, b)
    Ey3 = fm(v, eta, T, l, c)

    A = -dLdfD2
    B = -dLdfDdeta
    C = -dLdfDdv

    E = -dLdeta2.subs({y1:Ey1, y2:Ey2, y3:Ey3})  # here y remains and we take expectation
    D = -dLdvdeta.subs({y1:Ey1, y2:Ey2, y3:Ey3}) # here y remains and we take expectation
    F = -dLdv2

    FI = sym.Matrix(([A, B, C], [B, E, D], [C, D, F]))
    invFI = FI.inv()
    simp_inv_FI = sym.simplify(sym.trigsimp(invFI[0,0])) # simplified first element
    with open('CRB/simp_inv_FI.pkl', 'wb') as outf:
        outf.write(pickle.dumps(simp_inv_FI))
    return simp_inv_FI

def get_crb_heatmap(load=True):
    simp_inv_FI = get_inv_FI_00(load=load)
    sym.pprint(simp_inv_FI)
    varying_ab = np.zeros((360,360))
    varying_a = np.zeros(360)
    for i in range(360):
        print("iteration: ", i, end="\r")
        for j in range(1,361):
            temp = simp_inv_FI.subs({a:np.deg2rad(j),b:np.pi/3,c:np.deg2rad(i),T:0.08e-3,sp:1})
            if temp==sym.zoo:
                if j==360:
                    varying_ab[i,0] = 1e20
                else:
                    varying_ab[i,j] = 1e20
            else:
                if j==360:
                    varying_ab[i,0] = temp.evalf()
                else:
                    varying_ab[i,j] = temp.evalf()
        # if temp==sym.zoo:
        #     varying_a[i] = 1e20
        # else:
        #     varying_a[i] = temp.evalf()
    return varying_ab

#np.save('plots/CRB/varying_ac.npy', varying_ab)


varying_ab = np.load('CRB/varying_ac.npy')
for i in range(360):
    for j in range(360):
        if varying_ab[i,j] > 1e10:
            varying_ab[i,j] = 1e10
#plt.plot(varying_a)
plt.figure()
sns.heatmap(varying_ab, norm=LogNorm())
plt.ylabel('AoA static path 2')
#plt.xlabel('AoA static path 2')
plt.xlabel('AoA target path')
plt.show()