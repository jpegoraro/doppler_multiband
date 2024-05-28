import numpy as np
import matplotlib.pyplot as plt

delta_f = 240e3 # subcarrier spacing [Hz]
n_sc = 1666 # number of subcarriers
B = n_sc*delta_f # bandwidth [Hz] almost 400 MHz

def generate_16QAMsymbols(n_sc, unitAveragePower=True):
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



QAM_symbols = generate_16QAMsymbols(n_sc)
plt.scatter(QAM_symbols.real,QAM_symbols.imag)
power = np.mean(abs(QAM_symbols)**2)
QAM_symbols = QAM_symbols/power
plt.scatter(QAM_symbols.real,QAM_symbols.imag)
plt.show()
print(power)