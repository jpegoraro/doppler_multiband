import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tikzplotlib as tik
import os

def plot_boxplot(path, errors, xlabel, ylabel, xticks, title, name=''):
    """
        Plot boxplots using seaborn for the given list of errors.
    """
    plt.figure(figsize=(12,8))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    ax = sns.boxplot(data=errors, orient='v', palette='rocket', showfliers=False)
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.title(title)
    #plt.ylim(top=0.25)
    plt.savefig(path+name+'.png')
    tik.save(path+name+'.tex')
    plt.show()
    plt.close()

def load_data(path):
    files = os.listdir(path)
    data = []
    for f in files:
        if f=='eta' or f=='speed' or f=='time':
            continue
        info = f.split('_')
        data.append(
            {
                'data' : np.load(path + '/' + f),
                'interval' : int(info[1][1:]),
                'freq' : int(info[2][2:]),
                'npath' : int(info[3][2:]),
                'snr' : int(info[4].split('.')[0][3:])
            }
        )
    return data


path = 'cir_estimation_sim/data/varying_snr'
subdir = '/aoa3'
data = load_data(path+subdir)
#labels = [25,50,100,200,400]
labels = [-5,0,10,20,30]
#labels = [2,4,6,8,10]
fcs = [28,5]
for fc in fcs:
    errors = []
    for s in labels:
        for d in data:
            if d[path.split('_')[-1]]==s and d['freq']==fc:
                errors.append(d['data'])
                print('\nfc= ' + str(fc) + ' GHz, SNR= ' + str(s) + ' dB')
                print(np.median(d['data']))
    plot_boxplot('cir_estimation_sim/plot'+subdir+'/',errors,'SNR [dB]','fD relative error',np.array(labels),'varying snr AoA=5°, fc=%s GHz' %(fc),'var_snrfc_%s' %(fc))