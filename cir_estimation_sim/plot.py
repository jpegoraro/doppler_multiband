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
    plt.ylim(top=0.25)
    plt.savefig(path+name+'.png')
    plt.show()
    #tik.save(path+name+'.tex')
    plt.close()

def load_data(path):
    files = os.listdir(path)
    data = []
    for f in files:
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


path = 'cir_estimation_sim/data/varying_interval'
data = load_data(path)
labels = [25,50,100,200,400]
#labels = [0,10,20,30]
#labels = [2,4,6,8,10]
fcs = [60,28]
for fc in fcs:
    errors = []
    for s in labels:
        for d in data:
            if d[path.split('_')[3]]==s and d['freq']==fc:
                errors.append(d['data'])

    plot_boxplot('cir_estimation_sim/plot/',errors,'interval','fD relative error',np.array(labels)*0.08,'varying interval, fc=%s GHz' %(fc),'var_intervalfc_%s' %(fc))