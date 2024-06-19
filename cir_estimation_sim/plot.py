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
    #plt.ylim(top=60,bottom=0)
    plt.savefig(path+name+'.png')
    tik.save(path+name+'.tex')
    plt.show()
    plt.close()

def load_data(path, speed=False):
    files = os.listdir(path)
    if not speed:
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
    else:
        data_abs = []
        data_rel = []
        for f in files:
            info = f.split('_')
            if info[1] == 'abs':
                data_abs.append(
                    {
                        'data' : np.load(path + '/' + f),
                        'interval' : int(info[2][1:]),
                        'freq' : int(info[3][2:]),
                        'npath' : int(info[4][2:]),
                        'snr' : int(info[5].split('.')[0][3:])
                    }
                )
            else:
                data_rel.append(
                    {
                        'data' : np.load(path + '/' + f),
                        'interval' : int(info[2][1:]),
                        'freq' : int(info[3][2:]),
                        'npath' : int(info[4][2:]),
                        'snr' : int(info[5].split('.')[0][3:])
                    }
                )
        return data_abs, data_rel
    

def plot_fd():
    path = 'cir_estimation_sim/data/varying_npath'
    subdir = ''#'/static_rx/aoa5'
    data = load_data(path+subdir)
    #labels = [25,50,100,200,400]
    #labels = [12,24,48,96,192,288]
    #labels = [-5,0,10,20,30]
    labels = [2,4,6,8,10]
    fcs = [5,28,60]
    for fc in fcs:
        errors = []
        for s in labels:
            for d in data:
                if d[path.split('_')[-1]]==s and d['freq']==fc:
                    errors.append(d['data'])
                    print('\nfc= ' + str(fc) + ' GHz, SNR= ' + str(s) + ' dB')
                    print(np.median(d['data']))
        plot_boxplot('cir_estimation_sim/plot'+subdir+'/',errors,'No. of paths','fD relative error',np.array(labels),'varying npath AoA=5°, fc=%s GHz' %(fc),'var_npathfc_%s' %(fc))

def plot_speed():
    path = 'cir_estimation_sim/data/varying_snr/speed'
    #labels = [25,50,100,200,400]
    labels = [-5,0,5,10,20]
    #labels = [2,4,6,8,10]
    fcs = [60]
    for fc in fcs:
        if fc==28:
            data = np.load(path+'/v_abs_k89_fc28_ns2.npy')
        if fc==5:
            data = np.load(path+'/v_abs_k32_fc5_ns2.npy')
        if fc==60:
            data = np.load(path+'/v_abs_k96_fc60_ns2.npy')
        print('\nfc= ' + str(fc))
        aoa=[5]
        for i in range(1): 
            print('\n\naverage speed estimate absolute error, AoA std=%s° '%(aoa[i]) + str(np.mean(data[:,:,i], axis=0))+'\n')
            print('median speed estimate absolute error, AoA std=%s° '%(aoa[i]) + str(np.median(data[:,:,i],axis=0))+'\n')
            plot_boxplot('cir_estimation_sim/plot/',data[:,:,i],'SNR [dB]','speed absolute error',np.array(labels),'varying snr AoA=%s°, fc=%s GHz' %(aoa[i],fc),'var_snr_speed_fc_%sa_%s' %(fc,aoa[i]))

def plot_eta():
    path = 'cir_estimation_sim/data/varying_snr/eta'
    #labels = [25,50,100,200,400]
    labels = [-5,0,5,10,20]
    #labels = [2,4,6,8,10]
    fcs = [28,5]
    for fc in fcs:
        if fc==28:
            data = np.load(path+'/eta_rel_k89_fc28_ns2.npy')
        if fc==5:
            data = np.load(path+'/eta_rel_k32_fc5_ns2.npy')
        print('\nfc= ' + str(fc))
        aoa=[5]
        for i in range(1): 
            print('\n\naverage eta estimate relative error, AoA std=%s° '%(aoa[i]) + str(np.mean(data[:,:,i], axis=0))+'\n')
            print('median eta estimate relative error, AoA std=%s° '%(aoa[i]) + str(np.median(data[:,:,i],axis=0))+'\n')
            plot_boxplot('cir_estimation_sim/plot/',data[:,:,i],'SNR [dB]','eta relative error [°]',np.array(labels),'varying snr AoA=%s°, fc=%s GHz' %(aoa[i],fc),'rel_var_snr_eta_fc_%sa_%s' %(fc,aoa[i]))

def plot_npath():
    path = 'cir_estimation_sim/data/varying_npath'
    labels = [2,4,6,8]
    fcs = [5,28,60]
    for fc in fcs:
        data = np.load(path+'/tot_5_fd_error_fc%s.npy'%(fc))
        print('\nfc= ' + str(fc))
        print(np.median(data))
        plot_boxplot('cir_estimation_sim/plot/',data,'No. of paths','fD relative error',np.array(labels),'varying npath AoA=5°, fc=%s GHz' %(fc),'var_npathfc_%s' %(fc))

def plot_interval():
    path = 'cir_estimation_sim/data/varying_interval'
    labels = [48,32,16,8,4,2]
    fcs = [28]
    for fc in fcs:
        data = np.load(path+'/fd_snr5_fc%s_ns2.npy'%(fc))
        print('\nfc= ' + str(fc))
        print(np.median(data))
        plot_boxplot('cir_estimation_sim/plot/',data,'interval [ms]','fD relative error',np.array(labels),'varying interval AoA=5°, fc=%s GHz' %(fc),'var_interfc_%s' %(fc))

def plot_snr():
    path = 'cir_estimation_sim/data/varying_snr'
    labels = [-5,0,5,10,20]
    fcs = [60]
    for fc in fcs:
        data = np.load(path+'/fd_k96_fc%s_ns2.npy'%(fc))
        print('\nfc= ' + str(fc))
        aoa=[1,3,5]
        for i in range(3): 
            print('\n\naverage fd estimate relative error, AoA std=%s° '%(aoa[i]) + str(np.mean(data[:,:,i], axis=0))+'\n')
            print('median fd estimate relative error, AoA std=%s° '%(aoa[i]) + str(np.median(data[:,:,i],axis=0))+'\n')
            plot_boxplot('cir_estimation_sim/plot/',data[:,:,i],'SNR [dB]','fD relative error',np.array(labels),'varying snr AoA=%s°, fc=%s GHz' %(aoa[i],fc),'var_snrfc_%sa_%s' %(fc,aoa[i]))

plot_interval()