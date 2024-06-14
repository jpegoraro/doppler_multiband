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
    path = 'cir_estimation_sim/data/varying_snr'
    subdir = '/aoa1'
    data = load_data(path+subdir+'/eta',speed=True)
    #labels = [25,50,100,200,400]
    labels = [-5,0,10,20,30]
    #labels = [2,4,6,8,10]
    fcs = [28,5]
    for fc in fcs:
        errors_abs = []
        errors_rel = []
        for s in labels:
            for i in range(len(data[0])):
                d_abs = data[0][i]
                d_rel = data[1][i]
                if d_abs[path.split('_')[-1]]==s and d_abs['freq']==fc:
                    errors_abs.append(d_abs['data'])
                    print('\nfc= ' + str(fc) + ' GHz, SNR= ' + str(s) + ' dB')
                    print('absolute median error: '+str(np.median(d_abs['data'])))
                if d_rel[path.split('_')[-1]]==s and d_rel['freq']==fc:
                    errors_rel.append(d_rel['data'])
                    # print('\nfc= ' + str(fc) + ' GHz, SNR= ' + str(s) + ' dB')
                    # print('relative median error: '+str(np.median(d_rel['data'])))
        #plot_boxplot('cir_estimation_sim/plot'+subdir+'/',errors_rel,'SNR [dB]','speed relative error',np.array(labels),'varying snr AoA=5°, fc=%s GHz' %(fc),'speed_rel_var_snrfc_%s' %(fc))
        #plot_boxplot('cir_estimation_sim/plot'+subdir+'/',errors_abs,'SNR [dB]','speed absolute error',np.array(labels),'varying snr AoA=5°, fc=%s GHz' %(fc),'speed_abs_var_snrfc_%s' %(fc))

def plot_npath():
    path = 'cir_estimation_sim/data/varying_npath'
    labels = [2,4,6,8]
    fcs = [5,28,60]
    for fc in fcs:
        data = np.load(path+'/last_tot_5_fd_error_fc%s.npy'%(fc))
        print('\nfc= ' + str(fc))
        print(np.median(data))
        plot_boxplot('cir_estimation_sim/plot/',data,'No. of paths','fD relative error',np.array(labels),'varying npath AoA=5°, fc=%s GHz' %(fc),'var_npathfc_%s' %(fc))

def plot_interval():
    path = 'cir_estimation_sim/data/varying_interval'
    labels = [48,32,16,8,4,2]
    fcs = [5,28]
    for fc in fcs:
        data = np.load(path+'/fd_snr5_fc%s_ns2.npy'%(fc))
        print('\nfc= ' + str(fc))
        print(np.median(data))
        plot_boxplot('cir_estimation_sim/plot/',data,'No. of paths','fD relative error',np.array(labels),'varying npath AoA=5°, fc=%s GHz' %(fc),'var_interfc_%s' %(fc))

def plot_snr():
    path = 'cir_estimation_sim/data/varying_snr'
    labels = [-5,0,5,10,20]
    fcs = [28,60,5]
    for fc in fcs:
        data = np.load(path+'/fd_k89_fc%s_ns2.npy'%(fc))
        print('\nfc= ' + str(fc))
        aoa=[1,3,5]
        for i in range(3): 
            print('\n\naverage fd estimate relative error, AoA std=%s° '%(aoa[i]) + str(np.mean(data[:,:,i], axis=0))+'\n')
            print('median fd estimate relative error, AoA std=%s° '%(aoa[i]) + str(np.median(data[:,:,i],axis=0))+'\n')
            plot_boxplot('cir_estimation_sim/plot/',data[:,:,i],'No. of paths','fD relative error',np.array(labels),'varying snr AoA=%s°, fc=%s GHz' %(aoa[i],fc),'var_snrfc_%sa_%s' %(fc,aoa[i]))

plot_snr()