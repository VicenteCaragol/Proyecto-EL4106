import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import random
from PyEMD import EMD

dt= 1.0/500

def ShowRandomSample(filename):
    data = loadmat(filename)
    dt= 1.0/500
    for i in range(3, len(data), 2):
        key_1= list(data)[i]
        key_2= list(data)[i+1]
        ch1= np.array(data[key_1])
        ch2= np.array(data[key_2])
        tfinal= len(ch1[0])/500
        t= np.arange(0, tfinal, dt)
        sample= random.randint(0,len(ch1[:,0])-1) 
        
        fig = plt.figure(figsize=(13, 5))
        ax = plt.axes()
        ax.set_title('Ejemplo Nº{} clase: {}'.format(sample, key_1[:-4]))
        ax.plot(t, ch1[sample], linewidth=0.5, alpha=0.5, label='ch1')
        ax.plot(t, ch2[sample], linewidth=0.5, alpha=0.5, label='ch2')
        ax.set_xlabel('Tiempo [s]')
        ax.set_ylabel('Voltaje [V]')
        ax.legend()    

#ShowRandomSample('female_1.mat')

data = loadmat('female_1.mat')
cyl_ch1 = data['cyl_ch1']
IMF = EMD().emd(cyl_ch1[0], np.arange(0, 6, dt))
N = IMF.shape[0]+1

# Plot results
plt.figure(figsize=(10, 25))
plt.subplot(N,1,1)
plt.plot(np.arange(0, 6, dt), cyl_ch1[0], 'r')
plt.xlabel("Time [s]")
for n, imf in enumerate(IMF):
    plt.subplot(N,1,n+2)
    plt.plot(np.arange(0, 6, dt), imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()