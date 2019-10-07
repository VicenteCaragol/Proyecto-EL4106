import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import random
from PyEMD import EMD
from scipy.stats import kurtosis, skew
from statistics import mean, median, variance, stdev
import math
import os
import time

start = time.time()


names = ["cyl", "hook", "lat", "palm", "spher", "tip"]
dt= 1.0/500

female_1 = loadmat('female_1.mat')
female_2 = loadmat('female_2.mat')
female_3 = loadmat('female_3.mat')
male_1 = loadmat('male_1.mat')
male_2 = loadmat('male_2.mat')
male_day_1 = loadmat('male_day_1.mat')
male_day_2 = loadmat('male_day_2.mat')
male_day_3 = loadmat('male_day_3.mat')


def JoinData(Data1, Data2):
    for key in names:
        Data1[key] = np.append(Data1[key], Data2[key], axis=0)

def MakeSmall(datos):
    for key in names:
        datos[key] = datos[key][:, 0:2500, :]
    
def Reorder(data):
    cyl_ch1 = data['cyl_ch1']
    cyl_ch2 = data['cyl_ch2']
    hook_ch1 = data['hook_ch1']
    hook_ch2 = data['hook_ch2']
    lat_ch1 = data['lat_ch1']
    lat_ch2 = data['lat_ch2']
    palm_ch1 = data['palm_ch1']
    palm_ch2 = data['palm_ch2']
    spher_ch1 = data['spher_ch1']
    spher_ch2 = data['spher_ch2']
    tip_ch1 = data['tip_ch1']
    tip_ch2 = data['tip_ch2']
    
    cyl = makeArray(cyl_ch1, cyl_ch2)
    hook = makeArray(hook_ch1, hook_ch2)
    lat = makeArray(lat_ch1, lat_ch2)
    palm = makeArray(palm_ch1, palm_ch2)
    spher = makeArray(spher_ch1, spher_ch2)
    tip = makeArray(tip_ch1, tip_ch2)
    
    return {"cyl": cyl, "hook" : hook, "lat" : lat, "palm" : palm, "spher" : spher, "tip": tip}
    
def makeduple(A1, A2):
    container = []
    for i in range(len(A1)):
        container.append([A1[i], A2[i]])
    return np.array(container)

def makeArray(ch1, ch2):
    container = []
    for i in range(len(ch1)):
        duple = makeduple(ch1[i], ch2[i])
        container.append(duple)
    return np.array(container)

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
     
        
def ShowIMF(vec):    
    IMF = EMD().emd(vec[0], np.arange(0, 6, dt))
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
    
    return IMF
    
    
'''---------------Time domain characteristics---------------------'''
    
#Integrated Electromyogram (IEMG): sum of the absolute values of EMG       
def IEMG(array):
    IEMG = 0
    for i in array:
        IEMG += abs(i)
    return IEMG

#Mean Absolute Value (MAV):  average value of the absolute values of EMG
def MAV(array):
    return IEMG(array)/len(array)

#Modified Mean Absolute Value 1 (MMAV1): is an extension of MAV using weighting window function
def MMAV1(array):
    MMAV =0
    N= len(array)
    for i in range(0, N):
        if (0.25*N <= i)or( i <= 0.75*N):
            MMAV += abs(array[i])
        else:
            MMAV += 0.5*abs(array[i])
    return MMAV/N

#Modified Mean Absolute Value 2 (MMAV2): an extension of MAV using a continuous weighting window function
def MMAV2(array):
    MMAV =0
    N= len(array)
    for i in range(0, N):
        if (0.25*N <= i)or( i <= 0.75*N):
            MMAV += abs(array[i])
        elif (0.25*N > i):
            MMAV += ((4*i)/N)*abs(array[i])
        else:
            MMAV += ((4*(i-N))/N)*abs(array[i])
    return MMAV

#Simple Square Integral (SSI): Energy of the signal.
def SSI(array):
    SSI= 0
    for x in array:
        SSI += x*x
    return SSI

#Root Mean Square (RMS): modeled as amplitude modulated GRP
def RMS(array):
    return math.sqrt(SSI(array)/len(array))

#Waveform Length (WL): the cumulative length of the waveform over the time segment
def WL(array):
    wl = 0
    for i in range(0, len(array)-1):
        wl += abs(array[i+1] - array[i])
    return wl
        
#Zero Crossing (ZC): counts the times that the signal changes sign.
def ZC(array):
    ZC= 0
    for i in range(0, len(array)-1):
        if (array[i] > 0 and array[i+1] < 0) or (array[i] < 0 and array[i+1] > 0):
            ZC +=1
    return ZC

#Slope Sign Changes (SSC): counts the times the slope of the signal changes sign.
def SSC(array):
    SSC = 0
    for i in range(0, len(array)-1):
        if (array[i] < array[i+1] and array[i] < array[i-1]) or (array[i] > array[i+1] and array[i] > array[i-1]):
            SSC += 1
    return SSC

# Willison Amplitude (WAMP): number of times that the difference between sEMG signal amplitude among two adjacent segments that exceeds a predefined threshold to reduce noise effects same as ZC and SSC
def WAMP(array, threshold = 0.05): #threshold depends of the signal, 5-10 mV suggested for arm signals
    wamp= 0
    for i in range(0, len(array)-1):
        if (abs(array[i] - array[i+1]) >= threshold):
            wamp += 1
    return wamp
            
#mean, median, variance, stdev
#kurtosis, skew
#histogram
#Estas vienen en las librerias importadas
        
'''---------------Frequency domain characteristics---------------------'''      
#para pasar a frequency domain
def Fdom(array):    
    n= len(array)
    Y = np.fft.fft(array)/n # fft computing and normalization
    fft = Y[range(int(n/2))]
    abf= abs(fft)
    return abf

def MedianFreq(array):
    pSum = 0
    freq = 0
    while pSum < sum(array)/2:
        pSum += array[freq]
        freq += 1
    return freq

def MeanFreq(array):
    sumFreq= 0
    for i in range(len(array)):
        sumFreq += i*array[i]
    return sumFreq/sum(array)

#======================================================================================#
binstime = 9
binsfreq = 10
def ExtractFeatures(array):
    Features = []
    Features.append(MAV(array))
    Features.append(MMAV1(array))
    Features.append(MMAV2(array))
    Features.append(SSI(array))
    Features.append(RMS(array))
    Features.append(WL(array))
    Features.append(ZC(array))
    Features.append(SSC(array))
    Features.append(WAMP(array)) #revisar
    Features.append(mean(array))
    Features.append(median(array))
    Features.append(variance(array))
    Features.append(stdev(array))
    Features.append(kurtosis(array))
    Features.append(skew(array))
    Features += np.ndarray.tolist(np.histogram(array, binstime)[0])
    Features.append(MedianFreq(Fdom(array)))
    Features.append(MeanFreq(Fdom(array)))
    Features += np.ndarray.tolist(np.histogram(Fdom(array), binsfreq)[0])
    
    return Features

HistTime = []
for i in range(binstime):
    HistTime.append('HistTime_b{}'.format(i)) 
HistFreq = []
for i in range(binsfreq):
    HistFreq.append('HistFreq_b{}'.format(i)) 
FeaturesNames = ['MAV', 'MMAV1', 'MMAV2', 'SSI', 'RMS', 'WL', 'ZC', 'SSC', 
                 'WAMP', 'Mean', 'Median', 'Variance', 'StDev', 'Kurtosis',
                 'Skew'] + HistTime + ['MedianFreq', 'MeanFreq'] + HistFreq
                 
FeaturesNames = [x + '_ch1' for x in FeaturesNames] + [x + '_ch2' for x in FeaturesNames] + ['Class']
        
def TranformDictToDataFrame(Dic):
    df = pd.DataFrame(columns= FeaturesNames)
    clase = 0
    for key in names:
        for i in range(450):
            arf = ExtractFeatures(Dic[key][i, :, 0]) + ExtractFeatures(Dic[key][i, :, 1]) + [clase]
            df.loc[len(df)] = arf
            print(clase, ' ', i)
        clase += 1
    return df
            

datanames = [male_day_2, male_day_3, female_1, female_2, female_3, male_1, male_2]
c=0
male_day_1 = Reorder(male_day_1)
MakeSmall(male_day_1)
for name in datanames:
    print(c)    
    name = Reorder(name)
    MakeSmall(name)
    JoinData(male_day_1, name)
    c += 1
    
AllData = male_day_1
data_final= TranformDictToDataFrame(AllData)
dir_path = os.path.dirname(os.path.realpath(__file__))

data_final.to_csv(dir_path + '\features_extraction.csv', index = None, header=True)

end = time.time()
print("Tiempo total de ejecución: ", end - start, 's')
     