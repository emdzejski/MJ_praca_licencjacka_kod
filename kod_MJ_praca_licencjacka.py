import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model


#fitting functions

def sigmoid(x,a=-0.56,z=90,c=-3.5,d=0.5):
    y = a/( 1 + np.exp( (x-z)/c ) ) + d
    return y


def DoseResponse(x, a =1, b = 1, z=90,d=1): #dose response sigmoid
    y = a + (b-a)/(1+np.power(10, (z-x)*d ))
    return y

def hill(x,a=0.7,b=0.01,z=90,d=15): #hill function
    y= a + (b-a)*np.power(x,d) / (np.power(z,d) + np.power(x,d))
    return y

def logDoseResponse(x,a=1,b=1,z=90,d=1): #logistic dose response
    y = b + (a-b)/(1 + np.power(x/z,d))
    return y

def logistic5params(x,a=1,b=1,z=90,d=1,e=1): #logistic with 5 params
    y = a + (b-a)/ np.power( (1 + np.power(z/x,d)), e )
    return y

def atan(x, a=20, z = 5, c=1):
    y = -np.arctan(x/a - z ) + c
    return y

def tanh(x, a=10,z=90,c=0.5):
   y = -np.tanh(x/a - z) + c 
   return y

#lmfit models
sig = Model(sigmoid)
doseResp = Model(DoseResponse)
h = Model(hill,nan_policy = 'propagate') #nan policy flag was necessary 
#to handle one pencil beam profile 
logDoseResp = Model(logDoseResponse)
log5params = Model(logistic5params)
my_atan = Model(atan)
my_tanh = Model(tanh)

models = [sig,doseResp,h,logDoseResp,log5params] #list of models
models_funcs = {sig:sigmoid, doseResp:DoseResponse, h:hill,logDoseResp:logDoseResponse,log5params:logistic5params, my_atan:atan,my_tanh:tanh}

def genLin(bins,shift=1.25,step=2.5): # bins=50 for a pencil beam, 60 for ccb cubes
    rng = []
    for i in range(0,bins):
        rng.append(i*step + shift)
    return np.array(rng)


def rf(path): #function to read a file
    data = []
    with open(path, mode = 'r') as f:
        init_data = f.read().splitlines() #reading a file line by line
    for i in range(len(init_data)):
        row = init_data[i].split()
        data = data + row
    #print(init_data[8])
    return np.array([float(n) for n in data ])

def find_min(xs,data): #computes a differential of data, returns min value
    
    idx = np.where(xs>=20)
    diff = np.gradient(data[idx],xs[idx])
    short_range = xs[idx]

    #the index of the found value needs to be shifted by one 
    
    #np.where returns a tuple, so the first element needs to be extracted
    min_idx = np.where(diff == np.min(diff))[0] - 1
    return short_range[min_idx]
    


def PlotandFit(path,model,start=0,stop=0):
    data = rf(path)
    #check whether the file is a PMMA or  water phantom profile
    if np.where(data == np.max(data))[0] > 20: #PMMA
        rng = genLin(bins = 50)
        if ( (start == 0) and (stop ==0) ):
            #min = find_min(rng, data)
            start_idx = np.where(data==np.max(data))
            start = rng[start_idx] #start for pencil beam 
            stop = start+18
    elif np.where(data == np.max(data))[0] < 10: #water
        rng = genLin(bins = 60)
        if ( (start == 0) and (stop ==0) ):
            min = find_min(rng, data)
            start = min-28 
            stop = min+30
            
    r = []
    # +1 to include "stop" bin value in the range
    for i in range(0, int((stop-start)/2.5)+1):
        r.append( start + i*2.5 )   
    r = np.array(r).flatten()
    idx = np.where( (rng>=start) & (rng<=stop) )

    result = model.fit(data[idx],x = r)
    fit_params = list(result.best_values.values())
    xs = np.linspace(start, stop, 10000)
    func = models_funcs[model]
    plt.step(rng, data,label='dane',c = 'black')
    plt.plot(xs, func(xs, *fit_params),label='dopasowanie',linewidth=2.5, color='red')
    plt.axvline(result.params['z'].value,color='m', label = 'parametr z',linewidth=2.25)
    plt.ylim(0,1.1)
    plt.xlabel('zasięg (mm)',fontsize = 12)
    plt.ylabel('aktywność (j.u)',fontsize = 12)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return result.params['z'].value, result.params['z'].stderr, result.chisqr
PlotandFit("/home/mateusz/licencjat/ccb_kostki/Phantom2_Field7_fov20.txt", sig)


 