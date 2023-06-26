import os
import numpy as np



# Get the current working directory
cwd = os.getcwd()


def dataset(noTrSamples, noTsSamples,loaded, Normalize):
    np.random.seed(1)
    loaded = np.nan_to_num(loaded)
    tr_index = np.random.choice(loaded.shape[0],noTrSamples,replace=False)
    ind = np.zeros(loaded.shape[0], dtype=bool)
    ind[tr_index] = True

    trX = loaded[ind,:-1]
    trY = loaded[ind,-1]

    tsX = loaded[~ind,:-1]
    tsY = loaded[~ind,-1]

   # Normalize the data 
    if Normalize:
      trX = np.nan_to_num(trX/trX.max(axis=0))
      tsX = np.nan_to_num(tsX/tsX.max(axis=0))
      

    return trX, trY, tsX, tsY

#variable to the data folder path
datapath = cwd

#seperate functions for the dataset
def ridge_reg_data(noTrSamples=300, noTsSamples=100):
    data_dir = os.getcwd()
    fd = os.path.join(data_dir, 'ridge_signal.csv')
    loaded = np.genfromtxt(fname=fd,skip_header=1,delimiter=',')
    return dataset(noTrSamples, noTsSamples,loaded,Normalize= False)
    
#seperate functions for the dataset
def pima_data(noTrSamples=500, noTsSamples=268):
    data_dir = os.getcwd()
    fd = os.path.join(data_dir, 'diabetes.csv')
    loaded = np.genfromtxt(fname=fd,skip_header=1,delimiter=',')
    return dataset(noTrSamples, noTsSamples,loaded,Normalize= False)
