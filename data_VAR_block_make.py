import pickle
import numpy as np
import pandas as pd

data= '10k'

with open('../data/cleaned_data_'+data+'.pickle','rb') as f:
    df = pickle.load(f)


Xs = {lag:np.empty((0,5+2*lag),dtype=np.float32) for lag in range(1,5)}
Ys = {lag:np.empty((0,2),dtype=np.float32) for lag in range(1,5)}

for lag in range(1,5):
    for index,row in df.iterrows():
        T = row['LENGTH']
        if T<=lag: continue
        block = np.tile(np.concatenate(([1],row['ENDPOINTS'])),(T-lag,1))
        for t in range(lag):
            block = np.concatenate((block,row['POLYLINE'][t:t+(T-lag),:]),axis=1)
        Xs[lag] = np.concatenate((X[lag],block),axis=0)
        Ys[lag] = np.concatenate((Y[lag],row['POLYLINE'][lag:T,:]),axis=0)

with open('../data/VAR_blocks'+data+'.pickle','wb') as f:
    pickle.dump((Xs,Ys),f,protocol=pickle.HIGHEST_PROTOCOL)
