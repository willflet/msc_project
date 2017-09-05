import pickle
import numpy as np
import pandas as pd

data='10k'

with open('../data/cleaned_data_'+data+'.pickle','rb') as f:
    df = pickle.load(f)
n = len(df)

folds=[]
for i in range(0,n-n%5,n//5):
    Xs = {lag:np.empty((0,2+2*lag),dtype=np.float16) for lag in range(1,5)}
    Ys = {lag:np.empty((0,2),dtype=np.float16) for lag in range(1,5)}

    lengths = []
    halftrajs = []
    dests = []
    for index,row in df.iloc[i:i+n//5].iterrows():
        T = row['LENGTH']
        lengths.append(T)
        halftrajs.append(row['POLYLINE'][:T//2,:])
        dests.append(row['ENDPOINTS'][2:])

        for lag in range(1,5):
            if T<=lag: continue

            block = np.tile(row['ENDPOINTS'][:2],(T-lag,1))
            for t in range(lag):
                block = np.concatenate((block,row['POLYLINE'][t:t+(T-lag),:]),axis=1)
            Xs[lag] = np.concatenate((Xs[lag],block),axis=0)
            Ys[lag] = np.concatenate((Ys[lag],row['POLYLINE'][lag:T,:]),axis=0)

    folds.append((Xs,Ys,lengths,halftrajs,dests))
with open('../data/prediction_folds'+data+'.pickle','wb') as f:
    pickle.dump(folds,f,protocol=pickle.HIGHEST_PROTOCOL)
