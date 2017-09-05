import theano
import theano.tensor as tt
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

data='1k'
lag = 3
D = 3+2*lag
with open('../data/prediction_folds'+data+'.pickle','rb') as f:
    folds = pickle.load(f)

performance = []
normalizedTperformance = []
normalizedDperformance = []

for testfold in range(5):
    X_train = np.empty((0,D),dtype=np.float32)
    Y_train = np.empty((0,2),dtype=np.float32)
    for (X,Y,_,_,_) in folds[0:testfold]:
        X_train = np.concatenate((X_train,X[lag]),axis=0)
        Y_train = np.concatenate((Y_train,Y[lag]),axis=0)
    for (X,Y,_,_,_) in folds[testfold+1:]:
        X_train = np.concatenate((X_train,X[lag]),axis=0)
        Y_train = np.concatenate((Y_train,Y[lag]),axis=0)

    lm = LinearRegression()
    lm.fit(X_train,Y_train)


    (_,_,lengths,halftrajs,dests) = folds[testfold]
    avT = np.mean(lengths)
    avL = 0
    for j,halftraj in enumerate(halftrajs):
        avL += np.linalg.norm(halftraj[0,:]-dests[j])
    avL /= len(halftrajs)

    for j,halftraj in enumerate(halftrajs):
        T = lengths[j]
        L = np.linalg.norm(halftraj[0,:]-dests[j])
        if T//2<3: continue
        for t in range(T//2,T):
            X = np.concatenate(([1],
                                halftraj[0,:],
                                halftraj[-lag:,:].reshape(2*lag))).reshape((1,-1))
            halftraj = np.concatenate((halftraj,lm.predict(X)),axis=0)
        performance.append(np.linalg.norm(halftraj[-1,:]-dests[j]))
        normalizedTperformance.append(performance[-1]*avT/T)
        normalizedDperformance.append(performance[-1]*avL/L)

print(np.mean(performance))

plt.figure(figsize=(4,3))
plt.hist(performance,bins=np.linspace(0,12,100))
plt.xlim((0,12))
plt.ylim((0,60))
plt.xlabel('prediction error (km)')
plt.ylabel('frequency')
plt.tight_layout()
plt.savefig('../images/VAR_destpredwithorigin_lag'+str(lag)+'.png')

plt.figure(figsize=(4,3))
plt.hist(normalizedTperformance,bins=np.linspace(0,12,100))
plt.xlim((0,12))
plt.ylim((0,60))
plt.xlabel('effective prediction error (km)')
plt.ylabel('frequency')
plt.tight_layout()
plt.savefig('../images/VAR_destpredwithoriginT_lag'+str(lag)+'.png')

plt.figure(figsize=(4,3))
plt.hist(normalizedTperformance,bins=np.linspace(0,12,100))
plt.xlim((0,12))
plt.ylim((0,60))
plt.xlabel('effective prediction error (km)')
plt.ylabel('frequency')
plt.tight_layout()
plt.savefig('../images/VAR_destpredwithoriginD_lag'+str(lag)+'.png')
