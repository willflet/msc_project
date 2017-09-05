import pickle
import numpy as np
import pandas as pd
from traj_dist.distance import edr,dtw
from sys import exit

data='10k'
metric = 'DTW'

with open('../data/cleaned_data_'+data+'.pickle','rb') as f:
    df = pickle.load(f)
dist = np.zeros((len(df),len(df)),dtype=np.float32)


if metric == 'euclidean':
    ends = df['ENDPOINTS']

    for i in range(len(ends)-1):
        for j in range(i+1,len(ends)):
            dist[i,j] = np.linalg.norm(ends.iloc[i] - ends.iloc[j])
            dist[j,i] = dist[i,j]

    with open('../data/euclidean_matrix_'+data+'.pickle','wb') as f:
        pickle.dump(dist,f,protocol=pickle.HIGHEST_PROTOCOL)

elif metric == 'EDR':
    trajs = df['POLYLINE']
    threshold = 0.1 # in km

    for i in range(len(trajs)-1):
        n_i = df['LENGTH'].iloc[i]
        for j in range(i+1,len(trajs)):
            n_j = df['LENGTH'].iloc[j]
            dist[i,j] = edr(trajs.iloc[i],trajs.iloc[j],
                            eps = threshold) * max(n_i,n_j)
            dist[j,i] = dist[i,j]

elif metric == 'DTW':
    trajs = df['POLYLINE']

    DTW = np.zeros((len(trajs),len(trajs)),dtype=np.float32)
    for i in range(len(trajs)-1):
        n_i = df['LENGTH'].iloc[i]
        for j in range(i+1,len(trajs)):
            n_j = df['LENGTH'].iloc[j]
            dist[i,j] = dtw(trajs.iloc[i],trajs.iloc[j])
            dist[j,i] = dist[i,j]

else:
    print('metric should be "EDR","DTW" or "euclidean"')
    exit()


with open('../data/'+metric+'_matrix_'+data+'.pickle','wb') as f:
    pickle.dump(dist,f,protocol=pickle.HIGHEST_PROTOCOL)
