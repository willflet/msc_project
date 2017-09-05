import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN,SpectralClustering,AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

data = '10k'
method = 'spectral'

K = 48
min_samples = 10
eps = 0.1
gamma = 0.01

with open('../data/cleaned_data_'+data+'.pickle','rb') as f:
    df = pickle.load(f)
ends = np.array(df['ENDPOINTS'].tolist())
trajs = df['POLYLINE']

if method == 'density':
    model = DBSCAN(min_samples=min_samples,
                    eps=eps)
    prefix = 'eps'+str(eps)+'_minsamples'+str(min_samples)
elif method == 'spectral':
    model = SpectralClustering(n_clusters=K,
                                gamma=gamma,
                                assign_labels='kmeans')
    prefix = 'gamma'+str(gamma)+'_'+str(K)+'clusters'
elif method == 'hierarchical':
    model = AgglomerativeClustering(n_clusters=K,
                                    linkage='average')
    prefix = str(K)+'clusters'
else:
    print('metric should be "density","hierarchical" or "spectral"')
    exit()

startlabels = model.fit(ends[:,:2]).labels_
endlabels = model.fit(ends[:,2:]).labels_

if method == 'density':
    K = len(set(startlabels)) - (1 if -1 in startlabels else 0)
    print('found',K,'clusters of starts')
    print('noise:',np.sum(startlabels==-1),'out of',len(startlabels))

    K = len(set(endlabels)) - (1 if -1 in endlabels else 0)
    print('found',K,'clusters of ends')
    print('noise:',np.sum(endlabels==-1),'out of',len(endlabels))

startpopulations=[]
for k in range(K):
    startpopulations.append(np.sum(startlabels==k))
print(startpopulations)
endpopulations=[]
for k in range(K):
    endpopulations.append(np.sum(endlabels==k))
print(endpopulations)

sns.set_palette(sns.color_palette('Paired',12))
plt.figure(figsize=(4,4))
for _,traj in trajs.iteritems():
    plt.plot(traj[:,0],traj[:,1],linewidth=0.1,color='0.8',zorder=1)
idx = np.where(startlabels==-1)[0]
plt.scatter(ends[idx,0],ends[idx,1],s=3,color='0.8',zorder=2)
for k in range(max(startlabels)):
    idx = np.where(startlabels==k)
    plt.scatter(ends[idx,0],ends[idx,1],s=6,zorder=3,alpha=0.5/int(data[:-1]))
plt.axes().set_aspect('equal','datalim')
plt.axis('off')
plt.xlim((-960,-950))
plt.ylim((5015,5045))
plt.savefig('../images/termini/'+method+'/'+prefix+'starts.png',dpi=250)
plt.close()

plt.figure(figsize=(4,4))
for _,traj in trajs.iteritems():
    plt.plot(traj[:,0],traj[:,1],linewidth=0.1,color='0.8',zorder=1)
idx = np.where(endlabels==-1)[0]
plt.scatter(ends[idx,2],ends[idx,3],s=3,color='0.8',zorder=2)
for k in range(max(endlabels)):
    idx = np.where(endlabels==k)
    plt.scatter(ends[idx,2],ends[idx,3],s=6,zorder=3,alpha=0.5/int(data[:-1]))
plt.axes().set_aspect('equal','datalim')
plt.axis('off')
plt.xlim((-960,-950))
plt.ylim((5015,5045))
plt.savefig('../images/termini/'+method+'/'+prefix+'finishes.png',dpi=250)
plt.close()

with open('../data/populations/starts'+method+prefix+'.pickle','wb') as f:
    pickle.dump(startpopulations,f)
with open('../data/populations/ends'+method+prefix+'.pickle','wb') as f:
    pickle.dump(endpopulations,f)
