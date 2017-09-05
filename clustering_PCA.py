import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN,SpectralClustering,AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

data = '10k'
method = 'density'

K = 48
min_samples = 6
eps = 2.5
gamma = 0.0001

with open('../data/interpolated'+data+'_11D.pickle','rb') as f:
    projected_trajs = pickle.load(f)

if method == 'density':
    model = DBSCAN(min_samples=min_samples,
                    eps=eps)
    prefix = 'eps'+str(eps)+'_minsamples'+str(min_samples)
elif method == 'spectral':
    model = SpectralClustering(n_clusters=K,
                                gamma=gamma)
    prefix = 'gamma'+str(gamma)+'_'+str(K)+'clusters'
elif method == 'hierarchical':
    model = AgglomerativeClustering(n_clusters=K,
                                    linkage='average')
    prefix = str(K)+'clusters'
else:
    print('metric should be "density","hierarchical" or "spectral"')
    exit()

labels = model.fit(projected_trajs).labels_


if method == 'density':
    # Number of clusters in labels, ignoring noise if present.
    K = len(set(labels)) - (1 if -1 in labels else 0)
    print('found',K,'clusters')
    print('noise:',np.sum(labels==-1),'out of',len(labels))

populations=[]
for k in range(K):
    populations.append(np.sum(labels==k))
print(populations)

with open('../data/cleaned_data_'+data+'.pickle','rb') as f:
    df = pickle.load(f)
trajs = df['POLYLINE']
ends = df['ENDPOINTS']

pal = sns.color_palette('Paired',12)

plt.figure(figsize=(4,4))
i=0
for _,traj in trajs.iteritems():
    k = labels[i]
    if k==-1:
        plt.plot(traj[:,0],traj[:,1],linewidth=0.1,color='0.8',zorder=1)
    i += 1
i=0
for _,traj in trajs.iteritems():
    k = labels[i]
    if k!=-1:
        plt.plot(traj[:,0],traj[:,1],linewidth=0.5,color=pal[k%12],alpha=0.25/int(data[:-1]),zorder=2)
    i += 1
plt.axes().set_aspect('equal','datalim')
plt.axis('off')
plt.xlim((-960,-950))
plt.ylim((5015,5045))
plt.savefig('../images/PCA/'+method+'/'+prefix+'endpoints.png',dpi=250)
plt.close()

plt.figure(figsize=(4,4))
for _,traj in trajs.iteritems():
    plt.plot(traj[:,0],traj[:,1],linewidth=0.1,color='0.8',zorder=1)
i=0
for _,OD in ends.iteritems():
    k = labels[i]
    if k==-1:
        plt.plot(OD[::2],OD[1::2],linewidth=0.2,color='0.8',alpha=0.5/int(data[:-1]),zorder=2)
    else:
        plt.plot(OD[::2],OD[1::2],linewidth=0.5,color=pal[k%12],alpha=0.5/int(data[:-1]),zorder=3)
    i += 1
plt.axes().set_aspect('equal','datalim')
plt.axis('off')
plt.xlim((-960,-950))
plt.ylim((5015,5045))
plt.savefig('../images/PCA/'+method+'/'+prefix+'beelines.png',dpi=250)
plt.close()

for k in range(K):
    if populations[k] < 50: continue
    plt.figure(figsize=(4,4))
    i=0
    for _,traj in trajs.iteritems():
        label = labels[i]
        if label!=k:
            plt.plot(traj[:,0],traj[:,1],linewidth=0.1,color='0.8',zorder=1)
    i=0
    for _,traj in trajs.iteritems():
        label = labels[i]
        if label==k:
            plt.plot(traj[:,0],traj[:,1],linewidth=0.5,color=pal[k%12],alpha=0.25/int(data[:-1]),zorder=2)
        i += 1
    plt.axes().set_aspect('equal','datalim')
    plt.axis('off')
    plt.xlim((-960,-950))
    plt.ylim((5015,5045))
    plt.savefig('../images/PCA/'+method+'/'+prefix+'endpoints'+str(k)+'.png',dpi=250)
    plt.close()

    plt.figure(figsize=(4,4))
    i=0
    for _,traj in trajs.iteritems():
        label = labels[i]
        if label!=k:
            plt.plot(traj[:,0],traj[:,1],linewidth=0.1,color='0.8')
        i += 1
    i=0
    for _,OD in ends.iteritems():
        label = labels[i]
        if label==k:
            plt.plot(OD[::2],OD[1::2],linewidth=0.5,color=pal[k%12],alpha=0.8/int(data[:-1]))
        i += 1
    plt.axes().set_aspect('equal','datalim')
    plt.axis('off')
    plt.xlim((-960,-950))
    plt.ylim((5015,5045))
    plt.savefig('../images/PCA/'+method+'/'+prefix+'beelines'+str(k)+'.png',dpi=250)
    plt.close()

with open('../data/populations/PCA'+method+prefix+'.pickle','wb') as f:
    pickle.dump(populations,f)
