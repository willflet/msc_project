import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

data='10k'
K=2
lag=3

prefix=data+'K'+str(K)+'lag'+str(lag)

with open('../data/cleaned_data_'+data+'.pickle','rb') as f:
    df = pickle.load(f)
trajs = df['POLYLINE']
ends = df['ENDPOINTS']
times = df['TIMESTAMP'].values
print(times)

with open('../data/Mo'+str(K)+'G_results'+data+'_lag'+str(lag)+'_clusters.pickle','rb') as f:
    labels = pickle.load(f)

populations=[]
for k in range(K):
    populations.append(np.sum(labels==k))
print(populations)


times_per_cluster = []
for k in range(K):
    idx = np.where(labels==k)
    times_per_cluster.append(np.mod(times[idx[0]],86400)/ 3600)

for k in range(K):
    plt.figure()
    plt.hist(times_per_cluster[k],bins=96)
    plt.savefig('../images/MoGhist'+str(k)+'.png')


pal = sns.color_palette()

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
        plt.plot(traj[:,0],traj[:,1],linewidth=0.5,color=pal[k%12],alpha=0.25/int(data[:-1]),zorder=k+2)
    i += 1
plt.axes().set_aspect('equal','datalim')
plt.axis('off')
plt.xlim((-960,-950))
plt.ylim((5015,5045))
plt.savefig('../images/MoG/'+prefix+'endpoints.png',dpi=250)
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
        plt.plot(OD[::2],OD[1::2],linewidth=0.5,color=pal[k%12],alpha=0.5/int(data[:-1]),zorder=k+3)
    i += 1
plt.axes().set_aspect('equal','datalim')
plt.axis('off')
plt.xlim((-960,-950))
plt.ylim((5015,5045))
plt.savefig('../images/MoG/'+prefix+'beelines.png',dpi=250)
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
    plt.savefig('../images/MoG/'+prefix+'endpoints'+str(k)+'.png',dpi=250)
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
    plt.savefig('../images/MoG/'+prefix+'beelines'+str(k)+'.png',dpi=250)
    plt.close()

with open('../data/populations/MoG'+prefix+'.pickle','wb') as f:
    pickle.dump(populations,f)
