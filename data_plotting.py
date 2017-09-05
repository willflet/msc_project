import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('../data/subsampled_data_1k.pickle','rb') as f:
    df = pickle.load(f)

plt.figure(figsize=(4,4))
for index,traj in df['POLYLINE'].iteritems():
    if len(traj) > 120: continue    # exclude trips over 30 mins
    plt.plot(traj[:,0],traj[:,1],linewidth=0.3)
plt.axes().set_aspect('equal','datalim')
plt.axis('off')
#plt.xlim((-958,-954))
plt.ylim((5028,5040))
plt.savefig('../images/trajectories.png',dpi=250)

#sns.set_context(rc={'lines.markeredgewidth': 0.1})
#
#plt.figure()
#for index,traj in df['POLYLINE'].iteritems():
#    if len(traj) < 1: continue
#    plt.plot(traj[0,:],'bo')
#plt.axes().set_aspect('equal','datalim')
#plt.axis('off')
#plt.xlim((-8.74,-8.42))
#plt.ylim((41.05,41.28))
#plt.savefig('../images/origins.png')
#
#plt.figure()
#for index,traj in df['POLYLINE'].iteritems():
#    if len(traj) < 1: continue
#    plt.plot(traj[-1,:],'ro')
#plt.axes().set_aspect('equal','datalim')
#plt.axis('off')
#plt.xlim((-8.74,-8.42))
#plt.ylim((41.05,41.28))
#plt.savefig('../images/destinations.png')
