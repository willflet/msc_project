import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('../data/subsampled_data_100k.pickle','rb') as f:
    df = pickle.load(f)

df = df[df['LENGTH']<100]

counts=np.zeros((200,10))
good_indices = []
for index,traj in df['POLYLINE'].iteritems():
    diffs = np.linalg.norm(np.diff(traj,axis=0),axis=1)

    for threshold in range(200):
        jumps = np.sum(diffs>threshold/100)
        for j in range(10):
            if jumps>j:
                counts[threshold,j] += 1
        if threshold==75 and jumps==0:
            good_indices.append(index)

print(len(df))
df = df.loc[good_indices,:]
print(len(df))

with open('../data/cleaned_data_100k.pickle','wb') as f:
    pickle.dump(df,f,protocol=pickle.HIGHEST_PROTOCOL)

plt.figure()
line_colors = plt.cm.Blues(np.linspace(1,0.2,10))
for j in range(10):
    plt.plot(np.linspace(0,2,200),counts[:,j]/counts[0,0],
            c=line_colors[j],
            label=str(j+1))
plt.xlabel('coordinate change in 15 seconds (km)')
plt.ylabel('proportion of trajectories with at least n larger changes')
plt.legend(title='n')
plt.savefig('../images/diffsize.png')
