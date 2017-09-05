import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

space = 'EDR'
method = 'density'

K = 12
min_samples = 2
eps = 1
gamma = 10**-8


if method == 'density':
    prefix = 'eps'+str(eps)+'_minsamples'+str(min_samples)
elif method == 'spectral':
    prefix = 'gamma'+str(gamma)+'_'+str(K)+'clusters'
elif method == 'hierarchical':
    prefix = str(K)+'clusters'
else:
    print('metric should be "density","hierarchical" or "spectral"')
    exit()

with open('../data/populations/'+space+method+prefix+'.pickle','rb') as f:
    populations = pickle.load(f)

if space=='ends' or space=='starts':
    folder = 'termini'
    prefix = prefix+space
else:
    folder = space

noise = 8272-sum(populations)

idx = np.argsort(populations)[::-1]


pal = sns.color_palette('Paired',12)
plt.figure(figsize=(0.4,2))
plt.axis('off')
bottom=0
for i in range(len(populations)):
    value = populations[idx[i]]
    plt.bar(0,value,width=1,
             bottom=bottom,
             color=pal[idx[i]%12])
    bottom += value
plt.bar(0,noise,width=1,bottom=bottom,color='0.8')

plt.savefig('../images/'+folder+'/'+method+'/'+prefix+'bar.png')
