import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

data = '100k'

with open('../data/interpolated'+data+'.pickle','rb') as f:
    trajs_100D = pickle.load(f).reshape((-1,100))

pca = PCA(n_components=11)

trajs_11D = pca.fit_transform(trajs_100D)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure()
plt.plot(range(12),1-np.cumsum(np.concatenate(([0],explained_variance_ratio))))
plt.yscale('log')
plt.savefig('../images/PCA.png')

with open('../data/interpolated'+data+'_11D.pickle','wb') as f:
    pickle.dump(trajs_11D,f,protocol=pickle.HIGHEST_PROTOCOL)
