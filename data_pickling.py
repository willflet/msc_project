import pandas as pd
import numpy as np
import pickle
from pyproj import Proj

str2array = lambda x: np.array(eval(x))
df = pd.read_csv('../data/train.csv',usecols=[4,5,8],converters={'POLYLINE':str2array})
print('csv loaded')
df = df.assign(LENGTH=df['POLYLINE'].apply(len))
df = df[df['LENGTH']>1]
print('lengths calculated')

R = 6371 # radius of earth in km
ll2xy = lambda x: R*np.stack((np.pi*x[:,0]/180,np.log(np.tan(np.pi/4 + np.pi*x[:,1]/360))),axis=1)
df['POLYLINE'] = df['POLYLINE'].apply(ll2xy)
print('projection applied')

poly2ends = lambda x: np.concatenate((x[0,:], x[-1,:]))
df = df.assign(ENDPOINTS=df['POLYLINE'].apply(poly2ends))

with open('../data/data.pickle','wb') as f:
    pickle.dump(df,f,protocol=pickle.HIGHEST_PROTOCOL)

df_100k = df.sample(n=100000)
with open('../data/subsampled_data_100k.pickle','wb') as f:
    pickle.dump(df_100k,f,protocol=pickle.HIGHEST_PROTOCOL)
del df_100k

df_10k = df.sample(n=10000)
with open('../data/subsampled_data_10k.pickle','wb') as f:
    pickle.dump(df_10k,f,protocol=pickle.HIGHEST_PROTOCOL)
del df_10k

df_1k = df.sample(n=1000)
with open('../data/subsampled_data_1k.pickle','wb') as f:
    pickle.dump(df_1k,f,protocol=pickle.HIGHEST_PROTOCOL)
