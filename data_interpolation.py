import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

data = '100k'

with open('../data/cleaned_data_'+data+'.pickle','rb') as f:
    df = pickle.load(f)
trajs = df['POLYLINE']

interpolated = np.zeros((len(trajs),50,2))

i=0
interp_t = np.linspace(0,1,50)
for _,traj in trajs.iteritems():
    T = len(traj)
    t = np.linspace(0,1,T)
    if T<4:
        f = interp1d(t,traj,axis=0,kind='linear')
    else:
        f = interp1d(t,traj,axis=0,kind='cubic')
    interpolated[i,:,:] = f(interp_t)
    i += 1

with open('../data/interpolated'+data+'.pickle','wb') as f:
    pickle.dump(interpolated,f,protocol=pickle.HIGHEST_PROTOCOL)
