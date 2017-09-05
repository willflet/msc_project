import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/metaData_taxistandsID_name_GPSlocation.csv',usecols=[2,3])
ll = np.reshape(np.concatenate((df['Longitude'],df['Latitude'])).T,(2,-1)).T


with open('../data/cleaned_data_10k.pickle','rb') as f:
    df = pickle.load(f)
trajs = df['POLYLINE']

R = 6371 # radius of earth in km
ll2xy = lambda x: R*np.stack((np.pi*x[:,0]/180,np.log(np.tan(np.pi/4 + np.pi*x[:,1]/360))),axis=1)
xy = ll2xy(ll)
print('projection applied')

plt.figure(figsize=(4,4))
for _,traj in trajs.iteritems():
    plt.plot(traj[:,0],traj[:,1],linewidth=0.1,color='0.8',zorder=1)
plt.scatter(xy[:,0],xy[:,1],s=6,color='0.2',zorder=2)

plt.axes().set_aspect('equal','datalim')
plt.axis('off')
plt.xlim((-960,-950))
plt.ylim((5015,5045))
plt.savefig('../images/taxi_stands.png',dpi=250)
