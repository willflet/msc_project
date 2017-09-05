import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('../data/data.pickle','rb') as f:
    df = pickle.load(f)
lengths = []

for index,traj in df['POLYLINE'].iteritems():
    lengths.append(len(traj)*15)


plt.figure()
plt.hist(lengths,
    bins = np.exp(np.linspace(np.log(min(lengths)), np.log(max(lengths)), 100)))
plt.xscale('log')
plt.xlabel('Journey time (seconds)')
plt.ylabel('Frequency')
plt.savefig('../images/lengths.png')
