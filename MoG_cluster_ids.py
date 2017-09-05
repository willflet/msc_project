import theano
import theano.tensor as tt
import pickle
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.misc import logsumexp
from pymc3.distributions.transforms import stick_breaking
from numpy.linalg import det, inv
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

data='1k'

with open('../data/cleaned_data_'+data+'.pickle','rb') as f:
    df = pickle.load(f)

lengths = np.array(df['LENGTH'])
del df

K = 4
lag = 2
D = 2*(5 + 2*lag)
with open('../data/VAR_blocks'+data+'.pickle','rb') as f:
    X,Y = pickle.load(f)
X,Y = X[lag],Y[lag]
n_samples = X.shape[0]
del X,Y

with open('../data/Mo'+str(K)+'G_results'+data+'_lag'+str(lag)+'.pickle','rb') as f:
    results = pickle.load(f)
B = results['B']

mus = []
taus = []
pi = stick_breaking.backward(results['pi_stickbreaking__']).eval()
print(pi)
for k in range(K):
    mus.append(results['mu_'+str(k)])
    packed_chol = results['packed_chol'+str(k)+'_cholesky_cov_packed__']
    chol = np.zeros((D,D))
    chol[np.tril_indices(D)] = packed_chol
    cov = np.dot(chol,chol.T)
    taus.append(inv(cov))

# Log likelihood of normal distribution
def logp_normal(mu, tau, value):
    # log probability of individual samples
    dim = tau.shape[0]
    delta = lambda mu: value - mu
    return -0.5 * (dim * np.log(2 * np.pi) - np.log(det(tau)) +
                         (delta(mu).dot(tau)*delta(mu)).sum(axis=1))


cluster_probs = np.array([logp_normal(mus[k],taus[k],B) for k in range(K)]).T
cluster_probs = np.log(pi) + cluster_probs - logsumexp(cluster_probs,axis=1,keepdims=True)
print(cluster_probs)
labels = []
count=0
for T in lengths:
    if T<=lag:
        labels.append(-1)
        continue

    cluster_ids = np.argmax(cluster_probs[count:count+T-lag,:],axis=1)
    print(cluster_ids)
    #print(np.argmax(np.sum(cluster_probs[count:count+T-lag,:],axis=0)))
    #print(mode(cluster_ids[count:count+T-lag])[0][0])
    labels.append(mode(np.argmax(cluster_probs[count:count+T-lag,:],axis=1))[0][0])
    count += T-lag

labels = np.array(labels)

for k in range(K):
    print(np.sum(labels==k))

with open('../data/Mo'+str(K)+'G_results'+data+'_lag'+str(lag)+'_clusters.pickle','wb') as f:
    pickle.dump(labels,f,protocol=pickle.HIGHEST_PROTOCOL)
