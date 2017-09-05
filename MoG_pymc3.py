import theano
import theano.tensor as tt
import pickle
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.math import logsumexp
from theano.tensor.nlinalg import det
from theano.tensor.slinalg import solve_lower_triangular
import matplotlib.pyplot as plt
import seaborn as sns

data='1k'
K = 12
lag = 2
with open('../data/VAR_blocks'+data+'.pickle','rb') as f:
    X,Y = pickle.load(f)
X,Y = X[lag],Y[lag]

n_samples = X.shape[0]

D = 2*(5+2*lag)

# Log likelihood of normal distribution
def logp_normal(mu, tau, value):
    # log probability of individual samples
    dim = tau.shape[0]
    delta = lambda mu: value - mu
    return -0.5 * (dim * tt.log(2 * np.pi) + tt.log(1/det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

# Log likelihood of Gaussian mixture distribut ion
def logp_gmix(mus, pi, taus):
    def logp_(value):
        logps = [tt.log(pi[k]) + logp_normal(mus[k], taus[k], value)
                 for k in range(K)]
        return tt.sum(logsumexp(tt.stacklists(logps)[:,:n_samples], axis=0))
    return logp_



with pm.Model() as model:
    # Hyperpriors for mixture components' means/cov matrices
    mus = [pm.MvNormal('mu_'+str(k),
                        mu=np.zeros(D,dtype=np.float32),
                        cov=10000*np.eye(D),
                        shape=(D,))
            for k in range(K)]

    taus = []
    sd_dist = pm.HalfCauchy.dist(beta=10000)
    for k in range(K):
        packed_chol = pm.LKJCholeskyCov('packed_chol'+str(k),
                        n=D,
                        eta=1,
                        sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(n=D, packed=packed_chol)
        invchol = solve_lower_triangular(chol,np.eye(D))
        taus.append(tt.dot(invchol.T,invchol))


    # Mixture density
    pi = pm.Dirichlet('pi',a=np.ones(K),shape=(K,))
    B = pm.DensityDist('B', logp_gmix(mus,pi,taus), shape=(n_samples,D))

    Y_hat = tt.sum(X[:,:,np.newaxis]*B.reshape((n_samples,D//2,2)),axis=1)

    # Model error
    err = pm.HalfCauchy('err',beta=10)
    # Data likelihood
    Y_logp = pm.MvNormal('Y_logp', mu=Y_hat, cov=err*np.eye(2), observed=Y)

with model:
    approx = pm.variational.inference.fit(
                    n=1000,
                    obj_optimizer=pm.adagrad(learning_rate=0.1)
                    )

plt.figure()
plt.plot(approx.hist)
plt.savefig('../images/Mo_'+str(K)+'G_ADVI'+data+'_lag'+str(lag)+'convergence.png')
gbij = approx.gbij
means = gbij.rmap(approx.mean.eval())

with open('../data/Mo'+str(K)+'G_results'+data+'_lag'+str(lag)+'.pickle','wb') as f:
    pickle.dump(means,f,protocol=pickle.HIGHEST_PROTOCOL)
