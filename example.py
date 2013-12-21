from __future__ import division

import numpy as np
from numpy import newaxis as na
from matplotlib import pyplot as plt
# plt.ion()

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.basic.distributions import NegativeBinomialIntegerRVariantDuration

import autoregressive.models as m
from autoregressive.distributions import MNIW

###############
#  make data  #
###############

data = np.hstack((
    np.linspace(5,5,20),
    np.linspace(5,10,5),
    np.linspace(10,10,10),
    np.linspace(10,5,5),
    np.linspace(5,5,20),
    np.linspace(5,10,5),
    np.linspace(10,10,10),
    ))

data += 0.1*np.random.normal(size=data.shape)

# data = data[:,na]

plt.figure()
plt.plot(data)
plt.title('data')

##################
#  set up model  #
##################

Nmax = 20
model = m.ARHSMMIntNegBinVariant(
        nlags=1,
        alpha=4.,gamma=4.,init_state_concentration=4.,
        obs_distns=[MNIW(dof=2,S=np.eye(1),M=np.zeros((1,2)),K=np.eye(2),affine=True)
            for state in range(Nmax)],
        dur_distns=[NegativeBinomialIntegerRVariantDuration(
            r_discrete_distn=np.ones(10), # can learn to be an HMM when r=1
            alpha_0=9,beta_0=1, # average geometric success probability 1/(9+1)
            ) for state in range(Nmax)],
        )

model.add_data(data)

###############
#  INFERENCE  #
###############

for itr in progprint_xrange(50):
    model.resample_model()

plt.figure()
model.plot()

plt.show()
