from __future__ import division

import numpy as np
from numpy import newaxis as na
from matplotlib import pyplot as plt
# plt.ion()

import pyhsmm
from pyhsmm.util.text import progprint_xrange

import autoregressive.models as m
import autoregressive.distributions as d

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

##################
#  set up model  #
##################

Nmax = 20
model = m.ARHSMM(
        nlags=1,
        alpha=4.,gamma=4.,init_state_concentration=4.,
        obs_distns=[d.MNIW(dof=2,S=np.eye(1),M=np.zeros((1,2)),K=np.eye(2),affine=True)
            for state in range(Nmax)],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=1*15,beta_0=1)
            for state in range(Nmax)],
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
