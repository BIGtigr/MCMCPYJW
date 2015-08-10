# -*- coding: utf-8 -*-
"""
    Simple sampling using MH from a multivariate Normal
Created on Sun Aug  9 11:31:25 2015

@author: jonaswallin
"""

import numpy.random as npr
import numpy as np
import MCMCPYJW
import matplotlib.pyplot as plt

SIGMA = np.array([[2, 1.], [1., 2.]])
Q     = np.linalg.inv(SIGMA)


def log_pi(x):
    
    return np.double(-  np.dot(x.transpose(), np.dot(Q, x) )/2. )
    
    
SIM    = 4000



x = np.zeros((2,1))
x_vec = np.zeros(SIM)
acc_vec = np.zeros(SIM)
sigma  = 2.


mh_object  = MCMCPYJW.MH()
mh_object.sigma = sigma
mh_object.set_amcmc_rr()

mh_object.set_ldensity(log_pi)
mh_object.set_x0( x)
mh_object.set_amcmc_Sigma(n_preupdate = 100)

for i in xrange(SIM):
    
    x_vec[i] = mh_object()[0]
    acc_vec[i] = mh_object.accept  
    
    
plt.figure()
plt.hist(x_vec,50)
plt.figure()
plt.plot(np.cumsum(acc_vec)/ np.cumsum(np.ones(SIM)))
plt.plot(mh_object.amcmc_rr.amcmc_desired_accept * np.ones(SIM),'r-')
