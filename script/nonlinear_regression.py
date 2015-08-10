# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:37:43 2015

@author: jonaswallin
"""
from __future__ import division
import numpy as np
import numpy.random as npr
import MCMCPYJW
import matplotlib.pyplot as plt

n = 100
sigma = .1
X = npr.random(n)
Y = 1 - X**2 + sigma*npr.randn(n)

fig, ax = plt.subplots()
ax.plot(X,Y,'.')


def log_pi(beta, sigma, X , Y ):
    """
        The log likelihood
    """
    n = len(Y)
    return n * np.log(sigma) - np.sum( (Y - beta[0] - beta[1] * X**beta[2])**2)/sigma**2
    

f = lambda x: log_pi(x, sigma, X, Y)




SIM    = 50000



x = np.zeros((3,1))
x_vec = np.zeros((SIM,3))
acc_vec = np.zeros(SIM)
sigma  = .1


mh_object  = MCMCPYJW.MH()
mh_object.sigma = sigma
mh_object.set_amcmc_rr()

mh_object.set_ldensity(f)
mh_object.set_x0( x)
mh_object.set_amcmc_Sigma(n_preupdate = 100)

for i in xrange(SIM):
    
    x_vec[i,:] = mh_object().reshape(3)
    acc_vec[i] = mh_object.accept  
    
    
plt.figure()
f_, ax2 = plt.subplots(3, 1)
ax2[0].hist(x_vec[:,0],50)
ax2[1].hist(x_vec[:,1],50)
ax2[2].hist(x_vec[:,2],50)
ax.plot(X, mh_object.x[0] + mh_object.x[1]*X**mh_object.x[2],'r.')
beta = mh_object.x
print('r2 = {resid}'.format( resid = np.sum( (Y - beta[0] - beta[1] * X**beta[2])**2)))
print('r2 = {resid}'.format( resid = np.sum( (Y - 1 +  X**2)**2)))