# -*- coding: utf-8 -*-
"""
    Extermly simple script for sampling a N(0,1) random variable
    using AMCMC MH uses matplotlib
    Showing that MCMCPYJW.Amcmc_RR converges to the desired accptance rate

Created on Sat Aug  8 23:40:59 2015

@author: jonaswallin
"""
import numpy.random as npr
import numpy as np
import MCMCPYJW
import matplotlib.pyplot as plt


def log_pi(x):
    
    return - (x - 0.)**2.
    
    
SIM    = 15000


x = 0.
x_vec = np.zeros(SIM)
acc_vec = np.zeros(SIM)
sigma  = 2.
amcmc_obj  = MCMCPYJW.Amcmc_RR(sigma = sigma)

for i in xrange(SIM):

    xs = x + sigma * npr.randn(1)
    accept = 0
    
    if np.log(npr.rand(1)) < log_pi(xs) - log_pi(x):
        accept = 1
        x = xs
    sigma = amcmc_obj(accept)
    acc_vec[i] = accept
    x_vec[i] = x

plt.hist(x_vec,50)
plt.figure()
plt.plot(np.cumsum(acc_vec)/ np.cumsum(np.ones(SIM)))
plt.plot(amcmc_obj.amcmc_desired_accept * np.ones(SIM),'r-')


###
# doing equivalent thing with MH object

mh_object  = MCMCPYJW.MH()
mh_object.sigma = sigma
mh_object.set_amcmc_rr()

mh_object.set_ldensity(log_pi)
mh_object.set_x0(np.array([0.]))

for i in xrange(SIM):
    
    x_vec[i] = mh_object()
    acc_vec[i] = mh_object.accept  
    
plt.figure()
plt.hist(x_vec,50)
plt.figure()
plt.plot(np.cumsum(acc_vec)/ np.cumsum(np.ones(SIM)))
plt.plot(amcmc_obj.amcmc_desired_accept * np.ones(SIM),'r-')