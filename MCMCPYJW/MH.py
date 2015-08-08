# -*- coding: utf-8 -*-
"""
Simple script for regular MH random walk


Created on Sun Aug  9 00:00:41 2015

@author: jonaswallin
"""
from . import AMCMC
import numpy as np
import numpy.random as npr

class MH(object):
    """
        An Metropolis hastings algorithm wrapper
        
    """
    
    
    def __init__(self, mh_within_gibbs = False):
        """
            *mh_within_gibbs*  is the algorithm used within Gibbs?
        """
        
        self.use_amcmc_rr     = False
        self.mh_within_gibbs  = mh_within_gibbs
        
        self.count   = 0
        
        self._llik_old = - np.inf
        
        self.sigma = 1.
        self.x0 = None
        self.pi = None
        self.accept  =  0
        

    def set_x0(self, x0):
        """
            setting starting value
        """
        self.x = np.zeros_like(x0)
        self.x[:] = x0[:]
        
        if self.pi is not None:
            self._llik_old = self.pi(self.x)

    def set_ldensity(self, pi):
        """
            setting the function to evalute the log density
            *pi* density function
        """
        
        self.pi = pi
        
    def set_amcmc_rr(self, **kwargs):
        """
            using the algorithm from AMCMC.py to set sigma
            however the inital sigma will be taken from the MH object!
        """
        
        self.amcmc_rr = AMCMC.Amcmc_RR(**kwargs)
        self.amcmc_rr.sigma = self.sigma
        self.use_amcmc_rr = True        
        
        
    def __call__(self):
        """
            running an iteration of the MH randomwalk
        """
        self.count   += 1
        self.accept  =  0
        
        xs = self.x + self.sigma * npr.randn(1)
        
        if self.mh_within_gibbs:
            self._llik_old = self.pi(self.x)
            
        _llik_star = self.pi(xs)
        
        if np.log(npr.rand(1)) < _llik_star - self._llik_old:
            self._llik_old = _llik_star
            self.x = xs
            self.accept = 1
            
        if self.use_amcmc_rr:
            self.sigma = self.amcmc_rr(self.accept)
            
        
        _x    = np.zeros_like(self.x)
        _x[:] = self.x[:]
        
        return _x