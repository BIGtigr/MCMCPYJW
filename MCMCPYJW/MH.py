# -*- coding: utf-8 -*-
"""
Simple script for regular MH random walk


Created on Sun Aug  9 00:00:41 2015

@author: jonaswallin
"""
from . import AMCMC
import numpy as np
import numpy.random as npr

#TODO: Add a Sigma that is used as the covariance matrix infront
#TODO: Add a wrapper so that each time Sigma is updated preform 
#      an update of the cholesky factor
#TODO: Add an other adaptive scheme that estimates the covariance matrix

class MH(object):
    """
        An Metropolis hastings algorithm wrapper
        
        
        
    """
    
    
    def __init__(self, mh_within_gibbs = False):
        """
            *mh_within_gibbs*  is the algorithm used within Gibbs?
        """
        
        self.use_amcmc_rr     = False
        self.use_amcmc_Sigma  = False 
        self.mh_within_gibbs  = mh_within_gibbs
        
        self.count   = 0
        
        self.llik_x = - np.inf
        
        self.sigma      = 1.
        self.x          = None
        self.pi         = None
        self.accept     =  0
        self.Sigma      = None
        self.r_cholesky = None
        

    def set_x0(self, x0):
        """
            setting starting value
        """
        
        self.x = np.zeros_like(x0)
        self.x[:] = x0[:]
        
        if self.pi is not None:
            self.llik_x = self.pi(self.x)
        
        if self.use_amcmc_Sigma:
            self.amcmc_Sigma.init(self.x.shape[0])
            self.r_cholesky = self.amcmc_Sigma.R

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
        
    def set_amcmc_Sigma(self, **kwargs):
        
        
        self.amcmc_Sigma = AMCMC.Amcmc_empericalcov(**kwargs)
        self.use_amcmc_Sigma = True
        
        self.Sigma = True
        
        if self.x is not None:
            self.amcmc_Sigma.init(self.x.shape[0])
            
            self.r_cholesky = self.amcmc_Sigma.R
            
    
    def _sample_x_star(self, z = None):
        """
            generates a proposal used in MH iteration
        """
        if z is None:
            z = npr.randn(*self.x.shape)
        
        if self.Sigma == None:
            x_star = self.x + self.sigma * npr.randn(*self.x.shape)
        
        else:
            
            if self.r_cholesky is None:
                self.r_cholesky = np.linalg.cholesky(self.Sigma)
                
            x_star = self.x + self.sigma * np.dot(self.r_cholesky, z)
        
        return x_star
    
    def __call__(self, z  = None):
        """
            running an iteration of the MH randomwalk
        """
        self.count   += 1
        self.accept  =  0
        
        x_star = self._sample_x_star(z)
        
        if self.mh_within_gibbs:
            self.llik_x = self.pi(self.x)
            
            
        llik_star = self.pi(x_star)
        
        if np.log(npr.rand(1)) < llik_star - self.llik_x:
            self.llik_x = llik_star
            self.x = x_star
            self.accept = 1
            
        if self.use_amcmc_rr:
            self.sigma = self.amcmc_rr(self.accept)
            
        if self.use_amcmc_Sigma:
            self.amcmc_Sigma(self.x)          
            
            
        _x    = np.zeros_like(self.x)
        _x[:] = self.x[:]
        
        return _x