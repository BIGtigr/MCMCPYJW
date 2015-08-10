# -*- coding: utf-8 -*-
"""
Various Adaptive MCMC scripts


Created on Sat Aug  8 23:21:39 2015

@author: jonaswallin
"""
from __future__ import division
import numpy as np


class Amcmc_empericalcov(object):
    """
        Adaptive MCMC script that uses the emperical covariance matrix as scaling
        matrix Using the adaptive mcmc method in Harrio et al.
    """
    
    
    def __init__(self, n_preupdate = 10, batch = 10):
        """
            *n_preupdate*  how many iteration to run before starting to update
                           the covariance matrix 
            *batch*        how often updating the Cholesky factor
        """

        self.n_preupdate  = n_preupdate
        self.batch        = batch
        self.reset_counts()
        
        self.Sigma = None # the covariance matrix
        self.mu    = None # the mean
        self.R     = None # the cholesky factor of the covariance
        
        
    def reset_counts(self):
        """
            reseting the counter used to updated the covariance matrices
            *iteration_counter* number of runs of Metropolis Hastings algorithm
        """        
        
        self.iteration_counter = 0
        
    def set_Sigma(self, Sigma):
        """
            Setting Sigma
        """
        self.Sigma = np.zeros_like(Sigma)
        self.Sigma[:] = Sigma[:]
        
    def set_mu(self, mu):
        """
            Setting mu
        """
        self.mu    = np.zeros_like(mu)
        self.mu[:] = mu[:]
        
    
    def init(self, dim):
        """
            setting up basic data
        """
        
        self.mu = np.zeros(  (dim, 1))
        self.Sigma = np.eye( dim)
        self.R     = np.eye( dim)
        
    def reset_param(self):
        """
            Reseting mu and Sigma
        """
        
        self.mu    *= 0.
        self.Sigma *= 0.
        self.Sigma += np.eye(self.Sigma.shape)
        self.R     *= 0.
        self.R     += np.eye(self.Sigma.shape)
        
        
    def __call__(self, x):
        """
            updating the cholesky factor
        """
        
        self.iteration_counter += 1.
        w = 1./(self.iteration_counter + 1)
        
        self.mu *= 1 - w
        self.mu += w * x.reshape(self.mu.shape) 
        
        x_mu = np.sqrt(w) * (x.reshape(self.mu.shape) - self.mu)

        self.Sigma *= 1 - w
        self.Sigma += np.outer(x_mu, x_mu)
        
        condition_1 = (self.iteration_counter +1) % self.batch == 0
        condition_2 = self.iteration_counter > self.n_preupdate 
        if (condition_1 and condition_2):
            R_ = np.linalg.cholesky(self.Sigma)
            self.R[:] = R_[:]

        
        
        
        
        

class Amcmc_RR(object):
    """
        Adaptive MCMC script to set sigma infront of the covmatrix in MH or MALA
        taken from Roberts o Rosenthal Examples of Adaptive MCMC
    """
    
    def __init__(self, sigma = 1., batch = 50, accpate = 0.2, delta_rate = .5):
        """
            *sigma*        - (double) the scaling coeffient
            *batch*        - (int) how often to update sigma_MCMC
            *accpate*      - [0,1] desired accpance rate (0.2)
            *delta_rate*   - [0,1] updating ratio for the amcmc
        """
        
        self.amcmc_delta_max      = 0.1
        self.amcmc_desired_accept = accpate
        self.batch                = batch
        self.amcmc_delta_rate     = delta_rate
        self.sigma                = sigma
        self.reset()
        
    
    def reset(self):
        """
            Clearing all counts
        """        
        self.amcmc_count  = 0.
        self.amcmc_accept = 0.
        self.count_mcmc   = 0.
        
    def __call__(self, accepted):
        """
            taking in the 0/1 variable if the MH sample where *accepted* or not
        """
        
        self.amcmc_accept += accepted
        self.count_mcmc  += 1.
        self.amcmc_count += 1.
        
        if (self.amcmc_count +1) % self.batch == 0:

            delta = np.min([self.amcmc_delta_max, 
                            (self.count_mcmc/self.batch)**(-self.amcmc_delta_rate)
                            ])
            
            if self.amcmc_accept / self.amcmc_count > self.amcmc_desired_accept:
                self.sigma *= np.exp(delta) 
            else:
                self.sigma /= np.exp(delta)
            
            self.amcmc_count  = 0.
            self.amcmc_accept = 0.
            
        return self.sigma
    
