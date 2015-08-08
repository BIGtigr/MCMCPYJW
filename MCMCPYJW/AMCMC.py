# -*- coding: utf-8 -*-
"""
Various Adaptive MCMC scripts


Created on Sat Aug  8 23:21:39 2015

@author: jonaswallin
"""
import numpy as np

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
        self.amcmc_batch          = batch
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
        
        if (self.amcmc_count +1) % self.amcmc_batch == 0:

            delta = np.min([self.amcmc_delta_max, 
                            (self.count_mcmc/self.amcmc_batch)**(-self.amcmc_delta_rate)
                            ])
            
            if self.amcmc_accept / self.amcmc_count > self.amcmc_desired_accept:
                self.sigma *= np.exp(delta) 
            else:
                self.sigma /= np.exp(delta)
            
            self.amcmc_count  = 0.
            self.amcmc_accept = 0.
            
        return self.sigma
    
