# -*- coding: utf-8 -*-
"""
Example
-------
Defining a custom model ::
    
    @code...

"""

# Imports

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from  gpytorch.distributions import MultivariateNormal

from sklearn.ensemble import RandomForestRegressor

from pd_utils import to_torch

# Gaussian Process model with Matern kernel
        
class gp_model(ExactGP):
    """Base gaussian process model.
    
    GPyTorch's exact gaussian process regression with Matern kernels. 
    """
    
    def __init__(self, X, y, likelihood, gpu=False, nu=2.5,
                 lengthscale_prior=None, outputscale_prior=None
                 ):
        """        
        Parameters
        ----------
        X : torch.tensor
            Training domain values.
        y : torch.tensor 
            Training response values.
        likelihood : (gpytorch.likelihoods)
            Model likelihood.
        gpu : bool 
            Use GPUs (if available) to run gaussian process computations. 
        nu : float 
            Matern kernel parameter. Options: 0.5, 1.5, 2.5.
        lengthscale_prior : [gpytorch.priors, init_value] 
            GPyTorch prior object and initial value. Sets a prior over length 
            scales.
        outputscale_prior : [gpytorch.priors, init_value] 
            GPyTorch prior object and initial value. Sets a prior over output s
            cales.
        """
        
        super(gp_model, self).__init__(X, y, likelihood)
        
        # ARD
        num_dims = len(X) if len(X) == 0 else len(X[0])
        
        # Base kernel
        if lengthscale_prior == None:
            kernel = MaternKernel(nu=nu, 
                               ard_num_dims=num_dims)
        else:
            kernel = MaternKernel(nu=nu, 
                               ard_num_dims=num_dims,
                               lengthscale_prior=lengthscale_prior[0])
        
        # Mean
        self.mean_module = ConstantMean()
        
        # Output scale
        if outputscale_prior == None:
            self.covar_module = ScaleKernel(kernel)
        else:
            self.covar_module = ScaleKernel(
                                kernel,
                                outputscale_prior=outputscale_prior[0])
        
        # Set initial values
        if lengthscale_prior != None:
            try:
                ls_init = to_torch(lengthscale_prior[1], gpu=gpu)
                self.covar_module.base_kernel.lengthscale = ls_init
            except:
                uniform = to_torch(lengthscale_prior[1], gpu=gpu)
                ls_init = torch.ones(num_dims) * uniform
                self.covar_module.base_kernel.lengthscale = ls_init
            
        if outputscale_prior != None:
            os_init = to_torch(outputscale_prior[1], gpu=gpu)
            self.covar_module.outputscale = os_init
        
    # forward prediction
    def forward(self, x):
        """        
        Parameters
        ----------
        x : torch.tensor
            Domain points which define multivariate normal distribution.
        
        Returns
        ----------
        gpytorch.MultivariateNormal
            Multivariate normal distribution.
        """ 
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return MultivariateNormal(mean_x, covar_x) 

# Random Forest model

class random_forest(RandomForestRegressor):
    """Random Forest regression algorithm.
    
    Class provides a method for specifiying RF model hyperparameters 
    using scikit-learn's ensemble package. The preset hyperparameter values 
    are selected via benchmarking studies on parameterized chemical reaction 
    data.
    """
    
    def __init__(self, n_jobs=-1, random_state=10, n_estimators=500,
				max_features='auto', max_depth=None, min_samples_leaf=1,
				min_samples_split=2):
        """        
        Parameters
        ----------
        n_jobs : int 
            Number of processers to use.
        random_state : int
            Insures identical data returns an identical ensemble of regression 
            trees.
        n_estimators : int
            Number of weak estimators to include in ensemble.
        max_features : 'auto', int
            Maximum number of features to consider per node in model training.
        max_depth : None, int
            Maximum depth of individual trees. 
        min_samples_leaf : int
            Minimum number of samples required at each leaf node in model 
            training.
        min_samples_split : int
            Minimum number of samples to require for a node to split.
        """
        
        super(random_forest, self).__init__(n_jobs=n_jobs, 
                                            random_state=random_state, 
                                            n_estimators=n_estimators, 
                                            max_features=max_features, 
                                            max_depth=max_depth, 
                                            min_samples_leaf=min_samples_leaf,
                                            min_samples_split=min_samples_split
                                            )

def fast_computation(fastQ):
    """Function for turning on/off GPyTorch fast computation features."""
    
    gpytorch.settings.fast_pred_var._state = fastQ
    gpytorch.settings.fast_pred_samples._state = fastQ 
    gpytorch.settings.fast_computations.covar_root_decomposition._state = fastQ
    gpytorch.settings.fast_computations.log_prob._state = fastQ
    gpytorch.settings.fast_computations.solves._state = fastQ
    gpytorch.settings.deterministic_probes._state = fastQ
    gpytorch.settings.memory_efficient._state = fastQ
    
    
    
    
    
    
    
    