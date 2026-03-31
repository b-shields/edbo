# -*- coding: utf-8 -*-

# Imports

import numpy as np
import torch
import gpytorch
from gpytorch.priors import GammaPrior

from copy import deepcopy

# Disctionary of distributions for randomm initialization

def build_dist_dict(noise_prior, outputscale_prior, lengthscale_prior):
    """
    Build a dictionary of distributions to sample for random restarts.
    """

    if noise_prior == None:
        noise_dist = GammaPrior(1.5,0.5)
    else:
        noise_dist = noise_prior[0]

    if outputscale_prior == None:
        output_dist = GammaPrior(3, 0.5)
    else:
        output_dist = outputscale_prior[0]
    
    if lengthscale_prior == None:
        lengthscale_dist = GammaPrior(3,0.5)
    else:
        lengthscale_dist = lengthscale_prior[0]

    distributions = {'likelihood.noise_covar.raw_noise': noise_dist,
                 'covar_module.raw_outputscale': output_dist,
                 'covar_module.base_kernel.raw_lengthscale': lengthscale_dist}
    
    return distributions

# Randomly set parmeters for model based on distributions

def set_init_params(dictionary, distributions, seed=0):
    """
    Generate a new random state dictionary with entries drawn from the list of 
    distributions.
    """
    
    dict_copy = deepcopy(dictionary)
    
    for key in distributions:
        
        # Get parameter values for dict entry
        params = dictionary[key]
        
        # Get distribution
        dist = distributions[key]
        
        # Generate inital points from distribution
        torch.manual_seed(seed)
        new_params = dist.expand(params.shape).sample().log()
        
        # Overwrite entry in copy
        dict_copy[key] = new_params
    
    return dict_copy

# Optimize a model via MLE

def optimize_mll(model, likelihood, X, y, learning_rate=0.1, 
                 n_restarts=0, training_iters=100, noise_prior=None,
                 outputscale_prior=None, lengthscale_prior=None):
    
    # Model and likelihood in training mode
    model.train()
    likelihood.train()
    
    # Use ADAM
    optimizer = torch.optim.Adam(
                [{'params': model.parameters()}, ],
                lr=learning_rate
                )
    # Marginal log likelihood loss                          
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Dictionary of distributions to draw random restarts from
    dist_dict = build_dist_dict(noise_prior, outputscale_prior, lengthscale_prior)
    
    # Restart optimizer with random inits drawn from priors
    states = []
    loss_list = []
    min_loss_list = []
    for restart in range(n_restarts + 1):
        
        step_losses = []
        # Optimization
        for i in range(training_iters):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, y)
            step_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        states.append(deepcopy(mll.model.state_dict()))
        loss_list.append(step_losses)
        min_loss_list.append(loss.item())
    
        new_state = set_init_params(states[0], dist_dict, seed=restart)
        mll.model.load_state_dict(new_state)

    # Set to best state
    mll.model.load_state_dict(states[np.argmin(min_loss_list)])
    
    return loss_list