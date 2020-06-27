############################################################################## Setup
"""
1D Bayesian Optimization Test:
(1) Gemerate 1D objective.
(2) Initialize with data.
(3) Test predictions, variance estimation, and sampling.
(4) Run single iteration of each acquisition function.
"""

# Imports

import numpy as np
import pandas as pd
from gpytorch.priors import GammaPrior
from edbo.bro import BO
from edbo.pd_utils import to_torch, torch_to_numpy
from edbo.models import RF_Model
import matplotlib.pyplot as plt
import random

############################################################################## Test Functions

# Objective

def f(x):
    """Noise free objective."""
    
    return np.sin(10 * x) * x * 100

# Test a precomputed objective

def BO_pred(acq_func, plot=False, return_='pred', append=False, init='external'):
    
    # Experiment index
    X = np.linspace(0,1,1000)
    exindex = pd.DataFrame([[x, f(x)] for x in X], columns=['x', 'f(x)'])
    training_points = [50, 300, 500, 900]
    
    # Instatiate BO class
    bo = BO(exindex=exindex,
            domain=exindex.drop('f(x)', axis=1),
            results=exindex.iloc[training_points],
            acquisition_function=acq_func,
            init_method=init,
            lengthscale_prior=[GammaPrior(1.2,1.1), 0.2],
            noise_prior=None,
            batch_size=random.sample([1,2,3,4,5,6,7,8,9,10],1)[0],
            fast_comp=False,
            model=RF_Model)
    
    bo.run(append=append)
    
    # Check prediction
    if return_ == 'pred':
        
        try:
            bo.model.predict(to_torch(bo.obj.domain))                          # torch.tensor
            bo.model.predict(bo.obj.domain.values)                             # numpy.array
            bo.model.predict(list(bo.obj.domain.values))                       # list
            bo.model.predict(exindex.drop('f(x)', axis=1))                     # pandas.DataFrame
        except:
            return False
        
        pred = bo.model.predict(bo.obj.domain.iloc[[32]])
        pred = bo.obj.scaler.unstandardize(pred)
        return (pred[0] + 1.11) < 0.1
    
    # Check predictive postrior variance
    elif return_ == 'var':
        
        try:
            bo.model.variance(to_torch(bo.obj.domain))                         # torch.tensor
            bo.model.variance(bo.obj.domain.values)                            # numpy.array
            bo.model.variance(list(bo.obj.domain.values))                      # list
            bo.model.variance(exindex.drop('f(x)', axis=1))                    # pandas.DataFrame
        except:
            return False
        
        var = bo.model.variance(bo.obj.domain.iloc[[32]])
        return (var[0] - 0.21) < 0.1
    
    # Make sure sampling works with tensors, arrays, lists, and DataFrames
    elif return_ == 'sample':
        try:
            bo.model.sample_posterior(to_torch(bo.obj.domain))                 # torch.tensor
            bo.model.sample_posterior(bo.obj.domain.values)                    # numpy.array
            bo.model.sample_posterior(list(bo.obj.domain.values))              # list
            bo.model.sample_posterior(exindex.drop('f(x)', axis=1))            # pandas.DataFrame
            return True
        except:
            return False
        
    # Plot model
    elif return_ == 'plot':
        next_points = bo.obj.get_results(bo.proposed_experiments)
        mean = bo.obj.scaler.unstandardize(bo.model.predict(bo.obj.domain))
        std = bo.obj.scaler.unstandardize(np.sqrt(bo.model.variance(bo.obj.domain))) * 2
        samples = bo.obj.scaler.unstandardize(bo.model.sample_posterior(bo.obj.domain, batch_size=3))

        plt.figure(1, figsize=(6,6))

        # Model mean and standard deviation
        plt.subplot(211)
        plt.plot(X, exindex['f(x)'], color='black')
        plt.plot(X, mean, label='GP')
        plt.fill_between(X, mean-std, mean+std, alpha=0.4)
        # Known results and next selected point
        plt.scatter(bo.obj.results_input()['x'], bo.obj.results_input()['f(x)'], color='black', label='known')
        plt.scatter(next_points['x'],next_points['f(x)'], color='red', label='next_experiments')
        plt.ylabel('f(x)')
        # Samples
        plt.subplot(212)
        for sample in samples:
            plt.plot(X, torch_to_numpy(sample))
        plt.xlabel('x')
        plt.ylabel('Posterior Samples')
        plt.show()

        return True
    
    elif return_ == 'simulate':
        
        if init != 'external':
            bo.init_seq.batch_size = random.sample([2,3,4,5,6,7,8,9,10],1)[0]
            
        bo.simulate(iterations=5)
        bo.plot_convergence()
        bo.model.regression()
        
        return True

############################################################################## Tests

# Test predicted mean and variance, sampling, and ploting

def test_BO_pred_mean_TS():
    assert BO_pred('TS', return_='pred')
    
def test_BO_var():
    assert BO_pred('TS', return_='var')

def test_BO_sample():
    assert BO_pred('TS', return_='sample')
    
def test_BO_plot():
    assert BO_pred('TS', return_='plot')
    
# Test simulations

def test_BO_simulate_TS():
    assert BO_pred('TS', return_='simulate')
    
def test_BO_simulate_EI():
    assert BO_pred('EI', return_='simulate')

# Init methods

def test_BO_simulate_kmeans():
    assert BO_pred('EI', return_='simulate', init='kmeans')

def test_BO_simulate_kmedoids():
    assert BO_pred('EI', return_='simulate', init='pam')
    
def test_BO_simulate_random():
    assert BO_pred('EI', return_='simulate', init='rand')

# Test different acquisition functions

def test_BO_pred_mean_EI():
    assert BO_pred('EI', return_='pred')

def test_BO_pred_mean_PI():
    assert BO_pred('PI', return_='pred')

def test_BO_pred_mean_UCB():
    assert BO_pred('UCB', return_='pred')

def test_BO_pred_mean_rand():
    assert BO_pred('rand', return_='pred')

def test_BO_pred_mean_MeanMax():
    assert BO_pred('MeanMax', return_='pred')

def test_BO_pred_mean_VarMax():
    assert BO_pred('VarMax', return_='pred')

def test_BO_pred_mean_EI_TS():
    assert BO_pred('EI-TS', return_='pred')

def test_BO_pred_mean_PI_TS():
    assert BO_pred('PI-TS', return_='pred')

def test_BO_pred_mean_UCB_TS():
    assert BO_pred('UCB-TS', return_='pred')

def test_BO_pred_mean_rand_TS():
    assert BO_pred('rand-TS', return_='pred')

def test_BO_pred_mean_MeanMax_TS():
    assert BO_pred('MeanMax-TS', return_='pred')

def test_BO_pred_mean_VarMax_TS():
    assert BO_pred('VarMax-TS', return_='pred')

def test_BO_pred_mean_eps_greedy():
    assert BO_pred('eps-greedy', return_='pred')




