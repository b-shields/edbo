############################################################################## Setup
"""
Acquisition Functions Test Parameters:
(0) Random Forest model.
(1) 1D objective.
(2) Initialize with random data.
(3) Test predictions, variance estimation, and sampling.
(4) Run single iteration of each acquisition function.
"""

# Imports

import numpy as np
import pandas as pd
from edbo.bro import BO
from edbo.pd_utils import to_torch
from edbo.models import RF_Model
import random

############################################################################## Test Functions

# Objective

def f(x):
    """Noise free objective."""
    
    return np.sin(10 * x) * x * 100

# Test a precomputed objective

def BO_pred(acq_func, plot=False, return_='pred', append=False, init='external', fast_comp=True):
    
    # Experiment index
    X = np.linspace(0,1,1000)
    exindex = pd.DataFrame([[x, f(x)] for x in X], columns=['x', 'f(x)'])
    training_points = [50, 300, 500, 900]
    
    # Instatiate BO class
    bo = BO(exindex=exindex,
            domain=exindex.drop('f(x)', axis=1),
            results=exindex.iloc[training_points],
            model=RF_Model,
            acquisition_function=acq_func,
            init_method=init,
            batch_size=random.sample([1,2,3,4,5,6,7,8,9,10],1)[0])
    
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
        return True
    
    # Check predictive postrior variance
    elif return_ == 'var':
        
        try:
            bo.model.predict(to_torch(bo.obj.domain))                          # torch.tensor
            bo.model.predict(bo.obj.domain.values)                             # numpy.array
            bo.model.predict(list(bo.obj.domain.values))                       # list
            bo.model.predict(exindex.drop('f(x)', axis=1))                     # pandas.DataFrame
        except:
            return False
        
        var = bo.model.variance(bo.obj.domain.iloc[[32]])
        return True
    
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
    
    elif return_ == 'simulate':
        
        if init != 'external':
            bo.init_seq.batch_size = random.sample([2,3,4,5,6,7,8,9,10],1)[0]
        
        bo.simulate(iterations=5)
        bo.plot_convergence()
        bo.model.regression()
        
        return True
    
    elif return_ == 'none':
        return True

############################################################################## Tests

# Test predicted mean, variance, and sampling

def test_BO_pred_mean():
    assert BO_pred('MeanMax', return_='pred')
    
def test_BO_pred_var():
    assert BO_pred('MeanMax', return_='var')

def test_BO_sample():
    assert BO_pred('MeanMax', return_='sample')

# Test different acquisition functions

def test_BO_EI():
    assert BO_pred('EI', return_='none')

def test_BO_PI():
    assert BO_pred('PI', return_='none')
    
def test_BO_UCB():
    assert BO_pred('UCB', return_='none')

def test_BO_rand():
    assert BO_pred('rand', return_='none')

def test_BO_MeanMax():
    assert BO_pred('MeanMax', return_='none')

def test_BO_VarMax():
    assert BO_pred('VarMax', return_='none')

def test_BO_EI_TS():
    assert BO_pred('EI-TS', return_='none')

def test_BO_PI_TS():
    assert BO_pred('PI-TS', return_='none')

def test_BO_UCB_TS():
    assert BO_pred('UCB-TS', return_='none')

def test_BO_rand_TS():
    assert BO_pred('rand-TS', return_='none')

def test_BO_MeanMax_TS():
    assert BO_pred('MeanMax-TS', return_='none')

def test_BO_VarMax_TS():
    assert BO_pred('VarMax-TS', return_='none')

def test_BO_eps_greedy():
    assert BO_pred('eps-greedy', return_='none')
