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
from edbo.bro import BO_express
from edbo.pd_utils import to_torch, torch_to_numpy
import matplotlib.pyplot as plt
import random

############################################################################## Test Functions

# Objective

def random_result(*kwargs):
    """Random objective."""
    
    return round(random.random(),3) * 100

# Test a precomputed objective

def BO_pred(acq_func, plot=False, return_='pred', append=False, init='rand'):
    
    # Define reaction space and auto-encode
    n_ligands = random.sample([3,4,5,6,7,8], 1)[0]
    ligands = pd.read_csv('data\ligands.csv').sample(n_ligands).values.flatten()
    bases = ['DBU', 'MTBD', 'potassium carbonate', 'potassium phosphate', 'potassium tert-butoxide']
    reaction_components={'aryl_halide':['chlorobenzene','iodobenzene','bromobenzene'],
                     'base':bases,
                     'solvent':['THF', 'Toluene', 'DMSO', 'DMAc'],
                     'ligand':ligands,
                     'concentration':[0.1, 0.2, 0.3],
                     'temperature': [20, 30, 40]
                     }
    encoding={
          'aryl_halide':'resolve',
          'base':'resolve',
          'solvent':'resolve',
          'ligand':'mordred',
          'concentration':'numeric',
          'temperature':'numeric'}
    
    # Instatiate BO class
    bo = BO_express(reaction_components=reaction_components, 
                    encoding=encoding,
                    acquisition_function=acq_func,
                    init_method=init,
                    batch_size=random.sample(range(30),1)[0],
                    computational_objective=random_result,
                    target='yield')
    
    bo.init_sample(append=True)
    bo.run(append=append)
    
    # Check prediction
    if return_ == 'pred':
        
        try:
            bo.model.predict(to_torch(bo.obj.domain))                          # torch.tensor
            bo.model.predict(bo.obj.domain.values)                             # numpy.array
            bo.model.predict(list(bo.obj.domain.values))                       # list
            bo.model.predict(bo.obj.domain)                                    # pandas.DataFrame
        except:
            return False
        
        return True
    
    # Check predictive postrior variance
    elif return_ == 'var':
        
        try:
            bo.model.predict(to_torch(bo.obj.domain))                          # torch.tensor
            bo.model.predict(bo.obj.domain.values)                             # numpy.array
            bo.model.predict(list(bo.obj.domain.values))                       # list
            bo.model.predict(bo.obj.domain)                                    # pandas.DataFrame
        except:
            return False
        
        return True
    
    # Make sure sampling works with tensors, arrays, lists, and DataFrames
    elif return_ == 'sample':
        try:
            bo.model.sample_posterior(to_torch(bo.obj.domain))                 # torch.tensor
            bo.model.sample_posterior(bo.obj.domain.values)                    # numpy.array
            bo.model.sample_posterior(list(bo.obj.domain.values))              # list
            bo.model.sample_posterior(bo.obj.domain)                           # pandas.DataFrame
            return True
        except:
            return False
        
    # Plot model
    elif return_ == 'plot':
        mean = bo.obj.scaler.unstandardize(bo.model.predict(bo.obj.domain))
        std = bo.obj.scaler.unstandardize(np.sqrt(bo.model.variance(bo.obj.domain))) * 2
        samples = bo.obj.scaler.unstandardize(bo.model.sample_posterior(bo.obj.domain, batch_size=3))

        plt.figure(1, figsize=(6,6))

        # Model mean and standard deviation
        plt.subplot(211)
        plt.plot(range(len(mean)), mean, label='GP')
        plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.4)
        # Known results and next selected point
        plt.scatter(bo.obj.results_input().index.values, bo.obj.results_input()['yield'], color='black', label='known')
        plt.ylabel('f(x)')
        # Samples
        plt.subplot(212)
        for sample in samples:
            plt.plot(range(len(mean)), torch_to_numpy(sample))
        plt.xlabel('x')
        plt.ylabel('Posterior Samples')
        plt.show()

        return True
    
    elif return_ == 'simulate':
        
        if init != 'external':
            bo.init_seq.batch_size = random.sample([2,3,4,5,6,7,8,9,10],1)[0]
        
        bo.simulate(iterations=3)
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




