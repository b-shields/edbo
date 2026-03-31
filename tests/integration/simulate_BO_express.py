############################################################################## Setup
"""
ND Bayesian Optimization Test:
(0) Gaussian process model.
(1) Gemerate ND objective using BO_express autobuilding features.
(2) Run simulations with select acquisition functions.
"""

# Imports

import pandas as pd
from edbo.bro import BO_express
from edbo.pd_utils import to_torch
import random

############################################################################## Test Functions

# Objective

def random_result(*kwargs):
    """Random objective."""
    
    return round(random.random(),3) * 100

# Test a precomputed objective

def BO_pred(acq_func, build_type='no spec', plot=False, return_='pred', append=False, init='rand'):
    
    # Define reaction space and auto-encode
    
    aryl_halides = ['chlorobenzene','iodobenzene','bromobenzene']
    n_ligands = random.sample([3,4,5,6,7,8], 1)[0]
    ligands = pd.read_csv('ligands.csv').sample(n_ligands).values.flatten()
    bases = ['DBU', 'MTBD', 'potassium carbonate', 'potassium phosphate', 'potassium tert-butoxide']
    solvents = ['THF', 'Toluene', 'DMSO', 'DMAc']
    concentrations = [0.1, 0.2, 0.3]
    temperatures =  [20, 30, 40]
    
    reaction_components = {'aryl_halide':aryl_halides,
                     'base':bases,
                     'solvent':solvents,
                     'ligand':ligands,
                     'concentration':concentrations,
                     'temperature': temperatures
                     }
    
    # Automatically OHE everything
    if build_type == 'no spec':
        encoding = {}
        matrices = {}
        
    # Build based on encoding specifications
    elif build_type == 'spec':
        encoding = {
          'aryl_halide':'resolve',
          'base':'resolve',
          'solvent':'resolve',
          'ligand':'mordred',
          'concentration':'numeric',
          'temperature':'numeric'}
        matrices = {}
        
    # Include one component as a external matrix
    elif build_type == 'external':
        encoding={'aryl_halide':'resolve', 'ligand':'smiles'}
        s = pd.DataFrame([['THF', 1, 2], 
                         ['Toluene', 3, 4],
                         ['DMSO', 5, 6],
                         ['DMAc', 7, 8]],
                         columns=['solvent', 'solvent_d1', 'solvent_d2'])
        ar = pd.DataFrame([['chlorobenzene', 1,2,3],
                           ['iodobenzene',4,5,6],
                           ['bromobenzene',7,8,9]],
                          columns=['aryl_halide', 'ar_d1', 'ar_d2', 'ar_d3'])
        matrices = {'aryl_halide': ar, 'solvent':s}
    
    # Instatiate BO class
    bo = BO_express(reaction_components=reaction_components, 
                    encoding=encoding,
                    descriptor_matrices=matrices,
                    acquisition_function=acq_func,
                    init_method=init,
                    batch_size=random.sample(range(1,20),1)[0],
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
    
    elif return_ == 'simulate':
        
        if init != 'external':
            bo.init_seq.batch_size = random.sample([2,3,4,5,6,7,8,9,10],1)[0]
        
        bo.simulate(iterations=3)
        
        return True
    
    elif return_ == 'none':
        return True

############################################################################## Tests

# Test simulation with different acquisition functions

def test_BO_TS():
    assert BO_pred('TS', return_='simulate')

def test_BO_EI():
    assert BO_pred('EI', return_='simulate')

def test_BO_MeanMax():
    assert BO_pred('MeanMax', return_='simulate')

def test_BO_UCB_TS():
    assert BO_pred('UCB-TS', return_='simulate')

