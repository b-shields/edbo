# -*- coding: utf-8 -*-

# Imports

import pandas as pd

from .pd_utils import load_csv_or_excel
from .pd_utils import load_experiment_results
from .pd_utils import to_torch
from .math_utils import standard

# Objective function class

class objective:
    """Objective funciton data container and operations.
    
    Note
    ----
    Objective internally standardizes response values to zero mean and unit
    variance.
    """
    
    def __init__(self, 
                 results_path=None, results=pd.DataFrame(),
                 domain_path=None, domain=pd.DataFrame(),
                 exindex_path=None, exindex=pd.DataFrame(),
                 target=-1, gpu=False, computational_objective=None):
        """
        Parameters
        ----------
        results_path : str, optional
            Path to experimental results.
        results : pandas.DataFrame, optional
            Experimental results with X values matching the domain.
        domain_path : str, optional
            Path to experimental domain.
            
            Note
            ----
            A domain_path or domain are required.
            
        domain : pandas.DataFrame, optional
            Experimental domain specified as a matrix of possible 
            configurations.
        exindex_path : str, optional
            Path to experiment results index if available.
        exindex : pandas.DataFrame, optional
            Experiment results index matching domain format. Used as lookup 
            table for simulations.
        target : str
            Column label of optimization objective. If set to -1, the last 
            column of the DataFrame will be set as the target.
        gpu : bool
            Carry out GPyTorch computations on a GPU if available.
        computational_objective : function, optional
            Function to be optimized for computational objectives.
        """
        
        # Initialize
        
        self.results_path = results_path
        self.results = results
        self.domain_path = domain_path
        self.domain = domain
        self.exindex_path = exindex_path
        self.exindex = exindex
        self.target = target
        self.gpu = gpu
        self.computational_objective = computational_objective
        
        # Load domain
        
        if domain_path != None:
            self.domain = load_csv_or_excel(self.domain_path)
        
        self.domain.reset_index(drop=True)
        
        # Load results
        
        if type(self.results) == type(pd.DataFrame()) and len(self.results) > 0:
            if target == -1:
                self.target = self.results.columns.values[-1]
        
        elif results_path != None:
            data = load_experiment_results(self.results_path)
            self.results = data
            if target == -1:
                self.target = self.results.columns.values[-1]
        
        # Load experiment index
        
        if exindex_path != None:
            self.exindex = load_csv_or_excel(exindex_path)
            if target == -1:
                self.target = self.exindex.columns.values[-1]
                
        if type(exindex) == type(pd.DataFrame()) and len(exindex) > 0:
            if target == -1:
                self.target = exindex.columns.values[-1]
                
        # Standardize targets (0 mean and unit variance)
        
        self.scaler = standard()
        self.results = self.scaler.standardize_target(self.results, self.target)
        
        # Torch tensors and labeld external data
        
        if len(self.results) > 0:
            self.X = to_torch(self.results.drop(self.target,axis=1), gpu=gpu)
            self.y = to_torch(self.results[self.target], gpu=gpu).view(-1)
            index = ['external' + str(i) for i in range(len(self.results))]
            self.results = pd.DataFrame(self.results.values, 
                                        columns=self.results.columns,
                                        index=index)
        else:
            self.X = to_torch([], gpu=gpu)
            self.y = to_torch([], gpu=gpu)
        
    # Get results from the index
    
    def get_results(self, domain_points, append=False):
        """Returns target values corresponding to domain_points. 
        
        Parameters
        ----------
        domain_points : pandas.DataFrame
            Points from experiment index to retrieve responses for. If the
            objective is a computational function, run function and return
            responses.
        append : bool
            If true append points to results and update X and y.
        
        Returns
        ----------
        pandas.DataFrame
            Proposed experiments.
        """
        
        # Computational objective
        
        if self.computational_objective != None:
            new_results = []
            for point in domain_points.values:
                result = self.computational_objective(point)
                new_results.append(result)
                
            batch = domain_points.copy()
            batch[self.target] = new_results
            
            if append == True:
                # Unstandardize results and append to know outcomes
                results = self.scaler.unstandardize_target(self.results, self.target)
                data = pd.concat([results, batch])
            
                # Restandardize
                self.results = self.scaler.standardize_target(data, self.target)
                self.X = to_torch(self.results.drop(self.target,axis=1), gpu=self.gpu)
                self.y = to_torch(self.results[self.target], gpu=self.gpu).view(-1)
            
            return batch
        
        # Human in the loop objective
        
        if type(self.exindex) == type(None):
            return print("edbo bot: Error no experiment index")
        
        # Retrieve domain points from index
        
        index = self.exindex.drop(self.target, axis=1)
        
        union_index = pd.merge(
                index.reset_index(), 
                domain_points, 
                how='inner'
                )['index']
        
        batch = self.exindex.iloc[list(union_index)]
        
        # Append to results
            
        if append == True:
            # Unstandardize results and append to know outcomes
            results = self.scaler.unstandardize_target(self.results, self.target)
            data = pd.concat([results, batch])
            
            # Restandardize
            self.results = self.scaler.standardize_target(data, self.target)
            self.X = to_torch(self.results.drop(self.target,axis=1), gpu=self.gpu)
            self.y = to_torch(self.results[self.target], gpu=self.gpu).view(-1)
        
        return batch
    
    # Clear results
    
    def clear_results(self):
        """Clear results and reset X and y.
        
        Returns
        ----------
        None
        """
        
        self.results = pd.DataFrame()
        self.X = to_torch([], gpu=self.gpu)
        self.y = to_torch([], gpu=self.gpu)
        
    # Return unstandardized results
    
    def results_input(self):
        """Return unstandardized results.
        
        Returns
        ----------
        pandas.DataFrame
            Unstandardized results.
        """
        
        if len(self.results) == 0:
            results = self.results
        else:
            results = self.scaler.unstandardize_target(self.results, self.target)
            
        return results
        
        