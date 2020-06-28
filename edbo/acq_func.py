# -*- coding: utf-8 -*-

# Imports

import pandas as pd
import numpy as np
from scipy.stats import norm
import math
from random import sample as random_sample

from .pd_utils import to_torch, join_to_df, argmax, complement, sample

# Thomposon Sampling

class thompson_sampling:
    """Class represents the Thompson sampling algorithm.
    
    Provides a framework for selecting experimental conditions for parallel 
    optimization via sampling the GP predictive posterior.
    """
    
    def __init__(self, batch_size, duplicates):
        """
        Parameters
        ----------
        batch_size : int
            Number of points to select.
        duplicates : bool
            Select duplicate domain points.
        
        """
        
        self.batch_size = batch_size
        self.duplicates = duplicates
        self.chunk_size = 20000
        
    def run(self, model, obj):
        """Run Thompson sampling algorithm on a trained model and user defined domain.
        
        Parameters
        ----------
        model : bro.models 
            Trained model to be sampled.
        obj : bro.objective 
            Objective object containing information about the domain.
        
        Returns
        ----------
        pandas.DataFrame 
            Selected domain points.
        """
        
        # Draw samples from posterior
        domain = to_torch(obj.domain, gpu=obj.gpu)
        self.samples = sample(model, domain, self.batch_size, gpu=obj.gpu, chunk_size=self.chunk_size)
        #self.samples = model.sample_posterior(domain, self.batch_size)
        columns = list(obj.domain.columns.values) + ['sample']

        # ArgMax each posterior draw
        arg_maxs = pd.DataFrame(columns=columns)
        for i in range(len(self.samples)):
            sample_i = join_to_df(self.samples[i], obj.domain, gpu=obj.gpu)
            known_X = pd.concat([
                        obj.results.drop(obj.target, axis=1),
                        arg_maxs.drop('sample', axis=1)],
                        sort=False
                        )
        
            arg_max_i = argmax(sample_i, known_X, duplicates=self.duplicates)
            arg_maxs = pd.concat([arg_maxs,arg_max_i],sort=False)
        
        return arg_maxs.drop('sample', axis=1)
    
# Top predictions

class top_predicted:
    """Class represents the batched pure exploitation algorithm.
    
    Provides a framework for selecting experimental conditions for parallel 
    optimization via the top predicted values.
    """
    
    def __init__(self, batch_size, duplicates):
        """
        Parameters
        ----------
        batch_size : int
            Number of points to select.
        duplicates : bool
            Select duplicate domain points.
        
        """
        
        self.batch_size = batch_size
        self.duplicates = duplicates
        
    def run(self, model, obj):
        """Run top_predicted on a trained model and user defined domain.
        
        Parameters
        ----------
        model : bro.models 
            Trained model to be sampled.
        obj : bro.objective 
            Objective object containing information about the domain.
        
        Returns
        ----------
        pandas.DataFrame 
            Selected domain points.
        """
        
        domain = to_torch(obj.domain, gpu=obj.gpu)
        pred = obj.domain.copy()
        pred['pred'] = model.predict(domain)
    
        proposed = argmax(
                pred, 
                obj.results.drop(obj.target, axis=1), 
                target='pred',
                duplicates=self.duplicates,
                top_n=self.batch_size)
        
        return proposed.drop('pred', axis=1)

def mean(model, obj, **kwargs):
    """Compute model mean for Kriging believer pure exploitation.
        
    Parameters
    ----------
    model : bro.models
        Trained model.
    obj : bro.objective 
        Objective object containing information about the domain.
    jitter : float
        Parameter which controls the degree of exploration.
        
    Returns
    ----------
    numpy.array 
        Computed mean at each domain point.
    """
        
    pred = np.array(model.predict(obj.domain))
    
    return pred

# Max variance

class max_variance:
    """Class represents the batched pure exploration algorithm.
    
    Provides a framework for selecting experimental conditions for parallel 
    optimization via domain points with the highest model variance.
    """
    
    def __init__(self, batch_size, duplicates):
        """
        Parameters
        ----------
        batch_size : int
            Number of points to select.
        duplicates : bool
            Select duplicate domain points.
        
        """
        
        self.batch_size = batch_size
        self.duplicates = duplicates
        
    def run(self, model, obj):
        """Run max_variance on a trained model and user defined domain.
        
        Parameters
        ----------
        model : bro.models 
            Trained model to be sampled.
        obj : bro.objective 
            Objective object containing information about the domain.
        
        Returns
        ----------
        pandas.DataFrame 
            Selected domain points.
        """
        
        domain = to_torch(obj.domain, gpu=obj.gpu)
        var = obj.domain.copy()
        var['var'] = model.variance(domain)
    
        proposed = argmax(
                var, 
                obj.results.drop(obj.target, axis=1), 
                target='var',
                duplicates=self.duplicates,
                top_n=self.batch_size)
        
        self.variance = var['var'].values
        
        return proposed.drop('var', axis=1)


def variance(model, obj, **kwargs):
    """Compute model variance for Kriging believer pure exploration.
        
    Parameters
    ----------
    model : bro.models
        Trained model.
    obj : bro.objective 
        Objective object containing information about the domain.
    jitter : float
        Parameter which controls the degree of exploration.
        
    Returns
    ----------
    numpy.array 
        Computed variance at each domain point.
    """
        
    var = np.array(model.variance(obj.domain))
    
    return var

# Expected Improvement (EI)

def expected_improvement(model, obj, jitter=0.01):
    """Compute expected improvement.
    
    EI attempts to balance exploration and exploitation by accounting
    for the amount of improvement over the best observed value.
        
    Parameters
    ----------
    model : bro.models
        Trained model.
    obj : bro.objective 
        Objective object containing information about the domain.
    jitter : float
        Parameter which controls the degree of exploration.
        
    Returns
    ----------
    numpy.array 
        Computed EI at each domain point.
    """
    
    # Domain
    domain = to_torch(obj.domain, gpu=obj.gpu)
    
    # Max obsereved objective value
    if len(obj.results) == 0:
        max_observed = 0
    else:
        max_observed = obj.results.sort_values(obj.target).iloc[-1]
        max_observed = max_observed[obj.target]
    
    # Mean and standard deviation
    mean = model.predict(domain)
    stdev = np.sqrt(model.variance(domain)) + 1e-6
    
    # EI parameter values
    z = (mean - max_observed - jitter)/stdev
    imp = mean - max_observed - jitter
    ei = imp * norm.cdf(z) + stdev * norm.pdf(z)
    
    ei[stdev < jitter] = 0.0
    
    return ei

# Probability of Improvement (PI)

def probability_of_improvement(model, obj, jitter=1e-2):
    """Compute probability of improvement.
    
    PI favors exploitation of exporation. Equally rewards any
    improvement over the best observed value.
    
    As implemented in: https://github.com/maxim5/hyper-engine
           
    Parameters
    ----------
    model : bro.models
        Trained model.
    obj : bro.objective 
        Objective object containing information about the domain.
    jitter : float
        Parameter which controls the degree of exploration.
        
    Returns
    ----------
    numpy.array 
        Computed PI at each domain point.
    """
    
    # Domain
    domain = to_torch(obj.domain, gpu=obj.gpu)
    
    # Max obsereved objective value
    if len(obj.results) == 0:
        max_observed = 0
    else:
        max_observed = obj.results.sort_values(obj.target).iloc[-1]
        max_observed = max_observed[obj.target]
    
    # Mean and standard deviation
    mean = model.predict(domain)
    stdev = np.sqrt(model.variance(domain)) + 1e-6
    
    # PI parameter values
    z = (mean - max_observed - jitter)/stdev
    cdf = norm.cdf(z)
    
    cdf[stdev < jitter] == 0.0
    
    return cdf

# Upper Confidence Bound (UCB)

def upper_confidence_bound(model, obj, jitter=1e-2, delta=0.5):
    """Computes upper confidence bound.
    
    As implemented in: https://github.com/maxim5/hyper-engine
        
    Parameters
    ----------
    model : bro.models
        Trained model.
    obj : bro.objective 
        Objective object containing information about the domain.
    jitter : float
        Parameter which controls the degree of exploration.
    delta : float
        UCB parameter value.
        
    Returns
    ----------
    numpy.array 
        Computed UCB at each domain point.
    """
    
    # Domain
    domain = to_torch(obj.domain, gpu=obj.gpu)
    
    # Mean and standard deviation
    mean = model.predict(domain)
    stdev = np.sqrt(model.variance(domain)) + 1e-6
    
    # PI parameter values
    dim = len(obj.domain.columns.values)
    iters = len(obj.results)
    beta = np.sqrt(2*np.log(dim*iters**2 * math.pi**2/(6*delta)))
    
    return mean + beta * stdev

# Batching via Kriging believer

class Kriging_believer:
    """Class represents the Kriging believer algorithm.
    
    Provides a framework for selecting experimental conditions for parallel 
    optimization.
    """
    
    def __init__(self, acq_function, batch_size, duplicates):
        """
        Parameters
        ----------
        acq_function : acq_func.function
            Base acquisition function to use with Kriging believer algorithm.
        batch_size : int
            Number of points to select.
        duplicates : bool
            Select duplicate domain points.
        
        """
        
        self.acq_function = acq_function
        self.batch_size = batch_size
        self.duplicates = duplicates
        self.jitter = 0.01
        
    def run(self, model, obj):
        """Run Kriging believer algorithm on a trained model and user defined domain.
        
        Parameters
        ----------
        model : bro.models 
            Trained model to be sampled.
        obj : bro.objective 
            Objective object containing information about the domain.
        
        Returns
        ----------
        pandas.DataFrame 
            Selected domain points.
        """
        
        # Make a copy of model dictionary
        model_dict = model.__dict__.copy()
        for entry in ['X', 'y']:
            del(model_dict[entry])
        
        # Save a copy of GP hyperparameters
        if 'GP' in str(model) and 'Linear' not in str(model):
            post_ls = model.model.covar_module.base_kernel.lengthscale.clone()
            post_os = model.model.covar_module.outputscale.clone()
            post_n = model.model.likelihood.noise.clone()

        # Get choices via the Kriging believer algorithm
        proposed = pd.DataFrame(columns=obj.domain.columns)
        beliefs = []
        self.projections = []
        self.projection_means = []
        for i in range(self.batch_size):
            
            # Run acquisition function
            next_acq = self.acq_function(model, obj, jitter=self.jitter)
            
            # Log projections and model predictions
            self.projections.append(next_acq)
            fant = model.predict(to_torch(obj.domain, gpu=obj.gpu))
            self.projection_means.append(obj.scaler.unstandardize(fant))
            
            # De-duplication
            if self.duplicates == False:
                next_df = obj.domain.copy()
                next_df['sample'] = next_acq
                
                known_X = pd.concat([
                        obj.results.drop(obj.target, axis=1),
                        proposed],
                        sort=False
                        )
            
                argmax_i = argmax(next_df, 
                                  known_X, 
                                  duplicates=self.duplicates)
                proposed_i = argmax_i.drop('sample', axis=1)
            
            else:
                proposed_i = pd.DataFrame(
                        data=obj.domain.iloc[[np.argmax(next_acq)]], 
                        columns=obj.domain.columns)
            
            proposed = pd.concat([proposed, proposed_i], sort=False)

            # Get predictions
            pred_i = model.predict(to_torch(proposed_i, gpu=obj.gpu))
            beliefs.append(pred_i[0])
            
            pred = proposed.copy()
            pred[obj.target] = beliefs
            
            # Append to results
            results_believer = pd.concat(
                    [obj.results.copy(), pred],
                    sort=False)
            
            # Reinitialize model
            X = to_torch(results_believer.drop(obj.target, axis=1), 
                         gpu=obj.gpu)
            y = to_torch(results_believer[obj.target], gpu=obj.gpu) 
            del(model.__dict__)
            
            # Cholskey issues durring training
            try:
                model.__init__(X, y)

                # Update dictionaries
                for key in model_dict:
                    model.__dict__[key] = model_dict.copy()[key]
                model.model.train_inputs = (X,)
                model.model.train_targets = y
            
                # Fit model
                model.fit()
                
            except Exception as e:
                print(e)
                print('Defaulting to previous iterations hyperparameters...')
                del(model.__dict__)
                model.__init__(X, y)

                # Update dictionaries
                for key in model_dict:
                    model.__dict__[key] = model_dict.copy()[key]
                model.model.train_inputs = (X,)
                model.model.train_targets = y
                
                if 'GP' in str(model):
                    model.model.covar_module.base_kernel.lengthscale = post_ls
                    model.model.covar_module.outputscale = post_os
                    model.model.likelihood.noise = post_n
            
                # Fit model
                model.training_iters = 0
                model.fit()
                
        # Refit model with origional data
        model.__init__(obj.X, obj.y)
        for key in model_dict:
            model.__dict__[key] = model_dict.copy()[key]
        model.model.train_inputs = (obj.X,)
        model.model.train_targets = obj.y
        model.fit()
                
        return proposed

# Batching via hybrid acquisition

class hybrid_TS:
    """Class represents the hybrid Thompson sampling algorithm.
    
    Provides a framework for selecting experimental conditions for parallel 
    optimization via any acquisition funciton for the first sample and 
    Thompson sampling for the remaining batch_size - 1 points.
    """
    
    def __init__(self, hybrid, batch_size, duplicates):
        """
        Parameters
        ----------
        hybrid : bro.acq_funcs:
            hybrid method to be used. 
        batch_size : int
            Number of points to select.
        duplicates : bool
            Select duplicate domain points.
        
        """
        
        self.hybrid = hybrid
        self.batch_size = batch_size
        self.duplicates = duplicates
        
    def run(self, model, obj):
        """Run Hybrid-TS algorithm on a trained model and user defined domain.
        
        Parameters
        ----------
        model : bro.models 
            Trained model to be sampled.
        obj : bro.objective 
            Objective object containing information about the domain.
        
        Returns
        ----------
        pandas.DataFrame 
            Selected domain points.
        """
        
        # Hybrid for first sample
        if self.hybrid == 'EI':
            first = expected_improvement(model, obj)
            self.ei = first
        elif self.hybrid == 'PI':
            first = probability_of_improvement(model, obj)
            self.pi = first
        elif self.hybrid == 'UCB':
            first = upper_confidence_bound(model, obj)
            self.ucb = first
        elif self.hybrid == 'Random':
            rand = random(1, self.duplicates)
            rand_point = rand.run(model, obj)
            first = np.zeros(len(obj.domain))
            first[rand_point.index.values[0]] = 1.0
            self.rand = first
        elif self.hybrid == 'MeanMax':
            top = top_predicted(1, self.duplicates)
            top_point = top.run(model, obj)
            first = np.zeros(len(obj.domain))
            first[top_point.index.values[0]] = 1.0
            self.mean = first
        elif self.hybrid == 'VarMax':
            var = max_variance(1, self.duplicates)
            var_point = var.run(model, obj)
            first = np.zeros(len(obj.domain))
            first[var_point.index.values[0]] = 1.0
            self.var = first
        
        # De-duplication
        if self.duplicates == False:
            first_df = obj.domain.copy()
            first_df['sample'] = first
            
            argmax_i = argmax(first_df, 
                              obj.results.drop(obj.target, axis=1), 
                              duplicates=self.duplicates)
            proposed = argmax_i.drop('sample', axis=1)
            
        else:
            proposed = pd.DataFrame(
                        data=obj.domain.iloc[[np.argmax(first)]], 
                        columns=obj.domain.columns)
        
        # TS for the rest
        if self.batch_size > 1:
            
            # Draw samples from posterior
            domain = to_torch(obj.domain, gpu=obj.gpu)
            samples = model.sample_posterior(domain, self.batch_size - 1)
            columns = list(obj.domain.columns.values)
            columns.append('sample')
        
            self.samples = samples.numpy()
    
            # ArgMax each posterior draw
            arg_maxs = pd.DataFrame(columns=columns)
            for i in range(len(samples)):
                sample_i = join_to_df(samples[i], obj.domain, gpu=obj.gpu)
                known_X = pd.concat([
                        obj.results.drop(obj.target, axis=1),
                        proposed,
                        arg_maxs.drop('sample', axis=1)],
                        sort=False
                        )
        
                arg_max_i = argmax(sample_i, known_X, duplicates=self.duplicates)
                arg_maxs = pd.concat([arg_maxs,arg_max_i],sort=False)
            
            proposed = pd.concat(
                    [proposed, arg_maxs.drop('sample', axis=1)],
                    sort=False)
        
        return proposed

# Batching via epsilon-greedy-like policy

class eps_greedy:
    """Class represents the pseudo epsilon-greedy acquisition policy.
    
    Provides a framework for selecting experimental conditions for parallel 
    optimization via the top predicted values with epsilon probability of 
    random choices.
    """
    
    def __init__(self, batch_size, duplicates):
        """
        Parameters
        ----------
        batch_size : int
            Number of points to select.
        duplicates : bool
            Select duplicate domain points.
        
        """
        
        self.batch_size = batch_size
        self.duplicates = duplicates
        self.eps = 0.05
        
    def run(self, model, obj):
        """Run eps-greedy algorithm on a trained model and user defined domain.
        
        Parameters
        ----------
        model : bro.models 
            Trained model to be sampled.
        obj : bro.objective 
            Objective object containing information about the domain.
        
        Returns
        ----------
        pandas.DataFrame 
            Selected domain points.
        """
        
        # Get predictions
        domain = to_torch(obj.domain, gpu=obj.gpu)
        pred = obj.domain.copy()
        pred['pred'] = model.predict(domain)
        
        # Make choice list
        choice_list = [0] * round(self.eps * 1000) + [1] * round((1 - self.eps) * 1000)
        
        # Select batch
        selected = pd.DataFrame(columns=obj.domain.columns.values)
        for i in range(self.batch_size):
            
            # Observed domain points
            known_X = pd.concat([
                        obj.results.drop(obj.target, axis=1),
                        selected],
                        sort=False)
            
            # Sample choice list
            choice = random_sample(choice_list, 1)[0]
            
            # Random sample with probability eps
            if choice == 0:
                if self.duplicates == True:
                    candidates = obj.domain
                elif self.duplicates == False:
                    candidates = complement(obj.domain, known_X)
                selected_i = candidates.sample(1)
            
            # Else argmax model predictions
            elif choice == 1:
                selected_i = argmax(pred,
                                    known_X, 
                                    target='pred',
                                    duplicates=self.duplicates,
                                    top_n=1).drop('pred', axis=1)
            
            # Append
            selected = pd.concat([selected, selected_i], sort=False)
        
        return selected

# Batching via random choices
    
class random:
    """Class represents the random selection algorithm.
    
    Provides a random sampling method used for providing a baseline for 
    simulations and evaluating the computation time of non-model components.
    """
    
    def __init__(self, batch_size, duplicates):
        """
        Parameters
        ----------
        batch_size : int
            Number of points to select.
        duplicates : bool
            Select duplicate domain points.
        
        """
        
        self.batch_size = batch_size
        self.duplicates = duplicates
        
    def run(self, model, obj):
        """Run random sampling on a user defined domain.
        
        Parameters
        ----------
        model : bro.models 
            Trained model to be sampled.
        obj : bro.objective 
            Objective object containing information about the domain.
        
        Returns
        ----------
        pandas.DataFrame 
            Selected domain points.
        """
        
        # De-duplication
        if self.duplicates == True:
            candidates = obj.domain
        else:
            candidates = complement(obj.domain, 
                                    obj.results.drop(obj.target, axis=1))
        
        return candidates.sample(self.batch_size)

# Main acquisition class

class acquisition:
    """Class represents the main acquisition function module.
    
    Class provides a container for different acquisition functions
    availible for Bayesian optimization.
    """
    
    def __init__(self, function, batch_size=1, duplicates=False):
        """
        Parameters
        ----------
        function : str
            Acquisition function to be used. Options include: 'TS', 'EI', 'PI'
            'UCB', 'EI-TS', 'PI-TS', 'UCB-TS', 'rand-TS', 'MeanMax-TS',
            'VarMax-TS', 'MeanMax', 'VarMax', 'rand', and 'eps-greedy'.
        batch_size : int
            Number of points to select.
        duplicates : bool
            Select duplicate domain points.
        
        """
        
        if function == 'TS':
            self.function = thompson_sampling(batch_size, duplicates)
        elif function == 'EI':
            self.function = Kriging_believer(expected_improvement, 
                                             batch_size, 
                                             duplicates)
        elif function == 'PI':
            self.function = Kriging_believer(probability_of_improvement, 
                                             batch_size, 
                                             duplicates)
        elif function == 'UCB':
            self.function = Kriging_believer(upper_confidence_bound, 
                                             batch_size, 
                                             duplicates)
        elif function == 'EI-TS':
            self.function = hybrid_TS('EI', batch_size, duplicates)
        elif function == 'PI-TS':
            self.function = hybrid_TS('PI', batch_size, duplicates)
        elif function == 'UCB-TS':
            self.function = hybrid_TS('UCB', batch_size, duplicates)
        elif function == 'rand-TS':
            self.function = hybrid_TS('Random', batch_size, duplicates)
        elif function == 'MeanMax-TS':
            self.function = hybrid_TS('MeanMax', batch_size, duplicates)
        elif function == 'VarMax-TS':
            self.function = hybrid_TS('VarMax', batch_size, duplicates)   
        elif function == 'MeanMax':
            self.function = Kriging_believer(mean, 
                                             batch_size, 
                                             duplicates)
        elif function == 'VarMax':
            self.function = Kriging_believer(variance, 
                                             batch_size, 
                                             duplicates)
        elif function == 'rand':
            self.function = random(batch_size, duplicates)
        elif function == 'eps-greedy':
            self.function = eps_greedy(batch_size, duplicates)
        else:
            print('Error: invalid acquisition type')
    
    def evaluate(self, model, obj):
        """Run the selected acquisition function.
        
        Parameters
        ----------
        model : bro.models
            Trained model.
        obj : bro.objective
            Objective object containining data and scalers.
        
        Returns
        ----------
        pandas.DataFrame 
            Proposed experiments.
        """
        
        return self.function.run(model, obj)
    

    