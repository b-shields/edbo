# -*- coding: utf-8 -*-
"""
Math utilities

"""

# Imports

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA

# Standardization

class standard:
    """
    Class for handling target standardization.
    """
    
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.unstandardized = np.array([])
    
    def standardize_target(self, df, target):
        """
        Standardize target vector to zero mean and unit variance.
        """
    
        if len(df) < 2:
            return df
    
        unstandard_vector = df[target].values
        self.unstandardized = unstandard_vector
        mean, std = unstandard_vector.mean(), unstandard_vector.std()
    
        # Prevent divide by zero error
        if std == 0.0:
            std = 1e-6
        
        self.mean = mean
        self.std = std
        
        standard_vector = (unstandard_vector - mean) / std
        new_df = df.copy().drop(target, axis=1)
        new_df[target] = standard_vector
    
        return new_df
    
    def unstandardize_target(self, df, target):
        """
        Retrun target data batck to unstandardized form
        via saved mean and standard deviation.
        """
        
        if len(df) < 2:
            return df
        
        if len(df) == len(self.unstandardized):
            new_df = df.copy().drop(target, axis=1)
            new_df[target] = self.unstandardized
            return new_df
        
        standard_vector = df[target].values
        
        unstandard_vector = (standard_vector * self.std) + self.mean 
        new_df = df.copy().drop(target, axis=1)
        new_df[target] = unstandard_vector
        
        return new_df
    
    def unstandardize(self, array):
        """
        Unstandardize an array of values.
        """
        
        return (array * self.std) + self.mean 
        
        
# Fit metrics

def model_performance(pred,obs):
    """
    Compute RMSE and R^2.
    """
    
    rmse = np.sqrt(np.mean((np.array(pred) - np.array(obs)) ** 2))
    r2 = metrics.r2_score(np.array(obs),np.array(pred))
    
    return rmse, r2

# PCA
    
def pca(X, n_components=1):
    """
    PCA reduction of dimensions.
    """
    
    model = PCA(n_components=n_components, copy=True)
    model.fit(X)
    ev = sum(model.explained_variance_ratio_[:n_components])
    
    print('Explained variance = ' + str(ev))
    
    columns = ['pc' + str(i + 1) for i in range(n_components)]
    pps = model.transform(X)
    
    return pd.DataFrame(data=pps, columns=columns)
    
