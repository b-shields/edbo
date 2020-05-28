# -*- coding: utf-8 -*-
"""
Feature selection functions

"""

# Imports

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# RF (Permutation importance feature selection)

class RF:
    """
    Addapted from Permutation Importance with Multicollinear or Correlated 
    Features in sklearn.
    
    Takes availible data as a DataFrame and returns a feature selection
    object.
    """
    def __init__(self, use_data='all'):
        """
        param: importance_type (str): which portion of the data to evaluate
               permutation importance on (all, training, test).
        """
        
        self.importance_type = use_data
        self.model = RandomForestRegressor(n_jobs=-1, 
                                           random_state=10, 
                                           n_estimators=500, 
                                           max_features='auto',
                                           max_depth=None,
                                           min_samples_leaf=1,
                                           min_samples_split=2)
    
    # Run model and compute permutation importance
    
    def run(self, df, target, n_repeats=5, random_state=1):
        """
        Fit random forest model and compute permulation imortance.
        """
        
        # Select training and test data
        
        if self.importance_type == 'all':
            X_train = df.drop(target, axis=1)
            y_train = df[target]
        else:
            X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),
                                                                df[target],
                                                                test_size=0.2,
                                                                random_state=random_state)
        
        # Fit RF model and compute permutation importance
        
        self.model.fit(X_train, y_train)
        
        result = permutation_importance(self.model, 
                                         X_train,
                                         y_train, 
                                         n_repeats=n_repeats,
                                         random_state=random_state)
        
        # Set results to class variables
        
        self.result = result
        self.features = df.drop(target, axis=1).columns.values
        self.importances = pd.DataFrame(self.result.importances, 
                                        index=self.features)
        self.importances['mean'] = self.importances.mean(axis=1)
    
    def plot_importances(self, top_k=10, export_path=None):
        
        perm_sorted_idx = self.result.importances_mean.argsort()
        
        plot_data = self.result.importances[perm_sorted_idx[-top_k:]].T
        plot_labels = self.features[perm_sorted_idx[-top_k:]]

        fig, ax = plt.subplots(1, figsize=(8, 16))
        ax.boxplot(plot_data, 
                   vert=False,
                   labels=plot_labels)
        
        if export_path != None:
            plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        
        return plt.show()
    
    def get_best(self, threshold):
        """
        Return descriptors with importance above threshold.
        """
        
        best = self.importances[self.importances['mean'] > threshold]
        s = best['mean'].sum()
        tot = self.importances['mean'].sum()
        n = len(best)
        
        print('N features = ' + str(n) + ' | % Importance = ' + str(s/tot*100))
        
        return best.index.values
        
# Uncorrelated set

def uncorrelated_features(df, target=None, threshold=0.7):
    """
    Returns an uncorrelated set of features.
    """
    
    if target == None:
        corr = df.corr().abs()
    else:    
        corr = df.drop(target,axis=1).corr().abs()
    
    keep = []
    for i in range(len(corr.iloc[:,0])):
        above = corr.iloc[:i,i]
        if len(keep) > 0: above = above[keep]
        if len(above[above < threshold]) == len(above):
            keep.append(corr.columns.values[i])
    
    new = df.copy()[keep]
    
    if target != None:
        new[target] = list(df[target])
    
    return new
        
# Remove columns with only a single value

def drop_single_value_columns(df):
    """
    Drop datafame columns with zero variance. Return a new dataframe.
    """
    
    keep = []
    for i in range(len(df.columns)):
        if len(df.iloc[:,i].drop_duplicates()) > 1:
            keep.append(df.columns.values[i])
            
    return df[keep]
    
# Remove columns with non-numeric entries
    
def drop_string_columns(df):
    """
    Drop dataframe columns with non-numeric values. Return a new dataframe.
    """
    
    keep = []
    for i in range(len(df.columns)):
        unique = df.iloc[:,i].drop_duplicates()
        keepQ = True
        for j in range(len(unique)):
            if type(unique.iloc[j]) == type(''):
                keepQ = False
                break
        if keepQ: keep.append(df.columns.values[i])
        
    return df[keep]
        
# Remove unwanted descriptors

def remove_features(df, drop_list):
    """
    Remove features from dataframe with columns containing substrings in
    drop_list. Return a new dataframe.
    """

    keep = []
    for column_name in list(df.columns.values):
        keepQ = True
        for substring in list(drop_list):
            if substring in column_name:
                keepQ = False
                break
        if keepQ: keep.append(column_name)
    
    return df[keep]

# Get features by column name

def get_features(df, keep_list):
    """
    Remove features from dataframe with columns not containing substrings in
    drop_list. Return a new dataframe.
    """

    keep = []
    for column_name in list(df.columns.values):
        for substring in list(keep_list):
            if substring in column_name:
                keep.append(column_name)
                break
                
    return df[keep]

# Standardize
    
def standardize(df, target=None, scaler='minmax'):
    """
    Standardize descriptors but keep target.
    """
    
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    if target == None:
        df_temp = df.copy()
    else:
        df_temp = df.copy().drop(target, axis=1)
    
    out = scaler.fit_transform(df_temp)
    new_df = pd.DataFrame(data=out, columns=df_temp.columns)
    
    if target != None:
        new_df[target] = df[target]
    
    return new_df