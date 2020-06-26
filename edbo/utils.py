# Imports

import time

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from .math_utils import pca
from .chem_utils import ChemDraw

# Wall-clock timer for testing

class timer:
    
    def __init__(self, name):
        
        self.start = time.time()
        self.name = name
        
    def stop(self):
        """
        Returns wall clock-time.
        """
        
        self.end = time.time()    
        print(self.name + ': ' + str(self.end - self.start) + ' s')

# Data handling class

class Data:
    """
    Class or defining experiment domains and pre-processing.
    """
    
    def __init__(self, data):
        self.data = data
        self.base_data = data
    
    def reset(self):
        self.data = self.base_data
    
    def clean(self):
        self.data = drop_single_value_columns(self.data)
        self.data = drop_string_columns(self.data)
        
    def drop(self, drop_list):
        self.data = remove_features(self.data, drop_list)
    
    def standardize(self, target='yield', scaler='minmax'):
        self.data = standardize(self.data, target, scaler=scaler)
        
    def PCA(self, target='yield', n_components=1):
        pps = pca(self.data.drop(target, axis=1), n_components=n_components)
        pps[target] = self.data[target]
        self.data = pps
        
    def uncorrelated(self, target='yield', threshold=0.7):
        self.data = uncorrelated_features(self.data, 
                                          target, 
                                          threshold=threshold)
    
    def visualize(self, experiment_index_value, svg=True):
        
        columns = self.base_data.columns.values
        smi_bool = ['SMILES' in columns[i] for i in range(len(columns))]
        index = self.base_data[self.base_data.columns[smi_bool].values]
        
        SMILES_list = index.iloc[experiment_index_value].values
        cd = ChemDraw(SMILES_list, ipython_svg=svg)
        
        try:
            entry = self.base_data[self.index_headers].iloc[[experiment_index_value]]
        except:
            entry = self.base_data.iloc[[experiment_index_value]]
            
        print('\n##################################################### Experiment\n\n',
              entry,
              '\n')
        
        return cd.show()
    
    def get_experiments(self, index_values):
        try:
            entries = self.base_data[self.index_headers].iloc[index_values]
        except:
            entries = self.base_data.iloc[index_values]
            
        return entries

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

# Standardize
    
def standardize(df, target, scaler='standard'):
    """
    Standardize descriptors but keep target.
    """
    
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    if target != None:
        data = df.drop(target,axis=1)
    else:
        data = df.copy()
    
    out = scaler.fit_transform(data)
    
    new_df = pd.DataFrame(data=out, columns=data.columns)
    
    if target != None:
        new_df[target] = df[target]
    
    return new_df

# Select uncorrelated set of features
    
def uncorrelated_features(df, target, threshold=0.95):
    """
    Returns an uncorrelated set of features.
    """
    
    if target != None:
        data = df.drop(target,axis=1)
    else:
        data = df.copy()
    
    corr = data.corr().abs()
    keep = []
    for i in range(len(corr.iloc[:,0])):
        above = corr.iloc[:i,i]
        if len(keep) > 0: above = above[keep]
        if len(above[above < threshold]) == len(above):
            keep.append(corr.columns.values[i])
    
    data = data[keep]
    
    if target != None:
        data[target] = list(df[target])
    
    return data