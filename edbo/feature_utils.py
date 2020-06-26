# -*- coding: utf-8 -*-
"""
Featurization utilities

"""

# Imports

import pandas as pd
import numpy as np
from rdkit import Chem
from itertools import product

try:
    from mordred import Calculator, descriptors
except:
    print('Mordred not installed.')
    
from .utils import Data

# Calculate Mordred descriptors

def mordred(smiles_list, name='', dropna=False):
    """
    Compute all mordred descriptors for a list of smiles strings.
    """
    
    smiles_list = list(smiles_list)
    
    # Initialize descriptor calculator with all descriptors

    calc = Calculator(descriptors)
    
    output = []
    for entry in smiles_list:
        try:
            data_i = calc(Chem.MolFromSmiles(entry)).fill_missing()
        except:
            data_i = np.full(len(calc.descriptors),np.NaN)
            
        output.append(list(data_i))
        
    descriptor_names = list(calc.descriptors)
    columns = []
    for entry in descriptor_names:
        columns.append(name + '_' + str(entry))
        
    df = pd.DataFrame(data=output, columns=columns)
    df.insert(0, name + '_SMILES', smiles_list)
    
    if dropna == True:
        df = df.dropna(axis=1)
    
    return df

# One-hot-encoder

def one_hot_row(value, possible_values):
    """
    One-hot-encode a row of data.
    """
    
    ohe = []
    for entry in possible_values:
        if entry == value:
            ohe.append(1)
        else:
            ohe.append(0)
    
    return ohe

def one_hot_encode(data_column, name=''):
    """
    Generate a one-hot-encoded data column. Acts on a pandas dataframe.
    """
    
    possible_values = list(data_column.drop_duplicates())
    
    ohe = []
    for value in list(possible_values):
        row = one_hot_row(value, possible_values)
        ohe.append(row)
         
    columns = []
    for entry in possible_values:
        columns.append(name + '=' + str(entry))
        
    ohe = pd.DataFrame(data=ohe, columns=columns)
    ohe.insert(0, name, possible_values)
    
    return ohe

# Generate a search space

def encode_component(df_column, encoding, name=''):
    """
    Encode a column of an experiment index data frame.
    """
    
    if encoding == 'ohe' or encoding == 'OHE':
        descriptor_matrix = one_hot_encode(df_column, name=name)
        
    elif encoding == 'Mordred' or encoding == 'mordred':
        descriptor_matrix = mordred(df_column.drop_duplicates().values, 
                                    dropna=True, 
                                    name=name)
    elif encoding == 'numeric' or encoding == 'Numeric':
        descriptor_matrix = pd.DataFrame(df_column)
    
    return descriptor_matrix

def expand_space(index, descriptor_dict):
    """
    Generate descriptor matrix for an experiment index and dictionary of
    descriptors for each component.
    """
    
    descriptor_matrix = pd.DataFrame()
    
    for col in index.columns.values:
        
        submatrix = descriptor_dict[col].data
        expanded = [submatrix[submatrix.iloc[:,0] == e].values[0] for e in index[col].values]
        df = pd.DataFrame(expanded, columns=submatrix.columns.values)
        
        descriptor_matrix = pd.concat([descriptor_matrix, df], axis=1)
    
    return descriptor_matrix
    
def reaction_space(component_dict, encoding = {}, clean=True, 
                   decorrelation_threshold=0.95):
    """
    Build a reaction space object form component lists. 
    """
    
    # Build the experiment index
    
    index = pd.DataFrame([row for row in product(*component_dict.values())], 
                         columns=component_dict.keys())
    index = index.drop_duplicates().reset_index(drop=True)
    
    # Build descriptor sets for individual components
    index_headers = []
    descriptor_dict = {}
    for key in component_dict:
        
        # If there is an entry in encoding_dict follow the instruction
        if key in encoding:
            
            series = pd.Series(component_dict[key], name=key)
            des = encode_component(series, encoding[key], name=key)
        
        # Otherwise one-hot-encode
        else:
            series = pd.Series(component_dict[key], name=key)
            des = encode_component(series, 'ohe', name=key)
            
        # Initialize data container
        des = Data(des)
    
        # Preprocessing
        des.clean()
        des.uncorrelated(threshold=decorrelation_threshold, target=None)
        des.data.insert(0, 
                        des.base_data.columns.values[0] + '_index', 
                        des.base_data.iloc[:,0])
        
        descriptor_dict[key] = des
        index_headers.append(des.base_data.columns.values[0] + '_index')
    
    # Generate index

    reaction = Data(expand_space(index, descriptor_dict))
            
    # Preprocessing
    reaction.clean()
    reaction.drop(['index'])
    reaction.standardize(target=None, scaler='minmax')
    
    # Include descriptor_dict and index_headers
    reaction.descriptor_dict = descriptor_dict
    reaction.index_headers = index_headers
    
    return reaction    
