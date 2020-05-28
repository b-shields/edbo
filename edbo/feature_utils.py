# -*- coding: utf-8 -*-
"""
Featurization utilities

"""

# Imports

import pandas as pd
import numpy as np
from rdkit import Chem
try:
    from mordred import Calculator, descriptors
except:
    print('Mordred not installed.')

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
        columns.append(name + str(entry))
        
    df = pd.DataFrame(data=output, columns=columns)
    df.insert(0, name + 'SMILES', smiles_list)
    
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
    for value in list(data_column):
        row = one_hot_row(value, possible_values)
        ohe.append(row)
         
    columns = []
    for entry in possible_values:
        columns.append(name + '=' + str(entry))
        
    ohe = pd.DataFrame(data=ohe, columns=columns)
    ohe.insert(0, name + '_' + 'SMILES', possible_values)
    
    return ohe

# Build a descriptor matrix

def descriptor_matrix(molecule_index, lookup_table, lookup='SMILES', name=''):
    """
    For each entry in molecule_index add the corresponding entry from the
    lookup_table.
    """
    
    # New column names
    
    columns = list(lookup_table.columns.values)
    new_columns = []
    for column in columns:
        if name != '':
            new_columns.append(name + '_' + str(column))
        else:
            new_columns.append(column)
    
    # Build descriptor matrix
        
    build = []
    for entry in list(molecule_index):
        match = lookup_table[lookup_table[lookup] == entry]
        if len(match) > 0:
            build.append(list(match.iloc[0]))
        else:
            build.append(np.full(len(columns),np.NaN))
            
    build = pd.DataFrame(data=build, columns=new_columns)
    
    return build

# Build a descriptor set
    
def build_experiment_index(index, index_list, lookup_table_list, lookup_list):
    """
    Build a descriptor matrix.
    """
    
    matrix = descriptor_matrix(index_list[0], 
                               lookup_table_list[0], 
                               lookup=lookup_list[0])
    
    matrix.insert(0, 'entry', list(index))
    
    for i in range(1,len(index_list)):
        new = descriptor_matrix(index_list[i], 
                                lookup_table_list[i], 
                                lookup=lookup_list[i])
        new['entry'] = list(index)
        matrix = matrix.merge(new, on='entry')
    
    return matrix



