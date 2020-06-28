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
    
from .utils import Data, bot
from .chem_utils import name_to_smiles

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
    
    # Spawn a bot to deal with issues
    
    edbo_bot = bot()
    
    # Encode components
    
    if encoding == 'ohe' or encoding == 'OHE':
        descriptor_matrix = one_hot_encode(df_column, name=name)
        
    elif encoding == 'Mordred' or encoding == 'mordred':
        descriptor_matrix = mordred(df_column.drop_duplicates().values, 
                                    dropna=True, 
                                    name=name)
        
        # Issue: Mordred didn't encode all SMILES strings
        
        if len(descriptor_matrix.columns.values) == 1:
            
            # Resolve issue using edbo bot
            
            question = 'Mordred failed to encode one or more SMILES strings in '
            question += name + '. Would you like to one-hot-encode instead?'
            question_root = 'would you like to one-hot-encode?'
            triggers = ['ohe', 'y', 'one']
            not_triggers = ['smi', 'no']
            
            def response():
                edbo_bot.talk('OK one-hot-encoding ' + name + '...')
                return one_hot_encode(df_column, name=name)
            
            def not_response():
                edbo_bot.talk('Identifying problematic SMILES string(s)...')
                edbo_bot.talk('Mordred failed with the following string(s):')
                
                i = 0
                for entry in df_column.values:
                    row = mordred([entry], dropna=True, name=name)
                    if len(row.iloc[0]) == 1:
                        print('(' + str(i) + ')  ', entry)
                    i += 1
                    
                edbo_bot.talk(name + ' was removed from the reaction space.' +
                              ' Resolve issues with SMILES string(s) and try again.')
                
                return descriptor_matrix

            descriptor_matrix = edbo_bot.resolve(question, 
                                                 question_root, 
                                                 triggers, 
                                                 not_triggers, 
                                                 response, 
                                                 not_response)
        
    elif encoding == 'numeric' or encoding == 'Numeric':
        descriptor_matrix = pd.DataFrame(df_column)
    
    elif encoding == 'resolve' or encoding == 'Resolve':
        
        # Resolve names using NIH database
        
        names = df_column.drop_duplicates().values
        smiles = pd.Series([name_to_smiles(s) for s in names], name=name)
        new_smiles = np.array(smiles.values)
        
        # Issue couldn't resolve some of the names
        
        if 'FAILED' in smiles.values:
            
            failed = np.argwhere(np.array(smiles) == 'FAILED').flatten()
            
            # Resolve issue using edbo bot
            edbo_bot.talk('The following names could not be converted to SMILES strings:')
            for name_i, i in zip(np.array(names)[failed], failed):
                print('(' + str(i) + ')  ', name_i)
            
            question = 'Would you like to enter a SMILES string or one-hot-encode this component?'
            question_root = 'which would you like to try (smiles or ohe)?'
            triggers = ['ohe', 'one']
            not_triggers = ['smi']
            
            def response():
                edbo_bot.talk('OK one-hot-encoding ' + name + '...')
                return one_hot_encode(df_column, name=name)
            
            def not_response():
                
                for i in failed:
                    name_i = names[i]
                    text = edbo_bot.get_response('SMILES string for ' + name_i + '?')
                    new_smiles[i] = text
                
                edbo_bot.talk('OK computing Mordred descriptors for ' + name + '...')
                descriptor_matrix = encode_component(pd.Series(new_smiles, name=df_column.name),
                                             'mordred', 
                                             name=name)
                
                return descriptor_matrix

            descriptor_matrix = edbo_bot.resolve(question, 
                                                 question_root, 
                                                 triggers, 
                                                 not_triggers, 
                                                 response, 
                                                 not_response)
            
        else:
            descriptor_matrix = encode_component(pd.Series(new_smiles, name=df_column.name),
                                             'mordred', 
                                             name=name)
            
    return descriptor_matrix

def expand_space(index, descriptor_dict):
    """
    Generate descriptor matrix for an experiment index and dictionary of
    descriptors for each component.
    """
    
    descriptor_matrix = pd.DataFrame()
    
    for col in index.columns.values:
        
        submatrix = descriptor_dict[col].data.values
        expanded = [submatrix[submatrix[:,0] == e][0] for e in index[col].values]
        df = pd.DataFrame(expanded, columns=descriptor_dict[col].data.columns.values)
        
        descriptor_matrix = pd.concat([descriptor_matrix, df], axis=1)
    
    return descriptor_matrix
    
def reaction_space(component_dict, encoding = {}, clean=True,
                   decorrelation_threshold=0.9):
    """
    Build a reaction space object form component lists. 
    """
    
    if component_dict == {}:
        reaction = Data(pd.DataFrame())
        reaction.descriptors = {}
        reaction.index_headers = []
        return reaction
        
    # Build descriptor sets for individual components
    index_headers = []
    descriptor_dict = {}
    final_component_dict = {}
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
        try:
            des.uncorrelated(threshold=decorrelation_threshold, target=None)
        except:
            None
        des.data.insert(0, 
                        des.base_data.columns.values[0] + '_index', 
                        des.base_data.iloc[:,0])
        
        descriptor_dict[key] = des
        final_component_dict[key] = des.data.iloc[:,0].values
        index_headers.append(des.base_data.columns.values[0] + '_index')
    
    # Build the experiment index
        
    index = pd.DataFrame([row for row in product(*final_component_dict.values())], 
                         columns=final_component_dict.keys())
    index = index.drop_duplicates().reset_index(drop=True)
    
    # Generate encoded index
    
    reaction = Data(expand_space(index, descriptor_dict))
            
    # Preprocessing
    reaction.clean()
    reaction.drop(['index'])
    reaction.standardize(target=None, scaler='minmax')
    
    # Include descriptor_dict and index_headers
    reaction.descriptors = descriptor_dict
    reaction.index_headers = index_headers
    
    return reaction    
