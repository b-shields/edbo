# -*- coding: utf-8 -*-

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
    """Compute chemical descriptors for a list of SMILES strings.
    
    Parameters
    ----------
    smiles_list : list 
        List of SMILES strings.
    name : str
        Name prepended to each descriptor name (e.g., nBase --> name_nBase).
    dropna : bool
        If true, drop columns which contain np.NaNs.
    
    Returns
    ----------
    pandas.DataFrame
        DataFrame containing overlapping Mordred descriptors for each SMILES
        string.
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
    """One-hot-encode a row of data.
    
    Parameters
    ----------
    value : obj
        Any one of the possible values to be one-hot-encoded.
    possible_values : list
        List of possible values.
    
    Returns
    ----------
    list
        One-hot-encoded value.
    """
    
    ohe = []
    for entry in possible_values:
        if entry == value:
            ohe.append(1)
        else:
            ohe.append(0)
    
    return ohe

def one_hot_encode(data_column, name=''):   
    """Generate a one-hot-encoded data column.
    
    Parameters
    ----------
    data_column : pandas.Series
        DataFrame column to be one-hot-encoded.
    name : str
        Name prepended to each descriptor name (e.g., KOH --> name=KOH).
    
    Returns
    ----------
    pandas.DataFrame
        DataFrame containing one-hot-encoded data_column.
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
    """Encode an experiment index column.
    
    Function will attempt to encode a data column according to specified encoding.
    If it encounters an issue, an edbo bot is spawned to help resulve the issue.
    
    Parameters
    ----------
    df_column : pandas.Series
        DataFrame column to be encoded.
    encoding : str
        Encoding method to be used: 'ohe', 'mordred', 'smiles', 'numeric', 
        'resolve'.
    name : str
        Name prepended to each descriptor.
    
    Returns
    ----------
    pandas.DataFarme
        Descriptor matrix.
    """
    # Spawn a bot to deal with issues
    
    edbo_bot = bot()
    
    # Encode components
    
    if encoding.lower() == 'ohe':
        descriptor_matrix = one_hot_encode(df_column, name=name)
        
    elif encoding.lower() == 'mordred' or encoding.lower() == 'smiles':
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
        
    elif encoding.lower() == 'numeric':
        descriptor_matrix = pd.DataFrame(df_column)
    
    elif encoding.lower() == 'resolve':
        
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
    """Generate reaction space from individual descriptor matrices.
    
    Parameters
    ----------
    index : pandas.DataFrame
        Index of experiments with columns corresponding to keys in the descriptor
        dictionary and values corresponding to rows in each descriptor matrix.
    descriptor_dict: dict
        Dictionary of descriptor matrices for each component in the experiment
        index. Has the form {'<index column name>': pandas.DataFrame,...}.
    
    Returns
    ----------
    pandas.DataFrame
        Full descriptor matrix for the reaction space.
    """
    
    descriptor_matrix = pd.DataFrame()
    
    for col in index.columns.values:
        
        submatrix = descriptor_dict[col].data.values
        expanded = [submatrix[submatrix[:,0] == e][0] for e in index[col].values]
        df = pd.DataFrame(expanded, columns=descriptor_dict[col].data.columns.values)
        
        descriptor_matrix = pd.concat([descriptor_matrix, df], axis=1)
    
    return descriptor_matrix
    
def reaction_space(component_dict, encoding={}, descriptor_matrices={}, 
                   clean=True, decorrelate=True, decorrelation_threshold=0.95, 
                   standardize=True):
    """Build a reaction space object form component lists.
    
    Parameters
    ----------
    reaction_components : dict
        Dictionary of reaction components of the form: 
                
        Example
        -------
        Defining reaction components ::
        
            {'A': [a1, a2, a3, ...],
             'B': [b1, b2, b3, ...],
             'C': [c1, c2, c3, ...],
                        .
             'N': [n1, n2, n3, ...]}
            
        Components can be specified as: (1) arbitrary names, (2) chemical 
        names or nicknames, (3) SMILES strings, or (4) numeric values.
    encodings : dict
        Dictionary of encodings with keys corresponding to reaction_components.
        Encoding dictionary has the form: 
                
        Example
        -------
        Defining reaction encodings ::
                
            {'A': 'resolve',
             'B': 'ohe',
             'C': 'smiles',
                  .
             'N': 'numeric'}
            
        Encodings can be specified as: ('resolve') resolve a compound name 
        using the NIH database and compute Mordred descriptors, ('ohe') 
        one-hot-encode, ('smiles') compute Mordred descriptors using a smiles 
        string, ('numeric') numerical reaction parameters are used as passed.
        If no encoding is specified, the space will be automatically 
        one-hot-encoded.
    descriptor_matrices : dict
        Dictionary of descriptor matrices where keys correspond to 
        reaction_components and values are pandas.DataFrames.
            
        Descriptor dictionary has the form: 
                
        Example
        -------
        User defined descriptor matrices ::
                
            # DataFrame where the first column is the identifier (e.g., a SMILES string)
                
            A = pd.DataFrame([....], columns=[...])
                
            --------------------------------------------
            A_SMILES  |  des1  |  des2  | des3 | ...
            --------------------------------------------
                .         .        .       .     ...
                .         .        .       .     ...
            --------------------------------------------
                
            # Dictionary of descriptor matrices defined as DataFrames
                
            descriptor_matrices = {'A': A}
            
        Note
        ----
        If a key is present in both encoding and descriptor_matrices then 
        the descriptor matrix will take precedence.
    
    clean : bool
        If True, remove non-numeric and singular columns from the space.
    decorrelate : bool
        If True, iteratively remove features which are correlated with selected
        descriptors.
    decorrelation_threshold : float
        Remove features which have a correlation coefficient greater than
        specified value.
    standardize : bool
        If True, standardize descriptors on the unit hypercube.
    
    Returns
    ----------
    edbo.utils.Data
        Reaction space data container.
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
        
        # If there is a descriptor matrix use it
        if key in descriptor_matrices:
            des = descriptor_matrices[key].copy()
        
        # If there is an entry in encoding_dict follow the instruction
        elif key in encoding:
            series = pd.Series(component_dict[key], name=key)
            des = encode_component(series, encoding[key], name=key)
        
        # Otherwise one-hot-encode
        else:
            series = pd.Series(component_dict[key], name=key)
            des = encode_component(series, 'ohe', name=key)
            
        # Initialize data container
        des = Data(des)
    
        # Preprocessing
        if clean:
            des.clean()
        
        if decorrelate:
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
    if clean:
        reaction.clean()
    reaction.drop(['index'])
    if standardize:
        reaction.standardize(target=None, scaler='minmax')
    
    # Include descriptor_dict and index_headers
    reaction.descriptors = descriptor_dict
    reaction.index_headers = index_headers
    
    return reaction    

# Build a descriptor matrix

def descriptor_matrix(molecule_index, lookup_table, lookup='SMILES', name=''):
    """Generate a descriptor matrix."""
    
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
    """Build a descriptor matrix."""

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

