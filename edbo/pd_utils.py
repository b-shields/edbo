# -*- coding: utf-8 -*-

# Imports

import pandas as pd
import torch
import numpy as np

# Load data from csv or excel file

def load_csv_or_excel(file_path, index_col=None):
    """
    Import csv or excel file using pandas.
    """
    
    file_path = str(file_path)
    
    if '.csv' in file_path:
        data = pd.read_csv(file_path, index_col=index_col)
    elif '.xlsx' in file_path:
        data = pd.read_excel(file_path, index_col=index_col)
    else:
        data = pd.DataFrame()
        
    return data

# Load and concat all data in experiment directory

def load_experiment_results(experiment_results_path):
    """
    Load and concatenate all csv or excel files in the 
    experiment_results_path folder.
    """
    
    from os import listdir
    from os.path import isdir
    
    experiment_results_path = str(experiment_results_path)
    
    if isdir(experiment_results_path):
        files = listdir(experiment_results_path)
    else:
        files = []
        print('Not a directory.')
    
    if len(files) > 0:
        data = load_csv_or_excel(experiment_results_path + '/' + files[0])
    else:
        data = pd.DataFrame()
        
    if len(files) > 1:
        for i in range(1,len(files)):
            data_i = load_csv_or_excel(experiment_results_path + '/' + files[i])
            data = pd.concat([data, data_i])
    
    return data.reset_index(drop=True)

# Export csv file ierativly to keep track of results
    
def write_experiment_results(data,experiment_results_path):
    """
    Write experiment results from simulations.
    """
    
    from os import listdir
    
    count = len(listdir(experiment_results_path)) + 1
    
    data.to_csv(
                experiment_results_path + '/batch' + str(count) + '.csv',
                index=False
                )
    
# Convert to torch tensors

def to_torch(data, gpu=False):
    """
    Convert from pandas dataframe or numpy array to torch array.
    """
    
    if 'torch' in str(type(data)):
        torch_data = data
    
    else:
        try:
            torch_data = torch.from_numpy(np.array(data).astype('float')).float()
        except:
            torch_data = torch.tensor(data).float()

    if torch.cuda.is_available() and gpu == True:
        torch_data = torch_data.cuda()
    
    return torch_data

def torch_to_numpy(data, gpu=False):
    """
    Convert from torch.tensor to a numpy.array.
    """
    
    # Torch conversion
    if torch.cuda.is_available() and gpu == True:
        out = np.array(data.detach().cpu())
    else:
        try:
            out = np.array(data.detach())
        except:
            out = np.array(data)
        
    return out

# Complement of two dataframes

def complement(df1, df2, rounding=False, boolean_out=False):
    """
    Complement of two dataframes. Remove elements of df2 in df1.
    Retains indices of df1. There must be a better way to do this
    but pandas was either slow or didn't properly remove
    duplicates.
    """
    
    df1 = df1.copy()
    df2 = df2.copy().reset_index(drop=True)
    
    if rounding != False:
        df1 = df1.round(decimals=rounding)
        df2 = df2.round(decimals=rounding)
        
    np1 = np.array(df1)
    np2 = np.array(df2)
    
    boolean = []
    for i in range(len(np1)):
        boolean_i = True
        if boolean.count(False) < len(np2):
            for j in range(len(np2)):
                if list(np1[i]) == list(np2[j]):
                    boolean_i = False
                    break
        boolean.append(boolean_i)
    
    if boolean_out == True:
        return boolean
    else:
        return df1[boolean]

# Sampling

def chunk_sample(model, domain_tensor, batch_size, gpu=False, chunk_size=5000):
    """
    Sample large spaces can lead to memory issues. To deal with this
    we can chop the space up into smaller portions, sample the posterior
    predictive distribution, and the concatenate them. Clunky but just
    a quick patch for now.
    """
    
    # Get number of chunks and remainder
    chunks = len(domain_tensor) // chunk_size
    remainder = len(domain_tensor) % chunk_size

    # Get samples
    samples = pd.DataFrame()
    for i in range(chunks):
    
        # Sample chunk
        X = domain_tensor[i*chunk_size:(i+1)*chunk_size]
        sample = model.sample_posterior(X, batch_size)
    
        # Torch conversion
        if torch.cuda.is_available() and gpu == True:
            sample = pd.DataFrame(np.array(sample.detach().cpu()))
        else:
            sample = pd.DataFrame(np.array(sample.detach()))
        
        # concatenate
        samples = pd.concat([samples, sample], axis=1)
    
    # Sample last chunk
    if remainder > 0:
        X = domain_tensor[-remainder:]
        sample = model.sample_posterior(X, batch_size)
        # Torch conversion
        if torch.cuda.is_available() and gpu == True:
            sample = pd.DataFrame(np.array(sample.detach().cpu()))
        else:
            sample = pd.DataFrame(np.array(sample.detach()))

        samples = pd.concat([samples, sample], axis=1)
    
    return to_torch(samples, gpu=gpu)
    
def sample(model, domain_tensor, batch_size, gpu=False, chunk_size=5000):
    """
    Sample posterior predictive distribution.
    """
    
    # If the domain is smaller than chunk_size then don't break it up
    if len(domain_tensor) < chunk_size:
        samples = model.sample_posterior(domain_tensor, batch_size)  
    else:
        samples = chunk_sample(model, domain_tensor, batch_size, gpu=False, chunk_size=chunk_size)
        
    return samples
    
# ArgMax a set of posterior draws
        
def join_to_df(sample, domain, gpu=False):
    """
    Join sample and candidates (X values). Works on torch arrays.
    Returns a dataframe. Column names from candidates.
    """

    if torch.cuda.is_available() and gpu == True:
        sample = np.array(sample.detach().cpu())
    else:
        sample = np.array(sample.detach())
        
    domain_sample = pd.DataFrame(
            data=np.array(domain),
            columns=domain.columns.values)
    
    domain_sample['sample'] = sample

    return domain_sample
            
def argmax(sample_x_y, known_X, target='sample', duplicates=False, top_n=1):
    """
    ArgMax with or without duplicates. Works on dataframes.
    """
    
    sorted_sample = sample_x_y.sort_values(by=target, ascending=False)
    
    if duplicates != False:
        arg_max = sorted_sample.iloc[[0]]
    else:
        keep = complement(sample_x_y.drop(target,axis=1),known_X,boolean_out=True)
        sample = sample_x_y[keep]
        sorted_sample = sample.sort_values(by=target, ascending=False)
        arg_max = sorted_sample.iloc[0:top_n]
    
    return arg_max    


