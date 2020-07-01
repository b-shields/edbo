# -*- coding: utf-8 -*-

# Imports

from .pd_utils import to_torch
import sklearn.model_selection as sklms
import torch

# Torch complement

def torch_complement(tensor1, tensor2, boolean_out=False):
    """
    Return the complement of tensors 1 and 2.
    """
    
    boolean = []
    
    for i in range(len(tensor1)):
        boolean_i = True
        if boolean.count(False) < len(tensor2):
            for j in range(len(tensor2)):
                if tensor1[i] == tensor2[j]:
                    boolean_i = False
                    break
        boolean.append(boolean_i)
    
    if boolean_out == True:
        return boolean
    else:
        return tensor1[boolean]  

# Train/test split

def train_test_split(X, y, test_size=0.2, random_state=10, gpu=False):
    """
    Training and test splits for torch array.
    """
       
    X_train, X_test, y_train, y_test = sklms.train_test_split(X.numpy(),
                                                             y.numpy(),
                                                             test_size=test_size, 
                                                             random_state=1)
    X_train = to_torch(X_train, gpu=gpu)
    X_test = to_torch(X_test, gpu=gpu)
    y_train = to_torch(y_train, gpu=gpu)
    y_test = to_torch(y_test, gpu=gpu)
    
    return X_train, X_test, y_train, y_test

# K-fold cross-validation split

def cv_split(X, n_splits=5, random_state=10):
    """
    CV indices for K-fold cross-validation.
    """
    
    if type(X) == torch.Tensor:
        if X.is_cuda:
            copy = X.cpu().numpy()
        else:
            copy = X.numpy()
    else:
        copy = X.copy()
    
    kf = sklms.KFold(n_splits=n_splits, random_state=random_state, shuffle=True)    
    splits = kf.split(copy)
    
    return splits


