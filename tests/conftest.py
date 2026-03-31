import matplotlib
matplotlib.use('Agg')

import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture
def sample_df():
    """A small DataFrame with numeric columns and a target."""
    np.random.seed(42)
    return pd.DataFrame({
        'feat1': np.random.randn(20),
        'feat2': np.random.randn(20),
        'feat3': np.random.randn(20),
        'yield': np.random.uniform(0, 100, 20),
    })


@pytest.fixture
def sample_df_with_strings():
    """DataFrame that includes a string column."""
    np.random.seed(42)
    return pd.DataFrame({
        'feat1': np.random.randn(10),
        'name': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        'feat2': np.random.randn(10),
        'yield': np.random.uniform(0, 100, 10),
    })


@pytest.fixture
def sample_df_with_constant_col():
    """DataFrame with a constant-value column."""
    np.random.seed(42)
    return pd.DataFrame({
        'feat1': np.random.randn(10),
        'constant': [5.0] * 10,
        'feat2': np.random.randn(10),
        'yield': np.random.uniform(0, 100, 10),
    })


@pytest.fixture
def small_tensor_X():
    """Small 2-D float tensor for model inputs."""
    torch.manual_seed(0)
    return torch.randn(20, 3)


@pytest.fixture
def small_tensor_y():
    """Small 1-D float tensor for targets."""
    torch.manual_seed(1)
    return torch.randn(20)


@pytest.fixture
def correlated_df():
    """DataFrame with highly correlated features."""
    np.random.seed(42)
    x = np.random.randn(30)
    return pd.DataFrame({
        'a': x,
        'b': x + np.random.randn(30) * 0.01,  # nearly identical to a
        'c': np.random.randn(30),
        'yield': np.random.uniform(0, 100, 30),
    })
