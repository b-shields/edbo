# -*- coding: utf-8 -*-
"""Tests for edbo.base_models — GP and RF base model classes."""

import pytest
import torch
import numpy as np
import gpytorch
from gpytorch.priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

from edbo.base_models import gp_model, random_forest, fast_computation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gp_train_data():
    torch.manual_seed(0)
    X = torch.randn(20, 4)
    y = torch.randn(20)
    return X, y


@pytest.fixture
def likelihood():
    return GaussianLikelihood()


# ---------------------------------------------------------------------------
# gp_model
# ---------------------------------------------------------------------------

class TestGpModel:

    def test_instantiation_defaults(self, gp_train_data, likelihood):
        X, y = gp_train_data
        model = gp_model(X, y, likelihood)
        assert hasattr(model, 'mean_module')
        assert hasattr(model, 'covar_module')

    def test_forward_returns_mvn(self, gp_train_data, likelihood):
        X, y = gp_train_data
        model = gp_model(X, y, likelihood)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output = model(X)
        assert isinstance(output, MultivariateNormal)
        assert output.mean.shape == (20,)

    def test_ard_dims(self, gp_train_data, likelihood):
        """ARD should set lengthscale dimension = number of features."""
        X, y = gp_train_data
        model = gp_model(X, y, likelihood)
        ls = model.covar_module.base_kernel.lengthscale
        assert ls.shape[-1] == 4

    def test_custom_nu(self, gp_train_data, likelihood):
        X, y = gp_train_data
        model = gp_model(X, y, likelihood, nu=1.5)
        assert model.covar_module.base_kernel.nu == 1.5

    def test_lengthscale_prior(self, gp_train_data, likelihood):
        X, y = gp_train_data
        prior = GammaPrior(3.0, 1.0)
        init_val = 2.0
        model = gp_model(X, y, likelihood, lengthscale_prior=[prior, init_val])
        ls = model.covar_module.base_kernel.lengthscale.detach()
        assert torch.allclose(ls, torch.tensor(2.0), atol=0.01)

    def test_outputscale_prior(self, gp_train_data, likelihood):
        X, y = gp_train_data
        prior = GammaPrior(3.0, 1.0)
        init_val = 5.0
        model = gp_model(X, y, likelihood, outputscale_prior=[prior, init_val])
        os_val = model.covar_module.outputscale.detach().item()
        assert abs(os_val - 5.0) < 0.1

    def test_empty_input(self, likelihood):
        """Empty X tensor should not crash the constructor."""
        X = torch.zeros(0, 0)
        y = torch.zeros(0)
        model = gp_model(X, y, likelihood)
        assert model is not None


# ---------------------------------------------------------------------------
# random_forest
# ---------------------------------------------------------------------------

class TestRandomForest:

    def test_instantiation_defaults(self):
        rf = random_forest()
        assert rf.n_estimators == 500
        assert rf.random_state == 10

    def test_custom_params(self):
        rf = random_forest(n_estimators=50, random_state=42, max_depth=5)
        assert rf.n_estimators == 50
        assert rf.random_state == 42
        assert rf.max_depth == 5

    def test_fit_predict(self):
        rf = random_forest(n_estimators=10)
        X = np.random.RandomState(0).randn(30, 3)
        y = np.random.RandomState(0).randn(30)
        rf.fit(X, y)
        pred = rf.predict(X)
        assert pred.shape == (30,)

    def test_is_sklearn_rf(self):
        from sklearn.ensemble import RandomForestRegressor
        rf = random_forest()
        assert isinstance(rf, RandomForestRegressor)


# ---------------------------------------------------------------------------
# fast_computation
# ---------------------------------------------------------------------------

class TestFastComputation:

    def test_toggle_on(self):
        fast_computation(True)
        assert gpytorch.settings.fast_pred_var._state is True

    def test_toggle_off(self):
        fast_computation(False)
        assert gpytorch.settings.fast_pred_var._state is False
        assert gpytorch.settings.memory_efficient._state is False
