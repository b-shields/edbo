# -*- coding: utf-8 -*-
"""Tests for edbo.models — GP_Model, RF_Model, and helpers."""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from edbo.models import GP_Model, RF_Model, Bayesian_Linear_Model, Random, score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gp_data():
    torch.manual_seed(1)
    X = torch.randn(20, 3)
    y = X[:, 0] * 2.0 + torch.randn(20) * 0.1  # roughly linear
    return X, y


@pytest.fixture
def rf_data():
    rng = np.random.RandomState(0)
    X = rng.randn(30, 4)
    y = X[:, 0] + rng.randn(30) * 0.1
    return X, y


# ---------------------------------------------------------------------------
# GP_Model
# ---------------------------------------------------------------------------

class TestGPModel:

    def test_init_stores_attributes(self, gp_data):
        X, y = gp_data
        m = GP_Model(X, y, training_iters=5)
        assert m.X is X
        assert m.y is y
        assert m.training_iters == 5

    @pytest.mark.slow
    def test_fit_and_predict(self, gp_data):
        X, y = gp_data
        m = GP_Model(X, y, training_iters=10)
        m.fit()
        pred = m.predict(X)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (20,)

    @pytest.mark.slow
    def test_variance(self, gp_data):
        X, y = gp_data
        m = GP_Model(X, y, training_iters=10)
        m.fit()
        var = m.variance(X)
        assert isinstance(var, np.ndarray)
        assert var.shape == (20,)
        assert (var >= 0).all()

    @pytest.mark.slow
    def test_sample_posterior(self, gp_data):
        X, y = gp_data
        m = GP_Model(X, y, training_iters=5)
        m.fit()
        samples = m.sample_posterior(X, batch_size=3)
        assert samples.shape == (3, 20)

    @pytest.mark.slow
    def test_fit_restart_loss_stored(self, gp_data):
        X, y = gp_data
        m = GP_Model(X, y, training_iters=5, n_restarts=1)
        m.fit()
        assert hasattr(m, 'fit_restart_loss')
        assert len(m.fit_restart_loss) == 2

    def test_noise_constraint(self, gp_data):
        X, y = gp_data
        m = GP_Model(X, y, noise_constraint=1e-4)
        # Constraint is registered on the likelihood
        assert m.model.likelihood is not None

    @pytest.mark.slow
    def test_regression_returns_scores(self, gp_data):
        X, y = gp_data
        m = GP_Model(X, y, training_iters=5)
        m.fit()
        result = m.regression(return_scores=True)
        # return_scores=True should return (RMSE, R2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# RF_Model
# ---------------------------------------------------------------------------

class TestRFModel:

    def test_init(self, rf_data):
        X, y = rf_data
        m = RF_Model(X, y, n_estimators=10)
        assert isinstance(m.X, np.ndarray)
        assert isinstance(m.y, np.ndarray)

    def test_fit_predict(self, rf_data):
        X, y = rf_data
        m = RF_Model(X, y, n_estimators=10)
        m.fit()
        pred = m.predict(X)
        assert pred.shape == (30,)

    def test_variance(self, rf_data):
        X, y = rf_data
        m = RF_Model(X, y, n_estimators=10)
        m.fit()
        var = m.variance(X)
        assert var.shape == (30,)
        assert (var >= 0).all()

    def test_sample_posterior(self, rf_data):
        X, y = rf_data
        m = RF_Model(X, y, n_estimators=10)
        m.fit()
        samples = m.sample_posterior(X, batch_size=3)
        assert samples.shape == torch.Size([3, 30])

    def test_regression_return_scores(self, rf_data):
        X, y = rf_data
        m = RF_Model(X, y, n_estimators=10)
        m.fit()
        result = m.regression(return_scores=True)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Bayesian_Linear_Model
# ---------------------------------------------------------------------------

class TestBayesianLinearModel:

    def test_fit_predict(self, rf_data):
        X, y = rf_data
        m = Bayesian_Linear_Model(X, y)
        m.fit()
        pred = m.predict(X)
        assert pred.shape == (30,)

    def test_variance(self, rf_data):
        X, y = rf_data
        m = Bayesian_Linear_Model(X, y)
        m.fit()
        var = m.variance(X)
        assert var.shape == (30,)
        assert (var >= 0).all()


# ---------------------------------------------------------------------------
# Random (dummy model)
# ---------------------------------------------------------------------------

class TestRandom:

    def test_init_and_fit(self):
        m = Random(None, None)
        assert m.fit() is None

    def test_predict_returns_none(self):
        m = Random(None, None)
        assert m.predict([1, 2, 3]) is None


# ---------------------------------------------------------------------------
# score function
# ---------------------------------------------------------------------------

class TestScore:

    def test_score_rf(self, rf_data):
        X, y = rf_data
        m = RF_Model(X, y, n_estimators=10)
        m.fit()
        rmse, r2 = score(m, X, y)
        assert rmse >= 0
        assert r2 <= 1.0
