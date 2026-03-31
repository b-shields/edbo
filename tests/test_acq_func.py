# -*- coding: utf-8 -*-
"""Tests for edbo.acq_func module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from edbo.acq_func import (
    acquisition,
    thompson_sampling,
    Kriging_believer,
    hybrid_TS,
    eps_greedy,
    random,
    top_predicted,
    max_variance,
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
    mean as acq_mean,
    variance as acq_variance,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_obj(n_domain=20, n_results=5, n_features=3, target="yield"):
    """Return a lightweight mock objective."""
    obj = MagicMock()
    obj.gpu = False
    obj.target = target

    cols = [f"x{i}" for i in range(n_features)]
    obj.domain = pd.DataFrame(
        np.random.RandomState(0).rand(n_domain, n_features), columns=cols
    )
    # results: a subset of domain rows + target column
    results_X = obj.domain.iloc[:n_results].copy()
    results_X[target] = np.random.RandomState(1).rand(n_results)
    obj.results = results_X

    obj.X = MagicMock()
    obj.y = MagicMock()

    # scaler mock for Kriging believer
    scaler = MagicMock()
    scaler.unstandardize = lambda x: x
    obj.scaler = scaler

    return obj


def _make_model(n_domain=20, n_features=3):
    """Return a mock model that responds to predict / variance / sample_posterior."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.random.RandomState(2).rand(n_domain))
    model.variance = MagicMock(return_value=np.abs(np.random.RandomState(3).rand(n_domain)))
    # sample_posterior returns a tensor-like 2-D array  (batch x domain)
    import torch
    model.sample_posterior = MagicMock(
        return_value=torch.randn(3, n_domain)
    )
    return model


# ---------------------------------------------------------------------------
# acquisition class – constructor dispatch
# ---------------------------------------------------------------------------

class TestAcquisitionDispatch:
    """Verify that the acquisition wrapper correctly dispatches to sub-classes."""

    @pytest.mark.parametrize(
        "name, expected_type",
        [
            ("TS", thompson_sampling),
            ("ts", thompson_sampling),
            ("EI", Kriging_believer),
            ("pi", Kriging_believer),
            ("UCB", Kriging_believer),
            ("EI-TS", hybrid_TS),
            ("PI-TS", hybrid_TS),
            ("UCB-TS", hybrid_TS),
            ("rand-TS", hybrid_TS),
            ("MeanMax-TS", hybrid_TS),
            ("VarMax-TS", hybrid_TS),
            ("MeanMax", Kriging_believer),
            ("VarMax", Kriging_believer),
            ("rand", random),
            ("eps-greedy", eps_greedy),
        ],
    )
    def test_dispatch(self, name, expected_type):
        acq = acquisition(name, batch_size=2, duplicates=False)
        assert isinstance(acq.function, expected_type)

    def test_invalid_function_does_not_raise(self, capsys):
        """An unrecognised name prints a message but does not raise."""
        acquisition("nonexistent_function_xyz")
        captured = capsys.readouterr()
        assert "valid acquisition function" in captured.out.lower()


# ---------------------------------------------------------------------------
# Standalone acquisition functions (EI, PI, UCB, mean, variance)
# ---------------------------------------------------------------------------

class TestStandaloneAcqFunctions:
    """Test the analytic acquisition functions with mock model/obj."""

    def setup_method(self):
        self.n_domain = 15
        self.n_features = 3
        self.obj = _make_obj(n_domain=self.n_domain, n_features=self.n_features)
        self.model = _make_model(n_domain=self.n_domain, n_features=self.n_features)

    def test_ei_returns_array_of_correct_length(self):
        ei = expected_improvement(self.model, self.obj)
        assert len(ei) == self.n_domain

    def test_ei_nonnegative(self):
        ei = expected_improvement(self.model, self.obj)
        assert np.all(ei >= 0.0)

    def test_pi_returns_array(self):
        pi = probability_of_improvement(self.model, self.obj)
        assert len(pi) == self.n_domain

    def test_pi_values_in_0_1(self):
        pi = probability_of_improvement(self.model, self.obj)
        assert np.all(pi >= 0.0) and np.all(pi <= 1.0)

    def test_ucb_returns_array(self):
        ucb = upper_confidence_bound(self.model, self.obj)
        assert len(ucb) == self.n_domain

    def test_mean_returns_array(self):
        # acq_mean calls model.predict with obj.domain (DataFrame, not tensor)
        self.model.predict = MagicMock(
            return_value=np.random.rand(self.n_domain)
        )
        m = acq_mean(self.model, self.obj)
        assert len(m) == self.n_domain

    def test_variance_returns_array(self):
        self.model.variance = MagicMock(
            return_value=np.abs(np.random.rand(self.n_domain))
        )
        v = acq_variance(self.model, self.obj)
        assert len(v) == self.n_domain

    def test_ei_with_empty_results(self):
        """When there are no observed results, max_observed defaults to 0."""
        self.obj.results = pd.DataFrame()
        ei = expected_improvement(self.model, self.obj)
        assert len(ei) == self.n_domain

    def test_pi_with_empty_results(self):
        self.obj.results = pd.DataFrame()
        pi = probability_of_improvement(self.model, self.obj)
        assert len(pi) == self.n_domain


# ---------------------------------------------------------------------------
# thompson_sampling
# ---------------------------------------------------------------------------

class TestThompsonSampling:
    def test_run_returns_dataframe(self):
        n_domain, n_features, batch = 20, 3, 2
        obj = _make_obj(n_domain=n_domain, n_features=n_features)
        model = _make_model(n_domain=n_domain, n_features=n_features)

        import torch
        model.sample_posterior = MagicMock(return_value=torch.randn(batch, n_domain))

        ts = thompson_sampling(batch_size=batch, duplicates=False)
        result = ts.run(model, obj)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == batch

    def test_run_columns_match_domain(self):
        n_domain, n_features, batch = 20, 3, 1
        obj = _make_obj(n_domain=n_domain, n_features=n_features)
        model = _make_model(n_domain=n_domain, n_features=n_features)

        import torch
        model.sample_posterior = MagicMock(return_value=torch.randn(batch, n_domain))

        ts = thompson_sampling(batch_size=batch, duplicates=False)
        result = ts.run(model, obj)

        assert list(result.columns) == list(obj.domain.columns)


# ---------------------------------------------------------------------------
# random
# ---------------------------------------------------------------------------

class TestRandomAcquisition:
    def test_run_returns_correct_batch_size(self):
        batch = 3
        obj = _make_obj(n_domain=30, n_results=5)
        model = _make_model(n_domain=30)

        rand = random(batch_size=batch, duplicates=False)
        result = rand.run(model, obj)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == batch

    def test_run_with_duplicates(self):
        batch = 3
        obj = _make_obj(n_domain=30, n_results=5)
        model = _make_model(n_domain=30)

        rand = random(batch_size=batch, duplicates=True)
        result = rand.run(model, obj)
        assert len(result) == batch


# ---------------------------------------------------------------------------
# eps_greedy
# ---------------------------------------------------------------------------

class TestEpsGreedy:
    def test_run_returns_correct_batch_size(self):
        batch = 3
        obj = _make_obj(n_domain=30, n_results=5)
        model = _make_model(n_domain=30)

        eg = eps_greedy(batch_size=batch, duplicates=False)
        result = eg.run(model, obj)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == batch

    def test_default_epsilon(self):
        eg = eps_greedy(batch_size=1, duplicates=False)
        assert eg.eps == 0.05


# ---------------------------------------------------------------------------
# top_predicted / max_variance
# ---------------------------------------------------------------------------

class TestTopPredicted:
    def test_run_returns_dataframe(self):
        batch = 2
        n_domain = 20
        obj = _make_obj(n_domain=n_domain, n_results=3)
        model = _make_model(n_domain=n_domain)

        tp = top_predicted(batch_size=batch, duplicates=False)
        result = tp.run(model, obj)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == batch


class TestMaxVariance:
    def test_run_returns_dataframe(self):
        batch = 2
        n_domain = 20
        obj = _make_obj(n_domain=n_domain, n_results=3)
        model = _make_model(n_domain=n_domain)

        mv = max_variance(batch_size=batch, duplicates=False)
        result = mv.run(model, obj)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == batch

    def test_stores_variance(self):
        batch = 1
        n_domain = 20
        obj = _make_obj(n_domain=n_domain, n_results=3)
        model = _make_model(n_domain=n_domain)

        mv = max_variance(batch_size=batch, duplicates=False)
        mv.run(model, obj)
        assert hasattr(mv, "variance")
        assert len(mv.variance) == n_domain


# ---------------------------------------------------------------------------
# acquisition.evaluate (integration-like, still mocked)
# ---------------------------------------------------------------------------

class TestAcquisitionEvaluate:
    def test_evaluate_rand(self):
        obj = _make_obj(n_domain=30, n_results=5)
        model = _make_model(n_domain=30)
        acq = acquisition("rand", batch_size=3, duplicates=False)
        result = acq.evaluate(model, obj)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
