# -*- coding: utf-8 -*-
"""Tests for edbo.objective — objective data container."""

import pytest
import pandas as pd
import numpy as np
import torch

from edbo.objective import objective


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_domain():
    """A small numeric domain DataFrame."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        'x1': rng.randn(20),
        'x2': rng.randn(20),
    })


@pytest.fixture
def simple_results():
    """Results DataFrame with two features and a target column."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        'x1': rng.randn(10),
        'x2': rng.randn(10),
        'yield': rng.rand(10) * 100,
    })


@pytest.fixture
def simple_exindex():
    """Experiment index for simulation lookups."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        'x1': rng.randn(20),
        'x2': rng.randn(20),
        'yield': rng.rand(20) * 100,
    })


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestObjectiveInit:

    def test_empty_domain(self):
        """Objective with only a domain, no results."""
        domain = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        obj = objective(domain=domain)
        assert len(obj.domain) == 3
        assert len(obj.results) == 0
        assert obj.X.shape == torch.Size([0])

    def test_with_results(self, simple_domain, simple_results):
        obj = objective(domain=simple_domain, results=simple_results)
        assert obj.target == 'yield'
        assert obj.X.shape[0] == 10
        assert obj.X.shape[1] == 2
        assert obj.y.shape == torch.Size([10])

    def test_target_auto_detected(self, simple_domain, simple_results):
        """When target=-1, the last column is used as target."""
        obj = objective(domain=simple_domain, results=simple_results)
        assert obj.target == 'yield'

    def test_target_explicit(self, simple_domain, simple_results):
        obj = objective(domain=simple_domain, results=simple_results, target='yield')
        assert obj.target == 'yield'

    def test_standardization(self, simple_domain, simple_results):
        """Results target should be standardized (mean ~ 0, std ~ 1)."""
        obj = objective(domain=simple_domain, results=simple_results)
        y_vals = obj.results['yield'].values
        assert abs(y_vals.mean()) < 1e-6
        assert abs(y_vals.std() - 1.0) < 0.1  # ddof differences possible

    def test_exindex_sets_target(self, simple_domain, simple_exindex):
        obj = objective(domain=simple_domain, exindex=simple_exindex)
        assert obj.target == 'yield'


# ---------------------------------------------------------------------------
# clear_results
# ---------------------------------------------------------------------------

class TestClearResults:

    def test_clear(self, simple_domain, simple_results):
        obj = objective(domain=simple_domain, results=simple_results)
        assert len(obj.results) > 0
        obj.clear_results()
        assert len(obj.results) == 0
        assert obj.X.shape == torch.Size([0])
        assert obj.y.shape == torch.Size([0])


# ---------------------------------------------------------------------------
# results_input (unstandardized results)
# ---------------------------------------------------------------------------

class TestResultsInput:

    def test_unstandardized_values(self, simple_domain, simple_results):
        obj = objective(domain=simple_domain, results=simple_results)
        raw = obj.results_input()
        # Unstandardized should be close to original input values
        np.testing.assert_allclose(
            raw['yield'].values,
            simple_results['yield'].values,
            atol=1e-4,
        )

    def test_empty_results(self, simple_domain):
        obj = objective(domain=simple_domain)
        raw = obj.results_input()
        assert len(raw) == 0


# ---------------------------------------------------------------------------
# get_results with computational objective
# ---------------------------------------------------------------------------

class TestGetResultsComputational:

    def test_computational_objective(self, simple_domain):
        fn = lambda x: float(np.sum(x))
        obj = objective(domain=simple_domain, computational_objective=fn, target='value')
        batch = simple_domain.head(3)
        result = obj.get_results(batch)
        assert 'value' in result.columns
        assert len(result) == 3

    def test_computational_objective_append(self, simple_domain):
        fn = lambda x: float(np.sum(x))
        results = simple_domain.head(5).copy()
        results['value'] = [fn(row) for row in results.values]
        obj = objective(domain=simple_domain, results=results,
                        computational_objective=fn, target='value')
        initial_len = len(obj.results)
        batch = simple_domain.tail(3)
        obj.get_results(batch, append=True)
        assert len(obj.results) == initial_len + 3


# ---------------------------------------------------------------------------
# get_results with exindex
# ---------------------------------------------------------------------------

class TestGetResultsExindex:

    def test_lookup_from_exindex(self, simple_domain, simple_exindex):
        obj = objective(domain=simple_domain, exindex=simple_exindex)
        # Select first few domain points — they should match exindex rows
        # because both use same random seed
        batch = simple_domain.head(3)
        result = obj.get_results(batch)
        assert 'yield' in result.columns
        assert len(result) == 3
