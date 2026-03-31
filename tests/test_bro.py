# -*- coding: utf-8 -*-
"""Tests for edbo.bro module (BO and BO_express classes)."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import tempfile
import os

from edbo.bro import BO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_domain(n=30, d=3):
    """Create a small numeric domain DataFrame."""
    rng = np.random.RandomState(42)
    cols = [f"x{i}" for i in range(d)]
    return pd.DataFrame(rng.rand(n, d), columns=cols)


def _simple_results(domain, n_results=5, target="yield"):
    """Create results by selecting from domain and adding a target column."""
    results = domain.iloc[:n_results].copy()
    results[target] = np.random.RandomState(7).rand(n_results)
    return results


def _simple_exindex(domain, target="yield"):
    """Create an experiment index with known yields for simulation."""
    exindex = domain.copy()
    exindex[target] = np.random.RandomState(99).rand(len(domain))
    return exindex


# ---------------------------------------------------------------------------
# BO construction
# ---------------------------------------------------------------------------

class TestBOConstruction:
    def test_init_with_domain_dataframe(self):
        domain = _simple_domain()
        bo = BO(domain=domain, batch_size=3)
        assert bo.batch_size == 3
        assert bo.obj.domain is not None
        assert len(bo.obj.domain) == len(domain)

    def test_init_default_target(self):
        domain = _simple_domain()
        bo = BO(domain=domain)
        # target defaults to -1 until results are provided
        assert bo.obj.target == -1

    def test_init_with_results(self):
        domain = _simple_domain()
        results = _simple_results(domain)
        bo = BO(domain=domain, results=results, batch_size=3)
        assert bo.obj.target == "yield"
        assert len(bo.obj.results) == 5

    def test_init_with_exindex(self):
        domain = _simple_domain()
        exindex = _simple_exindex(domain)
        bo = BO(domain=domain, exindex=exindex, batch_size=3)
        assert bo.obj.target == "yield"

    def test_batch_size_stored(self):
        domain = _simple_domain()
        bo = BO(domain=domain, batch_size=7)
        assert bo.batch_size == 7

    def test_acquisition_function_variants(self):
        domain = _simple_domain()
        for func in ["TS", "EI", "PI", "UCB", "rand"]:
            bo = BO(domain=domain, acquisition_function=func, batch_size=2)
            assert bo.acq is not None


# ---------------------------------------------------------------------------
# BO.init_sample
# ---------------------------------------------------------------------------

class TestBOInitSample:
    def test_init_sample_rand(self):
        domain = _simple_domain(n=50)
        bo = BO(domain=domain, batch_size=5, init_method="rand")
        proposed = bo.init_sample(seed=42)
        assert isinstance(proposed, pd.DataFrame)
        assert len(proposed) == 5

    def test_init_sample_returns_domain_columns(self):
        domain = _simple_domain(n=50)
        bo = BO(domain=domain, batch_size=3, init_method="rand")
        proposed = bo.init_sample(seed=1)
        assert list(proposed.columns) == list(domain.columns)

    def test_init_sample_with_append_computational(self):
        domain = _simple_domain(n=50)
        comp_obj = lambda x: float(np.sum(x))
        bo = BO(domain=domain, batch_size=3, init_method="rand",
                computational_objective=comp_obj, target="yield")
        proposed = bo.init_sample(seed=1, append=True)
        assert len(bo.obj.results) == 3


# ---------------------------------------------------------------------------
# BO.clear_results
# ---------------------------------------------------------------------------

class TestBOClearResults:
    def test_clear_results(self):
        domain = _simple_domain(n=50)
        comp_obj = lambda x: float(np.sum(x))
        bo = BO(domain=domain, batch_size=3, init_method="rand",
                computational_objective=comp_obj, target="yield")
        bo.init_sample(seed=1, append=True)
        assert len(bo.obj.results) > 0
        bo.clear_results()
        assert len(bo.obj.results) == 0


# ---------------------------------------------------------------------------
# BO.best
# ---------------------------------------------------------------------------

class TestBOBest:
    def test_best_returns_sorted(self):
        domain = _simple_domain(n=50)
        results = _simple_results(domain, n_results=10)
        bo = BO(domain=domain, results=results, batch_size=3)
        best = bo.best()
        assert isinstance(best, pd.DataFrame)
        assert len(best) <= 10


# ---------------------------------------------------------------------------
# BO.save / BO.load
# ---------------------------------------------------------------------------

class TestBOSaveLoad:
    def test_save_and_load(self, tmp_path):
        domain = _simple_domain()
        results = _simple_results(domain)
        bo = BO(domain=domain, results=results, batch_size=3)

        path = str(tmp_path / "bo_state.pkl")
        bo.save(path=path)
        assert os.path.exists(path)

        bo2 = BO(domain=domain, batch_size=3)
        bo2.load(path=path)
        assert bo2.obj.target == "yield"
        assert bo2.batch_size == 3


# ---------------------------------------------------------------------------
# BO.fit  (slow – actually trains a small GP)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBOFit:
    def test_fit_creates_trained_model(self):
        domain = _simple_domain(n=30, d=2)
        results = _simple_results(domain, n_results=8)
        bo = BO(domain=domain, results=results, batch_size=3)
        bo.fit(training_iters=5)
        # After fit, model should have predict/variance
        assert hasattr(bo.model, "predict")
        assert hasattr(bo.model, "variance")


# ---------------------------------------------------------------------------
# BO.run  (slow – trains GP + evaluates acquisition)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBORun:
    def test_run_returns_proposals(self):
        domain = _simple_domain(n=30, d=2)
        results = _simple_results(domain, n_results=8)
        bo = BO(domain=domain, results=results, batch_size=3,
                acquisition_function="rand")
        proposed = bo.run(training_iters=5)
        assert isinstance(proposed, pd.DataFrame)
        assert len(proposed) == 3


# ---------------------------------------------------------------------------
# BO.simulate  (slow – full loop)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBOSimulate:
    def test_simulate_computational_objective(self):
        domain = _simple_domain(n=30, d=2)
        comp_obj = lambda x: float(np.sum(x))
        bo = BO(domain=domain, batch_size=3,
                acquisition_function="rand",
                computational_objective=comp_obj,
                target="yield")
        bo.simulate(iterations=1, seed=42, training_iters=5)
        # After simulate, results should have grown
        assert len(bo.obj.results) > 3


# ---------------------------------------------------------------------------
# BO.acquisition_summary  (slow – requires trained model)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBOAcquisitionSummary:
    def test_acquisition_summary_columns(self):
        domain = _simple_domain(n=30, d=2)
        results = _simple_results(domain, n_results=8)
        bo = BO(domain=domain, results=results, batch_size=2,
                acquisition_function="rand")
        bo.run(training_iters=5)
        summary = bo.acquisition_summary()
        assert "predicted yield" in summary.columns
        assert "variance" in summary.columns
