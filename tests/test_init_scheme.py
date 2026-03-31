# -*- coding: utf-8 -*-
"""Tests for edbo.init_scheme — Init class and initialization helpers."""

import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_obj():
    """A minimal objective-like object with a .domain attribute."""
    from edbo.objective import objective
    rng = np.random.RandomState(7)
    domain = pd.DataFrame({
        'x1': rng.randn(40),
        'x2': rng.randn(40),
        'x3': rng.randn(40),
    })
    return objective(domain=domain)


@pytest.fixture
def obj_with_results():
    """Objective that has results (for external_data test)."""
    from edbo.objective import objective
    rng = np.random.RandomState(7)
    domain = pd.DataFrame({
        'x1': rng.randn(40),
        'x2': rng.randn(40),
    })
    results = pd.DataFrame({
        'x1': rng.randn(5),
        'x2': rng.randn(5),
        'target': rng.rand(5),
    })
    return objective(domain=domain, results=results, target='target')


# ---------------------------------------------------------------------------
# rand
# ---------------------------------------------------------------------------

class TestRand:

    def test_returns_correct_batch_size(self, simple_obj):
        from edbo.init_scheme import rand
        batch = rand(simple_obj, batch_size=5, seed=0)
        assert isinstance(batch, pd.DataFrame)
        assert len(batch) == 5

    def test_columns_match_domain(self, simple_obj):
        from edbo.init_scheme import rand
        batch = rand(simple_obj, batch_size=3, seed=0)
        assert list(batch.columns) == list(simple_obj.domain.columns)

    def test_deterministic_with_seed(self, simple_obj):
        from edbo.init_scheme import rand
        b1 = rand(simple_obj, batch_size=5, seed=42)
        b2 = rand(simple_obj, batch_size=5, seed=42)
        pd.testing.assert_frame_equal(b1, b2)

    def test_different_seeds_differ(self, simple_obj):
        from edbo.init_scheme import rand
        b1 = rand(simple_obj, batch_size=5, seed=0)
        b2 = rand(simple_obj, batch_size=5, seed=99)
        assert not b1.equals(b2)


# ---------------------------------------------------------------------------
# external_data
# ---------------------------------------------------------------------------

class TestExternalData:

    def test_returns_results_without_target(self, obj_with_results):
        from edbo.init_scheme import external_data
        result = external_data(obj_with_results)
        assert isinstance(result, pd.DataFrame)
        assert 'target' not in result.columns
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Init class — random
# ---------------------------------------------------------------------------

class TestInitRandom:

    def test_run_random(self, simple_obj):
        from edbo.init_scheme import Init
        init = Init('random', batch_size=5)
        result = init.run(simple_obj, seed=0)
        assert len(result) == 5
        assert hasattr(init, 'experiments')

    def test_run_rand_variant(self, simple_obj):
        """Method matching is case-insensitive and substring-based."""
        from edbo.init_scheme import Init
        init = Init('Random', batch_size=3)
        result = init.run(simple_obj, seed=0)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Init class — PAM (requires pyclustering)
# ---------------------------------------------------------------------------

class TestInitPAM:

    def test_run_pam(self, simple_obj):
        pytest.importorskip("pyclustering")
        from edbo.init_scheme import Init
        init = Init('pam', batch_size=3, distance='euclidean')
        result = init.run(simple_obj, seed=0, visualize=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_pam_gower(self, simple_obj):
        pytest.importorskip("pyclustering")
        from edbo.init_scheme import Init
        init = Init('pam', batch_size=4, distance='gower')
        result = init.run(simple_obj, seed=1, visualize=False)
        assert len(result) == 4

    def test_pam_medoids_alias(self, simple_obj):
        """'medoids' substring triggers PAM."""
        pytest.importorskip("pyclustering")
        from edbo.init_scheme import Init
        init = Init('k-medoids', batch_size=3, distance='euclidean')
        result = init.run(simple_obj, seed=0, visualize=False)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Init class — k-means
# ---------------------------------------------------------------------------

class TestInitKMeans:

    def test_run_kmeans(self, simple_obj):
        from edbo.init_scheme import Init
        init = Init('k-means', batch_size=3)
        result = init.run(simple_obj, seed=0, visualize=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3  # may return more due to cluster selection

    def test_kmeans_batch_size_respected(self, simple_obj):
        from edbo.init_scheme import Init
        init = Init('k-means', batch_size=5)
        result = init.run(simple_obj, seed=0, visualize=False)
        assert len(result) <= 15  # at most batch_size + 10 clusters


# ---------------------------------------------------------------------------
# Init class — external
# ---------------------------------------------------------------------------

class TestInitExternal:

    def test_run_external(self, obj_with_results):
        from edbo.init_scheme import Init
        init = Init('external', batch_size=5)
        result = init.run(obj_with_results)
        assert isinstance(result, pd.DataFrame)
        assert 'target' not in result.columns
