# -*- coding: utf-8 -*-
"""Smoke tests for edbo.plot_utils – verify functions run without error."""

import matplotlib
matplotlib.use("Agg")  # headless backend, must be before pyplot import

import pytest
import numpy as np
import pandas as pd

from edbo.plot_utils import (
    plot2d,
    scatter,
    max_observed,
    rate,
    plot_convergence,
    average_convergence,
    plot_avg_convergence,
    compare_convergence,
    pred_obs,
    hor_bar,
    spearman_map,
    embedding_plot,
)


# ---------------------------------------------------------------------------
# Pure-computation helpers
# ---------------------------------------------------------------------------

class TestMaxObserved:
    def test_basic(self):
        points = np.array([1, 2, 3, 4, 5, 6])
        index, mo = max_observed(points, batch_size=2)
        assert index == [1, 2, 3]
        assert mo == [2, 4, 6]

    def test_single_batch(self):
        points = np.array([5, 3, 7])
        index, mo = max_observed(points, batch_size=3)
        assert index == [1]
        assert mo == [7]


class TestRate:
    def test_basic(self):
        seq = [1, 3, 6, 10]
        r = rate(seq)
        assert r == [2, 3, 4]

    def test_empty(self):
        assert rate([5]) == []


class TestAverageConvergence:
    def test_output_shapes(self):
        data = [[1, 2, 3, 4], [2, 3, 4, 5]]
        index, mean, std = average_convergence(data, partition=2)
        assert len(index) == 2
        assert len(mean) == 2
        assert len(std) == 2


# ---------------------------------------------------------------------------
# Smoke tests – just verify no exceptions
# ---------------------------------------------------------------------------

class TestPlot2dSmoke:
    def test_runs(self):
        plot2d([1, 2, 3], [4, 5, 6])


class TestScatterSmoke:
    def test_runs(self):
        scatter(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 2.9]),
            "test_label",
        )


class TestPlotConvergenceSmoke:
    def test_runs(self):
        plot_convergence(np.arange(1, 11), batch_size=2)


class TestPlotAvgConvergenceSmoke:
    def test_runs(self):
        data = [list(range(1, 11)), list(range(2, 12))]
        plot_avg_convergence(data, batch_size=2)


class TestCompareConvergenceSmoke:
    def test_runs(self):
        d1 = [list(range(1, 11)), list(range(2, 12))]
        d2 = [list(range(3, 13)), list(range(4, 14))]
        compare_convergence([d1, d2], batch_sizes=2)


class TestPredObsSmoke:
    def test_runs_plot(self):
        pred_obs(np.array([1, 2, 3]), np.array([1.1, 2.1, 3.1]))

    def test_return_data(self):
        result = pred_obs(
            np.array([1, 2, 3]), np.array([1.1, 2.1, 3.1]), return_data=True
        )
        assert isinstance(result, pd.DataFrame)
        assert "pred" in result.columns
        assert "obs" in result.columns

    def test_return_scores(self):
        rmse, r2 = pred_obs(
            np.array([1, 2, 3]), np.array([1.1, 2.1, 3.1]), return_scores=True
        )
        assert isinstance(rmse, float)
        assert isinstance(r2, float)


class TestHorBarSmoke:
    def test_runs(self):
        hor_bar(np.array([3, 1, 2]))


class TestSpearmanMapSmoke:
    def test_runs(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame(rng.rand(20, 4), columns=["a", "b", "c", "d"])
        spearman_map(df)


class TestEmbeddingPlotSmoke:
    def test_runs_without_labels(self):
        rng = np.random.RandomState(0)
        data = pd.DataFrame(rng.rand(30, 5))
        embedding_plot(data)

    def test_runs_with_labels(self):
        rng = np.random.RandomState(0)
        data = pd.DataFrame(rng.rand(30, 5))
        labels = [0] * 15 + [1] * 15
        embedding_plot(data, labels=labels)
