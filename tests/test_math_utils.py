"""Tests for edbo.math_utils — standard, model_performance, pca."""

import numpy as np
import pandas as pd
import pytest

from edbo.math_utils import standard, model_performance, pca


# ---------------------------------------------------------------------------
# standard class
# ---------------------------------------------------------------------------

class TestStandard:

    def test_standardize_target_zero_mean_unit_var(self, sample_df):
        s = standard()
        result = s.standardize_target(sample_df, 'yield')

        vals = result['yield'].values
        assert abs(vals.mean()) < 1e-10
        assert abs(vals.std(ddof=0) - 1.0) < 1e-6  # population std

    def test_unstandardize_recovers_original(self, sample_df):
        s = standard()
        std_df = s.standardize_target(sample_df.copy(), 'yield')
        recovered = s.unstandardize_target(std_df, 'yield')

        np.testing.assert_allclose(
            recovered['yield'].values,
            sample_df['yield'].values,
            atol=1e-10,
        )

    def test_standardize_single_row_returns_unchanged(self):
        df = pd.DataFrame({'x': [1.0], 'yield': [50.0]})
        s = standard()
        result = s.standardize_target(df, 'yield')
        assert result.equals(df)

    def test_unstandardize_single_row_returns_unchanged(self):
        df = pd.DataFrame({'x': [1.0], 'yield': [50.0]})
        s = standard()
        result = s.unstandardize_target(df, 'yield')
        assert result.equals(df)

    def test_standardize_zero_std(self):
        """All-same targets should not raise (std forced to 1e-6)."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'yield': [5.0, 5.0, 5.0]})
        s = standard()
        result = s.standardize_target(df, 'yield')
        assert s.std == 1e-6

    def test_unstandardize_array(self, sample_df):
        s = standard()
        s.standardize_target(sample_df, 'yield')
        arr = np.array([0.0, 1.0, -1.0])
        result = s.unstandardize(arr)
        expected = arr * s.std + s.mean
        np.testing.assert_allclose(result, expected)

    def test_unstandardize_new_df_different_length(self, sample_df):
        """Unstandardize a df whose length differs from original."""
        s = standard()
        s.standardize_target(sample_df, 'yield')

        new_df = pd.DataFrame({
            'feat1': [0.0, 0.0],
            'feat2': [0.0, 0.0],
            'feat3': [0.0, 0.0],
            'yield': [1.0, -1.0],
        })
        result = s.unstandardize_target(new_df, 'yield')
        expected = np.array([1.0, -1.0]) * s.std + s.mean
        np.testing.assert_allclose(result['yield'].values, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# model_performance
# ---------------------------------------------------------------------------

class TestModelPerformance:

    def test_perfect_prediction(self):
        obs = [1, 2, 3, 4, 5]
        rmse, r2 = model_performance(obs, obs)
        assert rmse == pytest.approx(0.0)
        assert r2 == pytest.approx(1.0)

    def test_known_values(self):
        pred = [1.0, 2.0, 3.0]
        obs = [1.1, 2.1, 2.9]
        rmse, r2 = model_performance(pred, obs)
        assert rmse > 0
        assert r2 < 1.0

    def test_returns_tuple(self):
        result = model_performance([1, 2], [1, 2])
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# pca
# ---------------------------------------------------------------------------

class TestPCA:

    def test_output_shape(self):
        X = np.random.randn(20, 5)
        result = pca(X, n_components=2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (20, 2)

    def test_column_names(self):
        X = np.random.randn(15, 4)
        result = pca(X, n_components=3)
        assert list(result.columns) == ['pc1', 'pc2', 'pc3']

    def test_single_component(self):
        X = np.random.randn(10, 3)
        result = pca(X, n_components=1)
        assert result.shape[1] == 1
        assert result.columns[0] == 'pc1'
