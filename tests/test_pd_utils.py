"""Tests for edbo.pd_utils — I/O, to_torch, torch_to_numpy, complement, argmax, join_to_df."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from edbo.pd_utils import (
    load_csv_or_excel,
    load_experiment_results,
    write_experiment_results,
    to_torch,
    torch_to_numpy,
    complement,
    join_to_df,
    argmax,
)


# ---------------------------------------------------------------------------
# load_csv_or_excel
# ---------------------------------------------------------------------------

class TestLoadCsvOrExcel:

    def test_load_csv(self, tmp_path):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        path = tmp_path / 'data.csv'
        df.to_csv(path, index=False)
        result = load_csv_or_excel(str(path))
        assert list(result.columns) == ['a', 'b']
        assert len(result) == 2

    def test_load_xlsx(self, tmp_path):
        pytest.importorskip('openpyxl')
        df = pd.DataFrame({'x': [10, 20]})
        path = tmp_path / 'data.xlsx'
        df.to_excel(path, index=False)
        result = load_csv_or_excel(str(path))
        assert len(result) == 2

    def test_unknown_extension_returns_empty(self, tmp_path):
        path = tmp_path / 'data.txt'
        path.write_text('hello')
        result = load_csv_or_excel(str(path))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# load_experiment_results / write_experiment_results
# ---------------------------------------------------------------------------

class TestExperimentResults:

    def test_load_single_file(self, tmp_path):
        df = pd.DataFrame({'a': [1, 2]})
        path = tmp_path / 'results.csv'
        df.to_csv(path, index=False)
        result = load_experiment_results(str(path))
        assert len(result) == 2

    def test_load_directory(self, tmp_path):
        for i in range(3):
            pd.DataFrame({'a': [i]}).to_csv(tmp_path / f'batch{i}.csv', index=False)
        result = load_experiment_results(str(tmp_path))
        assert len(result) == 3

    def test_write_and_load(self, tmp_path):
        results_dir = tmp_path / 'results'
        results_dir.mkdir()
        df = pd.DataFrame({'a': [1, 2, 3]})
        write_experiment_results(df, str(results_dir))
        loaded = load_experiment_results(str(results_dir))
        assert len(loaded) == 3

    def test_load_nonexistent(self, tmp_path):
        result = load_experiment_results(str(tmp_path / 'nonexistent'))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# to_torch / torch_to_numpy
# ---------------------------------------------------------------------------

class TestTorchConversions:

    def test_to_torch_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        t = to_torch(arr)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32

    def test_to_torch_from_dataframe(self, sample_df):
        t = to_torch(sample_df[['feat1', 'feat2']])
        assert isinstance(t, torch.Tensor)
        assert t.shape[0] == len(sample_df)

    def test_to_torch_from_tensor(self):
        t = torch.tensor([1.0, 2.0])
        result = to_torch(t)
        assert torch.equal(t, result)

    def test_to_torch_from_list(self):
        result = to_torch([1.0, 2.0, 3.0])
        assert isinstance(result, torch.Tensor)

    def test_torch_to_numpy_basic(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        arr = torch_to_numpy(t)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_torch_to_numpy_plain_array(self):
        arr = np.array([4.0, 5.0])
        result = torch_to_numpy(arr)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# complement
# ---------------------------------------------------------------------------

class TestComplement:

    def test_basic_complement(self):
        df1 = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        df2 = pd.DataFrame({'a': [2, 4], 'b': [20, 40]})
        result = complement(df1, df2)
        assert len(result) == 2
        assert set(result['a'].values) == {1, 3}

    def test_complement_boolean_out(self):
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        df2 = pd.DataFrame({'a': [2], 'b': [20]})
        bools = complement(df1, df2, boolean_out=True)
        assert isinstance(bools, list)
        assert len(bools) == 3

    def test_complement_with_rounding(self):
        df1 = pd.DataFrame({'a': [1.001, 2.002, 3.003]})
        df2 = pd.DataFrame({'a': [2.0]})
        result = complement(df1, df2, rounding=1)
        # After rounding to 1 decimal, 2.002 -> 2.0 matches df2
        assert len(result) == 2

    def test_complement_empty_df2(self):
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': []}).astype(int)
        result = complement(df1, df2)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# join_to_df
# ---------------------------------------------------------------------------

class TestJoinToDf:

    def test_join_to_df_basic(self):
        domain = pd.DataFrame({'x1': [1.0, 2.0, 3.0], 'x2': [4.0, 5.0, 6.0]})
        sample = torch.tensor([10.0, 20.0, 30.0])
        result = join_to_df(sample, domain)
        assert 'sample' in result.columns
        assert len(result) == 3
        assert list(result['sample'].values) == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# argmax
# ---------------------------------------------------------------------------

class TestArgmax:

    def test_argmax_with_duplicates(self):
        sample_xy = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'sample': [10, 40, 30, 20],
        })
        known = pd.DataFrame({'x': [99]})  # no overlap
        result = argmax(sample_xy, known, duplicates=True)
        assert len(result) == 1
        assert result['sample'].values[0] == 40

    def test_argmax_without_duplicates(self):
        sample_xy = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'sample': [10, 40, 30, 20],
        })
        known = pd.DataFrame({'x': [2]})
        result = argmax(sample_xy, known, duplicates=False, top_n=1)
        assert len(result) == 1
        # Should skip x=2 (known) and pick x=3 (next highest)
        assert result['sample'].values[0] == 30

    def test_argmax_top_n(self):
        sample_xy = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'sample': [10, 50, 30, 40, 20],
        })
        known = pd.DataFrame({'x': [99]})
        result = argmax(sample_xy, known, duplicates=False, top_n=2)
        assert len(result) == 2
