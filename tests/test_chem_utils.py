# -*- coding: utf-8 -*-
"""Tests for edbo.chem_utils module."""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# name_to_smiles  (network-dependent, so we mock urlopen)
# ---------------------------------------------------------------------------

class TestNameToSmiles:
    def test_successful_lookup(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import name_to_smiles

        mock_response = MagicMock()
        mock_response.read.return_value = b"c1ccccc1"

        with patch("edbo.chem_utils.urlopen", return_value=mock_response):
            result = name_to_smiles("benzene")

        assert result == "c1ccccc1"

    def test_failed_lookup_returns_failed(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import name_to_smiles

        with patch("edbo.chem_utils.urlopen", side_effect=Exception("network error")):
            result = name_to_smiles("nonexistent_compound_xyz")

        assert result == "FAILED"

    def test_html_response_returns_failed(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import name_to_smiles

        mock_response = MagicMock()
        mock_response.read.return_value = b"<div>Error page</div>"

        with patch("edbo.chem_utils.urlopen", return_value=mock_response):
            result = name_to_smiles("bad_name")

        assert result == "FAILED"

    def test_space_encoding(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import name_to_smiles

        mock_response = MagicMock()
        mock_response.read.return_value = b"CC(=O)O"

        with patch("edbo.chem_utils.urlopen", return_value=mock_response) as mock_url:
            name_to_smiles("acetic acid")
            call_args = mock_url.call_args[0][0]
            assert "%20" in call_args
            assert " " not in call_args


# ---------------------------------------------------------------------------
# ChemDraw class  (requires rdkit)
# ---------------------------------------------------------------------------

class TestChemDraw:
    def test_init_creates_mols(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import ChemDraw

        smiles = ["c1ccccc1", "CC", "CCO"]
        cdx = ChemDraw(smiles)

        assert len(cdx.mols) == 3
        assert cdx.SMILES_list == smiles
        assert cdx.molsPerRow == 3  # auto

    def test_init_custom_row_size(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import ChemDraw

        cdx = ChemDraw(["c1ccccc1", "CC"], row_size=1)
        assert cdx.molsPerRow == 1

    def test_init_with_legends(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import ChemDraw

        legends = ["benzene", "ethane"]
        cdx = ChemDraw(["c1ccccc1", "CC"], legends=legends)
        assert cdx.legends == legends

    def test_mols_are_valid(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import ChemDraw

        cdx = ChemDraw(["c1ccccc1", "CCO"])
        for mol in cdx.mols:
            assert mol is not None

    def test_invalid_smiles_returns_none_mol(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import ChemDraw

        cdx = ChemDraw(["INVALID_SMILES"])
        assert cdx.mols[0] is None

    def test_show_calls_display(self):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import ChemDraw

        cdx = ChemDraw(["c1ccccc1"], ipython_svg=False)
        with patch("edbo.chem_utils.display.display") as mock_display:
            cdx.show()
            mock_display.assert_called_once()

    def test_export_creates_file(self, tmp_path):
        rdkit = pytest.importorskip("rdkit")
        from edbo.chem_utils import ChemDraw

        cdx = ChemDraw(["c1ccccc1"], ipython_svg=False)
        export_base = str(tmp_path / "test_mol")
        cdx.export(export_base)
        assert (tmp_path / "test_mol.png").exists()
