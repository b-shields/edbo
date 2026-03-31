# -*- coding: utf-8 -*-
"""Tests for edbo.opt_utils — likelihood optimization functions."""

import pytest
import torch
import numpy as np
from gpytorch.priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
from unittest.mock import patch, MagicMock
from copy import deepcopy

from edbo.opt_utils import build_dist_dict, set_init_params, optimize_mll
from edbo.base_models import gp_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_gp_data():
    """Small training data for GP tests."""
    torch.manual_seed(42)
    X = torch.randn(15, 3)
    y = torch.randn(15)
    return X, y


@pytest.fixture
def small_model(small_gp_data):
    """A gp_model + likelihood ready for optimisation."""
    X, y = small_gp_data
    likelihood = GaussianLikelihood()
    model = gp_model(X, y, likelihood)
    return model, likelihood, X, y


# ---------------------------------------------------------------------------
# build_dist_dict
# ---------------------------------------------------------------------------

class TestBuildDistDict:

    def test_default_priors(self):
        """All None priors should produce GammaPrior distributions."""
        d = build_dist_dict(None, None, None)
        assert set(d.keys()) == {
            'likelihood.noise_covar.raw_noise',
            'covar_module.raw_outputscale',
            'covar_module.base_kernel.raw_lengthscale',
        }
        # Default distributions should be GammaPrior instances
        for v in d.values():
            assert isinstance(v, GammaPrior)

    def test_custom_noise_prior(self):
        """Custom noise prior should override the default."""
        custom = GammaPrior(2.0, 1.0)
        d = build_dist_dict([custom], None, None)
        assert d['likelihood.noise_covar.raw_noise'] is custom

    def test_custom_outputscale_prior(self):
        custom = GammaPrior(5.0, 2.0)
        d = build_dist_dict(None, [custom], None)
        assert d['covar_module.raw_outputscale'] is custom

    def test_custom_lengthscale_prior(self):
        custom = GammaPrior(4.0, 1.5)
        d = build_dist_dict(None, None, [custom])
        assert d['covar_module.base_kernel.raw_lengthscale'] is custom


# ---------------------------------------------------------------------------
# set_init_params
# ---------------------------------------------------------------------------

class TestSetInitParams:

    def test_returns_new_dict(self, small_model):
        """set_init_params should not mutate the original dictionary."""
        model, _, _, _ = small_model
        state = model.state_dict()
        dists = build_dist_dict(None, None, None)
        new_state = set_init_params(state, dists, seed=0)
        # Keys should match
        assert set(new_state.keys()) == set(state.keys())
        # Values for distribution keys should differ (with high probability)
        for key in dists:
            assert not torch.equal(new_state[key], state[key])

    def test_deterministic_with_same_seed(self, small_model):
        """Same seed should produce identical parameters."""
        model, _, _, _ = small_model
        state = model.state_dict()
        dists = build_dist_dict(None, None, None)
        s1 = set_init_params(state, dists, seed=7)
        s2 = set_init_params(state, dists, seed=7)
        for key in dists:
            assert torch.equal(s1[key], s2[key])

    def test_different_seeds_differ(self, small_model):
        model, _, _, _ = small_model
        state = model.state_dict()
        dists = build_dist_dict(None, None, None)
        s1 = set_init_params(state, dists, seed=0)
        s2 = set_init_params(state, dists, seed=99)
        any_differ = any(
            not torch.equal(s1[k], s2[k]) for k in dists
        )
        assert any_differ


# ---------------------------------------------------------------------------
# optimize_mll
# ---------------------------------------------------------------------------

class TestOptimizeMll:

    @pytest.mark.slow
    def test_returns_loss_list(self, small_model):
        """optimize_mll should return a list of loss trajectories."""
        model, likelihood, X, y = small_model
        losses = optimize_mll(model, likelihood, X, y,
                              training_iters=5, n_restarts=0)
        assert isinstance(losses, list)
        assert len(losses) == 1  # n_restarts=0 => 1 trajectory
        assert len(losses[0]) == 5

    @pytest.mark.slow
    def test_restarts_produce_multiple_trajectories(self, small_model):
        """n_restarts=2 should give 3 loss trajectories."""
        model, likelihood, X, y = small_model
        losses = optimize_mll(model, likelihood, X, y,
                              training_iters=3, n_restarts=2)
        assert len(losses) == 3

    @pytest.mark.slow
    def test_loss_decreases(self, small_model):
        """The loss should generally decrease over training iterations."""
        model, likelihood, X, y = small_model
        losses = optimize_mll(model, likelihood, X, y,
                              training_iters=30, n_restarts=0)
        trajectory = losses[0]
        # Final loss should be less than initial loss
        assert trajectory[-1] < trajectory[0]

    @pytest.mark.slow
    def test_model_in_train_mode_after(self, small_model):
        """After optimization the model stays in train mode (by design)."""
        model, likelihood, X, y = small_model
        optimize_mll(model, likelihood, X, y, training_iters=3)
        # The function does not switch to eval mode
        assert model.training

    def test_custom_priors_accepted(self, small_model):
        """Custom priors should be accepted without error."""
        model, likelihood, X, y = small_model
        noise_p = [GammaPrior(2.0, 1.0)]
        out_p = [GammaPrior(2.0, 1.0)]
        ls_p = [GammaPrior(2.0, 1.0)]
        losses = optimize_mll(model, likelihood, X, y,
                              training_iters=2, n_restarts=0,
                              noise_prior=noise_p,
                              outputscale_prior=out_p,
                              lengthscale_prior=ls_p)
        assert len(losses) == 1
