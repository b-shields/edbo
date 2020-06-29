#!/bin/bash

################################################# Get into edbo environment

eval "$(conda shell.bash hook)"
conda activate edbo

################################################# Install pytest

pip install pytest

################################################# Tests

pytest 1d_test.py > 1d_test.log                  # (1) Test GP model (predictions, variance estimation, sampling),
                                                 #     acquisition functions, and simulations.
pytest 1d_test_RF.py > 1d_test_RF.log            # (2) Test RF model (predictions, variance estimation, sampling),
                                                 #     acquisition functions, and simulations.
pytest 1d_test_GPU.py > 1d_test_GPU.log          # (3) Same as (2) but with gpu computation. Note a CUDA  GPU must be available.
pytest 1d_test_fast.py > 1d_test_fast.log        # (4) Same as (2) but with fast computation enabled.