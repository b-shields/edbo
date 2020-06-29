#!/bin/bash

################################################# Get into edbo environment

eval "$(conda shell.bash hook)"
conda activate edbo

################################################# Install pytest

pip install pytest

################################################# Tests

pytest bo_express_test.py > bo_express_test.log  # (1) Test GP model (predictions, variance estimation, sampling),
                                                 #     acquisition functions, and simulations.