#!/bin/bash

################################################# Get into edbo environment

eval "$(conda shell.bash hook)"
conda activate edbo

################################################# Install pytest

pip install pytest

################################################# Tests

pytest test_test.py > test_test.py
pytest 1d_test_fast.py > 1d_test_fast.log