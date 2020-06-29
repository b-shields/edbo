#!/bin/bash

################################################# Get into edbo environment

eval "$(conda shell.bash hook)"
conda activate edbo

################################################# Install pytest

pip install pytest

################################################# Tests

