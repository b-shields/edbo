#!/bin/bash

################################################# Get into edbo environment

eval "$(conda shell.bash hook)"
conda activate edbo

################################################# Install pytest

pip install pytest

################################################# Tests

pytest acquisition_functions_gp.py > acquisition_functions_gp.log
pytest acquisition_functions_rf.py > acquisition_functions_rf.log
pytest autobuild_objective.py > autobuild_objective.log
pytest init_methods_autobuilt.py > init_methods_autobuilt.log
pytest simulate_BO.py > simulate_BO.log
pytest simulate_BO_express.py > simulate_BO_express.log