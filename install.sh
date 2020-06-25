#!/bin/bash

################################################# Create conda environment

echo "Creating conda environment..."

conda create --name edbo python=3.7.5

# get into environment
conda activate edbo

################################################# Clone edbo

echo "Cloning source..."

git clone https://github.com/b-shields/edbo.git

################################################# Install conda

echo "Installing dependencies via conda..."

# rdkit
conda install -c rdkit rdkit
# rest of dependencies in requirements_conda.txt
while read requirement; do conda install --yes $requirement; done < requirements_conda.txt

################################################# Install pip

echo "Installing dependencies via pip..."

pip install -r requirements_pip.txt

################################################# Install package namespace

echo "Installing edbo..."

python setup.py install

#################################################

echo "Installation complete!"