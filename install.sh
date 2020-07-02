#!/bin/bash

################################################# Create conda environment

echo "Creating conda environment..."

conda create -y --name edbo python=3.7.5

# get into environment
eval "$(conda shell.bash hook)"
conda activate edbo

################################################# Clone edbo

echo "Cloning source..."

git clone --single-branch --branch master https://github.com/b-shields/edbo.git

cd edbo

################################################# Install some dependencies

echo "#########################################"
echo "Installing dependencies..."
echo "-----------------------------------------"
echo "Responding 'y' to each prompt to proceed"
echo "#########################################"

conda install -y -c rdkit rdkit
conda install -y -c rdkit -c mordred-descriptor mordred
conda install -y pandas=0.25.3 numpy=1.17.4 xlrd
conda install -y pytorch=1.3.1 cudatoolkit=10.1 torchvision -c pytorch
conda install -y scikit-learn=0.22.1
conda install -y matplotlib seaborn
conda install -y dill

pip install gpytorch==1.0.0 pyclustering==0.9.3.1
pip install pyro-ppl==1.1

################################################# Install package namespace

echo "Installing edbo..."

python setup.py develop

################################################# Install an editor

conda install -y jupyterlab

#################################################

echo "Installation complete!"