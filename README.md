# edbo
Experimental Design via Bayesian Optimization

## Installation

0. Install conda (if you haven't already):

    See https://docs.conda.io/projects/conda/en/latest/user-guide/install/
	
1. Copy install.sh and run from anaconda prompt:
	
	chmod install.sh (linux only)
	sh install.sh	
	
	
	
	


1. Create a conda environment fro edbo:

    conda create --name edbo python=3.7.5

2. Clone the edbo repository:

    git clone https://github.com/b-shields/edbo.git

3. Install edbo:
	
	chmod install.sh
	sh install.sh

	# Editing
    conda install -c rdkit rdkit
	conda install jupyter pandas=0.25.3 numpy=1.17.4 pytorch=1.3.1 scikit-learn>=0.22.1 matplotlib seaborn
    pip install -r requirements.txt

    