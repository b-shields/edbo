# edbo

Experimental Design via Bayesian Optimization: *edbo* is a practical implementation of Bayesian optimization for chemical synthesis.

**Reference:** Shields, B. J.; Stevens, J.; Li, J.; Parasram, M.; Damani, F.; Alvarado, J. I. M.; Janey, J. M.; Adams, R. P.; Doyle, A. G. Bayesian Reaction Optimization as a Tool for Chemical Synthesis. Nature 2021, 590 (7844), 89–96. https://doi.org/10.1038/s41586-021-03213-y.

**Documentation:** https://b-shields.github.io/edbo/index.html

## Installation

(0) Create anaconda environment

```
conda create --name edbo python=3.7.5
```

(1) Install rdkit, Mordred, and PyTorch

```
conda activate edbo
conda install -c rdkit rdkit
conda install -c rdkit -c mordred-descriptor mordred
conda install -c pytorch pytorch=1.3.1
```

(2) Install EDBO

```
pip install edbo
```

### Running Notebooks

```
conda install jupyterlab
```
