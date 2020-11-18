# edbo

Experimental Design via Bayesian Optimization: *edbo* is a practical implementation of Bayesian optimization for chemical synthesis.

**Reference:** Shields, Benjamin J.; Stevens, Jason; Li, Jun; Parasram, Marvin; Damani, Farhan, Janey, Jacob; Adams, Ryan P.; Doyle, Abigail G. "Bayesian Reaction Optimization as A Tool for Chemical Synthesis" Manuscript Submitted.

**Documentation:** https://b-shields.github.io/edbo/index.html

## Installation

(0) Create an anaconda environment

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

### GPU Integration

```
conda install cudatookit=10.1, torchvision -c pytorch
```
