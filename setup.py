from setuptools import setup

setup(
   name='edbo',
   version='0.0.0',
   author='Ben Shields',
   packages=['edbo', 'edbo.acq_func', 'edbo.base_models', 'edbo.bro', 'edbo.chem_utils', 'edbo.encode', 'edbo.feature_selection', 'edbo.feature_utils', 'edbo.init_scheme', 'edbo.math_utils', 'edbo.models', 'edbo.objective', 'edbo.opt_utils', 'edbo.pd_utils', 'edbo.plot_utils', 'edbo.torch_utils', 'edbo.utils'],
   url='https://github.com/b-shields/edbo.git',
   license='LICENSE',
   description='Experimental design via Bayesian optimization',
   install_requires=[
        'jupyter',
        'pandas==0.25.3',
        'numpy==1.17.4',
        'pytorch==1.3.1',
        'scikit-learn>=0.22.1',
        'matplotlib',
        'seaborn',
        'gpytorch==1.0.0',
        'pyclustering==0.9.3.1'
    ]
)