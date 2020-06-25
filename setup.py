from setuptools import setup, find_packages

setup(
   name='edbo',
   version='0.0.0',
   author='Ben Shields',
   packages=find_packages()
   url='https://github.com/b-shields/edbo.git',
   license='LICENSE',
   description='Experimental design via Bayesian optimization',
   install_requires=[
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