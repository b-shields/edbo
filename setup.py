from setuptools import setup

setup(
   name='edbo',
   packages=['edbo'], 
   version='0.1.0',
   author='Benjamin J. Shields',
   author_email='shields.benjamin.j@gmail.com',
   url='https://github.com/b-shields/edbo',
   download_url = 'https://github.com/b-shields/edbo/archive/v_010.tar.gz',
   keywords=['Bayesian Optimization', 'Chemical Reaction Optimization'],
   license='MIT',
   description='Experimental design via Bayesian optimization',
   install_requires=[
        'pandas',
        'numpy',
        'xlrd',
        'scikit-learn>=0.22.1',
        'matplotlib',
        'seaborn',
        'dill',
        'gpytorch==1.0.0',
        'pyclustering==0.9.3.1',
        'pyro-ppl==1.1',
        'ipython', ###
    ],
   classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Chemists', 
    'Topic :: Bayesian Optimization :: Chemistry',
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3.7',
  ],
)