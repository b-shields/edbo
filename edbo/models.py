# -*- coding: utf-8 -*-

# Imports

import sys
import torch

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan

from sklearn.linear_model import ARDRegression
from sklearn.model_selection import GridSearchCV

import numpy as np
import warnings

from base_models import gp_model, random_forest

from plot_utils import pred_obs
from torch_utils import cv_split
from math_utils import model_performance
from pd_utils import to_torch
from opt_utils import optimize_mll
    
# Gaussian Process Model
        
class GP_Model:
    """Main gaussian process model used for Bayesian optimization.
    
    Provides a framework for specifiying exact GP models, hyperparameters, and 
    priors. This class also contains functions for training, sampling, forward 
    prediction, and variance estimation.
    """
    
    def __init__(self, X, y, training_iters=100, inference_type='MLE', 
                 learning_rate=0.1, noise_constraint=1e-5, gpu=False, nu=2.5,
                 lengthscale_prior=None, outputscale_prior=None,
                 noise_prior=None, n_restarts=0
                 ):
        """        
        Parameters
        ----------
        X : torch.tensor
            Training domain values.
        y : torch.tensor
            Training response values.
        training_iters : int
            Number of iterations to run ADAM optimizer durring training.
        inference_type : str
            Estimation procedue to be used. Currently only MLE is availible.
        learning_rate : float
            Learning rate for ADMA optimizer durring training.
        noise_constraint : float
            Noise is constrained to be positive. Set's the minimum noise level.
        gpu : bool 
            Use GPUs (if available) to run gaussian process computations. 
        nu : float 
            Matern kernel parameter. Options: 0.5, 1.5, 2.5.
        lengthscale_prior : [gpytorch.priors, init_value] 
            GPyTorch prior object and initial value. Sets a prior over length 
            scales.
        outputscale_prior : [gpytorch.priors, init_value] 
            GPyTorch prior object and initial value. Sets a prior over output
            scales.
        noise_prior : [gpytorch.priors, init_value]
            GPyTorch prior object and initial value. Sets a prior over output
            scales.
        n_restarts : int
            Number of random restarts for model training.
        
        Returns
        ----------
        None.
        """ 
        
        if inference_type == 'MCMC': print('Inference type not yet supported')
        
        # Initialization of main model components
        self.X = X
        self.y = y
        self.training_iters = training_iters
        self.inference_type = inference_type
        self.learning_rate = learning_rate
        self.noise_constraint = noise_constraint
        self.gpu = gpu  
        self.n_restarts = n_restarts
        self.lengthscale_prior = lengthscale_prior
        self.outputscale_prior = outputscale_prior
        self.noise_prior = noise_prior
        
        # Configure likelihood
        self.likelihood = GaussianLikelihood()
        if noise_prior != None:
            self.likelihood = GaussianLikelihood(noise_prior=noise_prior[0])
            self.likelihood.noise = torch.tensor([float(noise_prior[1])])
        
        # Set model
        self.model = gp_model(self.X, 
                              self.y, 
                              self.likelihood, 
                              gpu=gpu, 
                              nu=nu,
                              lengthscale_prior=lengthscale_prior,
                              outputscale_prior=outputscale_prior)
        
        # Set noise constraint
        self.model.likelihood.noise_covar.register_constraint(
		        "raw_noise", 
		        GreaterThan(noise_constraint)
		        )
        
        # GPU computation
        if torch.cuda.is_available() and gpu == True:
            self.model = self.model.cuda()
            
    # Maximum likelihood estimation
    def mle(self):
        """Uses maximum likelihood estimation to estimate model hyperparameters.
        
        """ 
        
        loss = optimize_mll(self.model, self.likelihood, self.X, self.y, 
                     learning_rate=self.learning_rate, 
                     n_restarts=self.n_restarts,
                     training_iters=self.training_iters, 
                     noise_prior=self.noise_prior,
                     outputscale_prior=self.outputscale_prior, 
                     lengthscale_prior=self.lengthscale_prior)
        
        self.fit_restart_loss = loss
    
    # Fit model
    def fit(self):
        """Train the gaussian process model.""" 
        
        if self.inference_type == 'MLE':
            self.mle()
        else:
            print('Please specify valid inference type.')
            sys.exit(0)    
        
    # Mean of predictive posterior
    def predict(self, points):
        """Mean of gaussian process posterior predictive distribution.
        
        Parameters
        ----------
        points : torch.tensor
            Domain points to be evaluated.
        
        Returns
        ----------
        numpy.array
            Predicted response values for points.
        """ 
        
        # Get into evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Make predictions
        points = to_torch(points, gpu=self.gpu)
        pred = self.model(points).mean.detach()
            
        if torch.cuda.is_available() and self.gpu == True:
            pred = pred.gpu()
        
        return pred.numpy()
    
    # GP prediction variance
    def variance(self, points):
        """Variance of gaussian process posterior predictive distribution.
        
        Parameters
        ----------
        points : torch.tensor
            Domain points to be evaluated.
        
        Returns
        ----------
        numpy.array 
            Model variance a points.
        """
        
        # Get into evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Compuate variance
        points = to_torch(points, gpu=self.gpu)
        var = self.model(points).variance.detach()
        
        if torch.cuda.is_available() and self.gpu == True:
            var = var.gpu()
        
        return var.numpy()
    
    # Sample posterior
    def sample_posterior(self, points, batch_size=1):
        """Sample functions from gaussian process posterior predictive distribution.
        
        Parameters
        ----------
        points : torch.tensor
            Domain points to be evaluated.
        batch_size : int
            Number of samples to draw.
        
        Returns
        ----------
        torch.tensor 
            Function values at points for samples.
        """
        
        # Get into evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Sample the posterior
        posterior = self.model(points)
        samples = posterior.sample(torch.Size([batch_size]))
        
        return samples
    
    # Regression results
    def regression(self, return_data=False, export_path=None):
        """Helper method for visualizing the models regression performance.
        
        Generates a predicted vs observed plot using the models training data.
        
        Parameters
        ----------
        return_data : bool
            Return predicted responses.
        export_path : None, str
            Export SVG image of predicted vs observed plot to export_path.
                   
        Returns
        ----------
        matplotlib.pyplot
            Scatter plot with computed RMSE and R^2.
        """
        
        # GPyTorch generates a warning when predicting training data.
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
        
            # Get into evaluation mode
            self.model.eval()
            self.likelihood.eval()
        
            # Make predictions
            pred = self.model(self.X).mean.detach()
            
        if torch.cuda.is_available() and self.gpu == True:
            pred = pred.cpu()        
            obs = self.y.cpu()
        else:
            obs = self.y
            
        return pred_obs(pred, 
                        obs, 
                        return_data=return_data, 
                        export_path=export_path)

# Random Forest Model

class RF_Model:
    """Main random forest regression model used for Bayesian optimization."""
    
    def __init__(self, X, y, n_jobs=-1, random_state=10, n_estimators=500,
				 max_features='auto', max_depth=None, min_samples_leaf=1,
				 min_samples_split=2, **kwargs):
        """        
        Parameters
        ----------
        X : list, numpy.array, pandas.DataFrame
            Domain points to be used for model training.
        y : list, numpy.array, pandas.DataFrame
            Response values to be used for model training.
        n_jobs : int 
            Number of processers to use.
        random_state : int
            Insures identical data returns an identical ensemble of regression 
            trees.
        n_estimators : int
            Number of weak estimators to include in ensemble.
        max_features : 'auto', int
            Maximum number of features to consider per node in model training.
        max_depth : None, int
            Maximum depth of individual trees. 
        min_samples_leaf : int
            Minimum number of samples required at each leaf node in model 
            training.
        min_samples_split : int
            Minimum number of samples to require for a node to split.
        """
        
        # Initialize RF model
        self.model = random_forest(n_jobs=n_jobs, 
                                   random_state=random_state, 
                                   n_estimators=n_estimators, 
                                   max_features=max_features, 
                                   max_depth=max_depth, 
                                   min_samples_leaf=min_samples_leaf,
                                   min_samples_split=min_samples_split)
        
        # Make sure X and y are numpy arrays
        self.X = np.array(X)
        self.y = np.array(y)
        
    # Fit    
    def fit(self):
        """Train the frandom forest model.""" 
        
        self.model.fit(self.X, self.y)
        
    # Predict   
    def predict(self, points):
        """Mean of the random forest ensemble predictions.
        
        Parameters
        ----------
        points : list, numpy.array, pandas.DataFrame
            Domain points to be evaluated.
        
        Returns
        ----------
        numpy.array
            Predicted response values.
        """ 
        
        # Make sure points in a numpy array
        points = np.array(points)
        
        # Make predicitons
        pred = self.model.predict(points)
        
        return pred
        
    # Regression   
    def regression(self, return_data=False, export_path=None):
        """Helper method for visualizing the models regression performance.
               
        Generates a predicted vs observed plot using the models training data.
        
        Parameters
        ----------
        return_data : bool
            Return predicted responses.
        export_path : None, str
            Export SVG image of predicted vs observed plot to export_path.
                   
        Returns
        ----------
        matplotlib.pyplot 
            Scatter plot with computed RMSE and R^2.
        """

        pred = self.predict(self.X)
        obs = self.y        
        return pred_obs(pred, 
                        obs, 
                        return_data=return_data, 
                        export_path=export_path) 
    
    # Pseudo-sample random forest model
    def sample_posterior(self, X, batch_size=1):
        """Sample weak estimators from the trained random forest model.
        
        Parameters
        ----------
        points : numpy.array
            Domain points to be evaluated.
        batch_size : int
            Number of estimators predictions to draw from ensemble.
        
        Returns
        ----------
        torch.tensor
            Weak estimator predictions at points.
        """
        
        n_estimators = self.model.n_estimators
        trees = np.random.choice(range(n_estimators), 
                                size=batch_size, 
                                replace=False)
        
        samples = []
        for tree in trees:
            tree_estimates = self.model.estimators_[tree].predict(X)
            samples.append(tree_estimates)
        
        return to_torch(samples)
    
    # Estimate variance from trees in the forest
    def variance(self, points):
        """Variance of random forest ensemble. 
        
        Model variance is estimated as the vairance in the individual tree 
        predictions.
        
        Parameters
        ----------
        points : numpy.array
            Domain points to be evaluated.
        
        Returns
        ----------
        numpy.array
            Ensemble variance at points.
        """
        
        n_estimators = self.model.n_estimators
        
        samples = []
        for tree in range(n_estimators):
            tree_estimates = self.model.estimators_[tree].predict(points)
            samples.append(tree_estimates)
        
        var = np.var(samples, axis=0)
        
        return var
    
# Bayesian Linear Model

class Bayesian_Linear_Model:
    """Bayesian linear regression object compatible with the BO framework."""
    
    def __init__(self, X, y, **kwargs):
        """
        Parameters
        ----------
        X : list, numpy.array, pandas.DataFrame
            Domain points to be used for model training.
        y : list, numpy.array, pandas.DataFrame
            Response values to be used for model training.
        """
        
        # CV set gamma prior parameters - no GS for now
        self.alphas = np.logspace(-6, 0.5, 7)
        
        # Initialize model
        self.model = ARDRegression(n_iter=50)
        
        # Make sure X and y are numpy arrays
        self.X = np.array(X)
        self.y = np.array(y)
        
    # Fit    
    def fit(self):
        """Train the model using grid search CV.""" 
        
        parameters = [{'alpha_1': self.alphas, 'alpha_2': self.alphas}]
        
        # Set the number of folds
        if len(self.X) < 5:
            n_folds = len(self.X)
        else:
            n_folds = 5
        
        # Run grid search
        if n_folds > 1:
        
            # Select l1 term via grid search
            self.grid_search = GridSearchCV(self.model, 
                                       parameters, 
                                       cv=n_folds, 
                                       refit=True,
                                       n_jobs=-1)
        
            self.grid_search.fit(self.X, self.y)
        
            # Set model to trained model
            self.model = self.grid_search.best_estimator_
        
        # Just fit model
        else:
            self.model.fit(self.X, self.y)
            
    def get_scores(self):
        """Get grid search cross validation results.
        
        
        Returns
        ----------
        (numpy.array, numpy.array)
            Average scores and standard deviation of scores for grid.
        """ 
        
        # Plot results
        scores = self.grid_search.cv_results_['mean_test_score']
        scores_std = self.grid_search.cv_results_['std_test_score']
        
        return scores, scores_std
        
    # Predict   
    def predict(self, points):
        """Model predictions.
        
        Parameters
        ----------
        points : list, numpy.array, pandas.DataFrame
            Domain points to be evaluated.
        
        Returns
        ----------
        numpy.array
            Predicted response values at points.
        """ 
        
        # Make sure points in a numpy array
        points = np.array(points)
        
        # Make predicitons
        pred = self.model.predict(points)
        
        return pred
        
    # Regression   
    def regression(self, return_data=False, export_path=None):
        """Helper method for visualizing the models regression performance.
        
        Generates a predicted vs observed plot using the models training data.
        
        Parameters
        ----------
        return_data : bool
            Return predicted responses.
        export_path : None, str
            Export SVG image of predicted vs observed plot to export_path.
                   
        Returns
        ----------
        matplotlib.pyplot 
            Scatter plot with computed RMSE and R^2.
        """

        pred = self.predict(self.X)
        obs = self.y        
        return pred_obs(pred, 
                        obs, 
                        return_data=return_data, 
                        export_path=export_path) 
    
    # Estimate variance
    def variance(self, points):
        """Estimated variance of Bayesian linear model.
        
        Parameters
        ----------
        points : numpy.array
            Domain points to be evaluated.
        
        Returns
        ----------
        numpy.array
            Model variance at points.
        """
        
        # Make sure points in a numpy array
        points = np.array(points)
        
        # Make predicitons
        pred, std = self.model.predict(points, return_std=True)
        
        return std**2

# Random sample
        
class Random:
    """Dummy class for random sampling. 
    
    Use with init_seed for benchmarking Bayesian optimization versus random 
    sampling. Class defined such that it can be called by the BO class in
    simulations. 
    
    Note
    ----
    Use Random with random acquisition function.
    """
    
    def __init__(self, X, y, **kwargs):
        None
    
    def fit(self):
        return None
    
    def predict(self, points):
        return None
        
# Score model performance
        
def score(trained_model, X, y):
    """Compute RMSE and R^2 for a trained model.
    
    Parameters
    ----------
    trainined_model : bro.models 
        Trained model.
    X : numpy.array, torch.tensor
        Domain points to be evaluated.
    y : numpy.array, torch.tensor
        Response values corresponding to X.
    
    Returns
    ----------
    (int, int)
        RMSE and R^2 values.
    """
    
    pred = np.array(trained_model.predict(X))
    obs = np.array(y)
    RMSE, R2 = model_performance(pred, obs)
    
    return RMSE, R2

# Compute cross-validation scores
        
def cross_validate(base_model, X, y, kfold=5, random_state=None, **kwargs):
    """Compute cross-validation scores for models.
    
    Parameters
    ----------
    base_model : bro.models
        Uninitialized model object.
    X : numpy.array, torch.tensor
        Domain points to be evaluated.
    y : numpy.array, torch.tensor
        Response values corresponding to domain points X.
    kfold : int
        Number of splits used in cross-validation.
    
    Returns
    ----------
    list
        Mean training and validation scores [train_RMSE, validation_RMSE,
        train_R^2, validation_R^2].
    """
    
    # CV Split  
    split = cv_split(X, n_splits=kfold, random_state=random_state)
    
    train = []
    validation = []
    
    for train_index, test_index in split:
        
        # Fit to training
        model_copy = base_model
        model = model_copy(X[train_index], 
                           y[train_index],
                           kwargs)
        model.fit()
        
        # Training performance
        fit = np.array(model.predict(X[train_index]))
        act = np.array(y[train_index])
        rmse_train, r2_train = model_performance(fit, act)
        train.append([rmse_train, r2_train])
        
        # Validation performance
        pred = np.array(model.predict(X[test_index]))
        obs = np.array(y[test_index])
        rmse_val, r2_val = model_performance(pred, obs)
        validation.append([rmse_val, r2_val])
        
    scores = [np.array(train)[:,0].mean(),
              np.array(validation)[:,0].mean(),
              np.array(train)[:,1].mean(),
              np.array(validation)[:,1].mean()]
    
    return scores
    
    
    
    
    
    
    
    