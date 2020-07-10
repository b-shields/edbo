# -*- coding: utf-8 -*-

# Imports

import pandas as pd
import numpy as np

import dill

from gpytorch.priors import GammaPrior

from .models import GP_Model
from .base_models import fast_computation
from .init_scheme import Init
from .objective import objective
from .acq_func import acquisition
from .plot_utils import plot_convergence
from .pd_utils import to_torch, load_csv_or_excel
from .chem_utils import ChemDraw
from .feature_utils import reaction_space
from .utils import bot

# Main class definition

class BO:
    """Main method for calling Bayesian optimization algorithm.
    
    Class provides a unified framework for selecting experimental 
    conditions for the parallel optimization of chemical reactions
    and for the simulation of known objectives. The algorithm is 
    implemented on a user defined grid of domain points and is
    flexible to any numerical encoding.
    """
        
    def __init__(self,                 
                 results_path=None, results=pd.DataFrame(),
                 domain_path=None, domain=pd.DataFrame(),
                 exindex_path=None, exindex=pd.DataFrame(),
                 model=GP_Model, acquisition_function='EI', init_method='rand', 
                 target=-1, batch_size=5, duplicate_experiments=False, 
                 gpu=False, fast_comp=False, noise_constraint=1e-5,
                 matern_nu=2.5, lengthscale_prior=[GammaPrior(2.0, 0.2), 5.0],
                 outputscale_prior=[GammaPrior(5.0, 0.5), 8.0],
                 noise_prior=[GammaPrior(1.5, 0.5), 1.0],
                 computational_objective=None
                 ):
        
        """
        Experimental results, experimental domain, and experiment index of 
        known results can be passed as paths to .csv or .xlsx files or as 
        DataFrames.
        
        Parameters
        ----------
        results_path : str, optional
            Path to experimental results.
        results : pandas.DataFrame, optional
            Experimental results with X values matching the domain.
        domain_path : str, optional
            Path to experimental domain.
            
            Note
            ----
            A domain_path or domain are required.
            
        domain : pandas.DataFrame, optional
            Experimental domain specified as a matrix of possible configurations.
        exindex_path : str, optional
            Path to experiment results index if available.
        exindex : pandas.DataFrame, optional
            Experiment results index matching domain format. Used as lookup 
            table for simulations.
        model : edbo.models 
            Surrogate model object used for Bayesian optimization. 
            See edbo.models for predefined models and specification of custom
            models.
        acquisition_function : str 
            Acquisition function used for for selecting a batch of domain 
            points to evaluate. Options: (TS) Thompson Sampling, ('EI') 
            Expected Improvement, (PI) Probability of Improvement, (UCB) 
            Upper Confidence Bound, (EI-TS) EI (first choice) + TS (n-1 choices), 
            (PI-TS) PI (first choice) + TS (n-1 choices), (UCB-TS) UCB (first 
            choice) + TS (n-1 choices), (MeanMax-TS) Mean maximization 
            (first choice) + TS (n-1 choices), (VarMax-TS) Variance 
            maximization (first choice) + TS (n-1 choices), (MeanMax) 
            Top predicted values, (VarMax) Variance maximization, (rand) 
            Random selection.
        init_method : str 
            Strategy for selecting initial points for evaluation. 
            Options: (rand) Random selection, (pam) k-medoids algorithm, 
            (kmeans) k-means algorithm, (external) User define external data
            read in as results.
        target : str
            Column label of optimization objective. If set to -1, the last 
            column of the DataFrame will be set as the target.
        batch_size : int
            Number of experiments selected via acquisition and initialization 
            functions.
        duplicate_experiments : bool 
            Allow the acquisition function to select experiments already 
            present in results. 
        gpu : bool
            Carry out GPyTorch computations on a GPU if available.
        fast_comp : bool 
            Enable fast computation features for GPyTorch models.
        noise_constraint : float
            Noise constraint for GPyTorch models.
        matern_nu : 0.5, 1.5, 2.5
            Parameter value for model Matern kernel.
        lengthscale_prior : [gytorch.prior, initial_value]
            Specify a prior over GP length scale prameters.
        outputscale_prior : [gytorch.prior, initial_value]
            Specify a prior over GP output scale prameter.
        noise_prior : [gytorch.prior, initial_value]
            Specify a prior over GP noice prameter.
        computational_objective : function, optional
            Function to be optimized for computational objectives.
            
        """
        
        # Fast computation
        self.fast_comp = fast_comp
        fast_computation(fast_comp)
        
        # Initialize data container
        self.obj = objective(results_path=results_path, 
                             results=results, 
                             domain_path=domain_path, 
                             domain=domain, 
                             exindex_path=exindex_path, 
                             exindex=exindex, 
                             target=target, 
                             gpu=gpu,
                             computational_objective=computational_objective)
        
        # Initialize acquisition function
        self.acq = acquisition(acquisition_function, 
                              batch_size=batch_size, 
                              duplicates=duplicate_experiments)
        
        # Initialize experiment init sequence
        self.init_seq = Init(init_method, batch_size)
        
        # Initialize other stuff
        self.base_model = model # before eval for retraining
        self.model = model      # slot for after eval
        self.batch_size = batch_size
        self.duplicate_experiments = duplicate_experiments
        self.gpu = gpu
        self.proposed_experiments = pd.DataFrame()
        self.nu = matern_nu
        self.noise_constraint = noise_constraint
        self.lengthscale_prior = lengthscale_prior
        self.outputscale_prior = outputscale_prior
        self.noise_prior = noise_prior
        
    # Initial samples using init sequence
    def init_sample(self, seed=None, append=False, export_path=None,
                    visualize=False):
        """Generate initial samples via an initialization method.
        
        Parameters
        ----------
        seed : None, int
            Random seed used for selecting initial points.
        append : bool
            Append points to results if computational objective or experiment
            index are available.
        export_path : str 
            Path to export SVG of clustering results if pam or kmeans methods 
            are used for selecting initial points.
        visualize : bool
            If initialization method is set to 'pam' or 'kmeans' and visualize
            is set to True then a 2D embedding of the clustering results will
            be generated.
        
        Returns
        ----------
        pandas.DataFrame
            Domain points for proposed experiments.
        """
        
        # Run initialization sequence
        if self.init_seq.method != 'external':
            self.obj.clear_results()
        self.proposed_experiments = self.init_seq.run(self.obj, 
                                                      seed=seed, 
                                                      export_path=export_path,
                                                      visualize=visualize)
        
        # Append to know results
        if append == True and self.init_seq.method != 'external':
            self.obj.get_results(self.proposed_experiments, append=append)

        return self.proposed_experiments
        
    # Run algorithm and get next round of experiments
    def run(self, append=False, n_restarts=0, learning_rate=0.1,
            training_iters=100):
        """Run a single iteration of optimization with known results.
        
        Note
        ----
        Use run for human-in-the-loop optimization.
        
        Parameters
        ----------
        append : bool
            Append points to results if computational objective or experiment
            index are available.
        n_restarts : int
            Number of restarts used when optimizing GPyTorch model parameters.
        learning_rate : float
            ADAM learning rate used when optimizing GPyTorch model parameters.
        training_iters : int
            Number of iterations to run ADAM when optimizin GPyTorch models
            parameters.
        
        Returns
        ----------
        pandas.DataFrame
            Domain points for proposed experiments.
        """        
        
        # Initialize and train model
        self.model = self.base_model(self.obj.X, 
                                     self.obj.y, 
                                     gpu=self.gpu,
                                     nu=self.nu,
                                     noise_constraint=self.noise_constraint,
                                     lengthscale_prior=self.lengthscale_prior,
                                     outputscale_prior=self.outputscale_prior,
                                     noise_prior=self.noise_prior,
                                     n_restarts=n_restarts,
                                     learning_rate=learning_rate,
                                     training_iters=training_iters
                                     )
        
        self.model.fit()
        
        # Select candidate experiments via acquisition function
        self.proposed_experiments = self.acq.evaluate(self.model, self.obj)
        
        # Append to know results
        if append == True:
            self.obj.get_results(self.proposed_experiments, append=append)
        
        return self.proposed_experiments
        
    # Simulation using known objectives
    def simulate(self, iterations=1, seed=None, update_priors=False, 
                 n_restarts=0, learning_rate=0.1, training_iters=100):
        """Run autonomous BO loop.
        
        Run N iterations of optimization with initial results obtained 
        via initialization method and experiments selected from 
        experiment index via the acquisition function. Simulations 
        require know objectives via an index of results or function.
        
        Note
        ----
        Requires a computational objective or experiment index.
        
        Parameters
        ----------
        append : bool
            Append points to results if computational objective or experiment
            index are available.
        n_restarts : int
            Number of restarts used when optimizing GPyTorch model parameters.
        learning_rate : float
            ADAM learning rate used when optimizing GPyTorch model parameters.
        training_iters : int
            Number of iterations to run ADAM when optimizin GPyTorch models
            parameters.
        seed : None, int
            Random seed used for initialization.
        update_priors : bool 
            Use parameter estimates from optimization step N-1 as initial 
            values for step N.

        """            
        
        # Initialization data
        self.init_sample(seed=seed, append=True)
        
        # Simulation
        for i in range(iterations):
                
            # Use pamater estimates from previous step as initial values
            if update_priors == True and i > 0 and 'GP' in str(self.base_model):
                post_ls = self.model.model.covar_module.base_kernel.lengthscale.detach()[0]
                post_os = self.model.model.covar_module.outputscale.detach()
                post_n = self.model.model.likelihood.noise.detach()[0]
                
                if self.lengthscale_prior == None:
                    self.lengthscale_prior = [None, post_ls]
                else:
                    self.lengthscale_prior[1] = post_ls
                
                if self.outputscale_prior == None:
                    self.outputscale_prior = [None, post_os]
                else:
                    self.outputscale_prior[1] = post_os
                    
                if self.noise_prior == None:
                    self.noise_prior = [None, post_n]
                else:
                    self.noise_prior[1] = post_n
            
            # Initialize and train model
            self.model = self.base_model(self.obj.X, 
                                     self.obj.y, 
                                     gpu=self.gpu,
                                     nu=self.nu,
                                     noise_constraint=self.noise_constraint,
                                     lengthscale_prior=self.lengthscale_prior,
                                     outputscale_prior=self.outputscale_prior,
                                     noise_prior=self.noise_prior,
                                     n_restarts=n_restarts,
                                     learning_rate=learning_rate,
                                     training_iters=training_iters
                                     )
            
            self.model.fit()
        
            # Select candidate experiments via acquisition function
            self.proposed_experiments = self.acq.evaluate(self.model, self.obj)
            
            # Append results to known data
            self.obj.get_results(self.proposed_experiments, append=True)
    
    # Clear results between simulations
    def clear_results(self):
        """Clear results manually. 
        
        Note
        ----
        'rand' and 'pam' initialization methods clear results automatically.
        
        """  
        
        self.obj.clear_results()
    
    # Plot convergence
    def plot_convergence(self, export_path=None):
        """Plot optimizer convergence.
        
        Parameters
        ----------
        export_path : None, str 
            Path to export SVG of optimizer optimizer convergence plot.
        
        Returns
        ----------
        matplotlib.pyplot 
            Plot of optimizer convergence.
        """ 
        
        plot_convergence(
                self.obj.results_input()[self.obj.target],
                self.batch_size,
                export_path=export_path)
    
    # Acquisition summary
    def acquisition_summary(self):
        """Summarize predicted mean and variance for porposed points.
        
        Returns
        ----------
        pandas.DataFrame
            Summary table.
        """
        
        proposed_experiments = self.proposed_experiments.copy()
        X = to_torch(proposed_experiments, gpu=self.gpu)
        
        # Compute mean and variance, then unstandardize
        mean = self.obj.scaler.unstandardize(self.model.predict(X))
        var = (np.sqrt(self.model.variance(X)) * self.obj.scaler.std)**2
        
        # Append to dataframe
        for col, name in zip([mean, var], ['predicted ' + self.obj.target, 'variance']):
            proposed_experiments[name] = col
        
        return proposed_experiments
        
    # Best observed result
    def best(self):
        """Best observed objective values and corresponding domain point."""
        
        sort = self.obj.results_input().sort_values(self.obj.target, ascending=False)
        return sort.head()
    
    # Save BO instance
    def save(self, path='BO.pkl'):
        """Save BO state.
        
        Parameters
        ----------
        path : str 
            Path to export <BO state dict>.pkl.
        
        Returns
        ----------
        None
        """ 
        
        file = open(path, 'wb')
        dill.dump(self.__dict__, file)
        file.close()
    
    # Load BO instance
    def load(self, path='BO.pkl'):
        """Load BO state.
        
        Parameters
        ----------
        path : str 
            Path to <BO state dict>.pkl.
        
        Returns
        ----------
        None
        """ 
        
        file = open(path, 'rb')
        tmp_dict = dill.load(file)
        file.close()          

        self.__dict__.update(tmp_dict) 
        
        
class BO_express(BO):
    """Quick method for auto-generating a reaction space, encoding, and BO.
    
    Class provides a unified framework for defining reaction spaces, encoding 
    reacitons, selecting experimental conditions for the parallel optimization 
    of chemical reactions, and analyzing results.
    
    BO_express automates most of the process required for BO such as the 
    featurization of the reaction space, preprocessing of data and selection of 
    gaussian process priors.
    
    Reaction components and encodings are passed to BO_express using 
    dictionaries. BO_express attempts to encode each component based on the 
    specified encoding. If there is an error in a SMILES string or the name
    could not be found in the NIH database an edbo bot is spawned to help
    resolve the issue. Once instantiated, BO_express.help() will also spawn
    an edbo bot to help with tasks.
    
    Example
    -------
    Defining a reaction space ::
    
        from edbo.bro import BO_express
            
        # (1) Define a dictionary of components
        reaction_components={
            'aryl_halide':['chlorobenzene','iodobenzene','bromobenzene'],
            'base':['DBU', 'MTBD', 'potassium carbonate', 'potassium phosphate'],
            'solvent':['THF', 'Toluene', 'DMSO', 'DMAc'],
            'ligand': ['c1ccc(cc1)P(c2ccccc2)c3ccccc3', # PPh3
                       'C1CCC(CC1)P(C2CCCCC2)C3CCCCC3', # PCy3
                       'CC(C)c1cc(C(C)C)c(c(c1)C(C)C)c2ccccc2P(C3CCCCC3)C4CCCCC4' # X-Phos
                       ],
            'concentration':[0.1, 0.2, 0.3],
            'temperature': [20, 30, 40],
            'additive': '<defined in descriptor_matrices>'}
        
        # (2) Define a dictionary of desired encodings
        encoding={'aryl_halide':'resolve',
                  'base':'ohe',
                  'solvent':'resolve',
                  'ligand':'smiles',
                  'concentration':'numeric',
                  'temperature':'numeric'}
        
        # (3) Add any user define descriptor matrices directly
        import pandas as pd
        
        A = pd.DataFrame(
                 [['a1', 1,2,3,4],['a2',1,5,2,0],['a3', 3,5,1,25]],
                 columns=['additive', 'A_des1', 'A_des2', 'A_des3', 'A_des4'])
        
        descriptor_matrices = {'additive': A}
        
        # (4) Instatiate BO_express
        bo = BO_express(reaction_components=reaction_components, 
                        encoding=encoding,
                        descriptor_matrices=descriptor_matrices,
                        batch_size=10,
                        acquisition_function='TS',
                        target='yield')
    
    """
    
    def __init__(self,
                 reaction_components={}, encoding={}, descriptor_matrices={},
                 model=GP_Model, acquisition_function='EI', init_method='rand', 
                 target=-1, batch_size=5, computational_objective=None):
        """        
        Parameters
        ----------
        reaction_components : dict
            Dictionary of reaction components of the form: 
                
            Example
            -------
            Defining reaction components ::
                
                {'A': [a1, a2, a3, ...],
                 'B': [b1, b2, b3, ...],
                 'C': [c1, c2, c3, ...],
                             .
                 'N': [n1, n2, n3, ...]}
            
            Components can be specified as: (1) arbitrary names, (2) chemical 
            names or nicknames, (3) SMILES strings, or (4) numeric values.
            
            Note
            ----
            A reaction component will not be encoded unless its key is present
            in the reaction_components dictionary.
            
        encodings : dict
            Dictionary of encodings with keys corresponding to reaction_components.
            Encoding dictionary has the form: 
                
            Example
            -------
            Defining reaction encodings ::
                
                {'A': 'resolve',
                 'B': 'ohe',
                 'C': 'smiles',
                        .
                 'N': 'numeric'}
            
            Encodings can be specified as: ('resolve') resolve a compound name 
            using the NIH database and compute Mordred descriptors, ('ohe') 
            one-hot-encode, ('smiles') compute Mordred descriptors using a smiles 
            string, ('numeric') numerical reaction parameters are used as passed.
            If no encoding is specified, the space will be automatically 
            one-hot-encoded.
        descriptor_matrices : dict
            Dictionary of descriptor matrices where keys correspond to 
            reaction_components and values are pandas.DataFrames.
            
            Descriptor dictionary has the form: 
                
            Example
            -------
            User defined descriptor matrices ::
                
                # DataFrame where the first column is the identifier (e.g., a SMILES string)
                
                A = pd.DataFrame([....], columns=[...])
                
                --------------------------------------------
                  A_SMILES  |  des1  |  des2  | des3 | ...
                --------------------------------------------
                      .         .        .       .     ...
                      .         .        .       .     ...
                --------------------------------------------
                
                # Dictionary of descriptor matrices defined as DataFrames
                
                descriptor_matrices = {'A': A}
            
            Note
            ----
            If a key is present in both encoding and descriptor_matrices then 
            the descriptor matrix will take precedence.
            
        model : edbo.models
            Surrogate model object used for Bayesian optimization. 
            See edbo.models for predefined models and specification of custom
            models.
        acquisition_function : str 
            Acquisition function used for for selecting a batch of domain 
            points to evaluate. Options: (TS) Thompson Sampling, ('EI') 
            Expected Improvement, (PI) Probability of Improvement, (UCB) 
            Upper Confidence Bound, (EI-TS) EI (first choice) + TS (n-1 choices), 
            (PI-TS) PI (first choice) + TS (n-1 choices), (UCB-TS) UCB (first 
            choice) + TS (n-1 choices), (MeanMax-TS) Mean maximization 
            (first choice) + TS (n-1 choices), (VarMax-TS) Variance 
            maximization (first choice) + TS (n-1 choices), (MeanMax) 
            Top predicted values, (VarMax) Variance maximization, (rand) 
            Random selection.
        init_method : str 
            Strategy for selecting initial points for evaluation. 
            Options: (rand) Random selection, (pam) k-medoids algorithm, 
            (kmeans) k-means algorithm, (external) User define external data
            read in as results.
        target : str
            Column label of optimization objective. If set to -1, the last 
            column of the DataFrame will be set as the target.
        batch_size : int
            Number of experiments selected via acquisition and initialization 
            functions.
        computational_objective : function, optional
            Function to be optimized for computational objectives.
        """
        
        # Initialize edbo_bot
        self.edbo_bot = bot()
        self.edbo_bot.talk('For help try BO_express.help() or see the documentation page.')
        
        # Check the input
        if len(reaction_components) > 0:
            N = 1
            for key in reaction_components:
                N *= len(reaction_components[key])
            
            self.edbo_bot.talk('Building reaction space...')

        # Build the search space (clean, decorrelate, standardize)
        self.reaction = reaction_space(reaction_components, 
                                       encoding=encoding,
                                       descriptor_matrices=descriptor_matrices)
        
        # Determine appropriate priors
        mordred = False
        for header in self.reaction.index_headers:
            if 'SMILES' in header:
                mordred = True
                break
        if mordred:
            if len(self.reaction.data.columns.values) < 50:
                mordred = False
        
        # low D priors
        if len(self.reaction.data.columns.values) < 5:
            lengthscale_prior = [GammaPrior(1.3, 0.5), 0.5]
            outputscale_prior = [GammaPrior(5.0, 0.2), 20.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]
        # DFT optimized priors or LS and OS
        elif mordred and len(self.reaction.data.columns.values) < 100:
            lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
            outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]  
        # Mordred optimized priors
        elif mordred:
            lengthscale_prior = [GammaPrior(2.0, 0.1), 10.0]
            outputscale_prior = [GammaPrior(2.0, 0.1), 10.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]
        # OHE optimized priors
        else:
            lengthscale_prior = [GammaPrior(3.0, 1.0), 2.0]
            outputscale_prior = [GammaPrior(5.0, 0.2), 20.0]
            noise_prior = [GammaPrior(1.5, 0.1), 5.0]
        
        super(BO_express, self).__init__(domain=self.reaction.data,
                                         model=model, 
                                         acquisition_function=acquisition_function,
                                         init_method=init_method,
                                         target=target,
                                         batch_size=batch_size,
                                         computational_objective=computational_objective,
                                         lengthscale_prior=lengthscale_prior,
                                         outputscale_prior=outputscale_prior,
                                         noise_prior=noise_prior,
                                         fast_comp=True)
        
        
    def get_experiments(self, structures=False):
        """Return indexed experiments proposed by Bayesian optimization algorithm.
        
        edbo.BO works directly with a standardized encoded reaction space. This 
        method returns proposed experiments as the origional smiles strings, 
        categories, or numerical values.
        
        Parameters
        ----------
        structures : bool
            If True, use RDKit to print out the chemical structures of any
            encoded smiles strings.
        
        Returns
        ----------
        pandas.DataFrame
            Proposed experiments.
        """
        
        # Index entries
        experiments = self.reaction.get_experiments(self.proposed_experiments.index.values)
        
        # SMILES columns
        smiles_cols = []
        for col in experiments.columns.values:
            if 'SMILES' in col:
                smiles_cols.append(col)
        
        if structures:
            for experiment in experiments[smiles_cols].values:
                cdx = ChemDraw(experiment)
                cdx.show()
                
        return experiments
    
    def add_results(self, results_path=None):
        """Add experimental results.
        
        Experimental results should be added with the same column headings as
        those returned by BO_express.get_experiments. If a path to the results
        is not specified, an edbo bot is spawned to help load results. It does
        so by exporting the entire reaction space to a CSV file in the working
        directory.
        
        Note: The first column in the CSV/EXCEL results file must have the same
        index as the experiment. Try BO_express.export_proposed() to export a
        CSV file with the proper format.
        
        Parameters
        ----------
        results_path : str
            Imports results from a CSV/EXCEL file with system path results_path.
        
        Returns
        ----------
        None
        """
        
        if results_path != None:
            results = load_csv_or_excel(results_path, index_col=0).dropna(axis=0)
        
        else:
            self.edbo_bot.talk('No path to <results>.csv was specified.')
            self.edbo_bot.talk('Exporting experiment domain to CSV file...')
            self.reaction.base_data[self.reaction.index_headers].to_csv('results.csv', 
                                                                        index=True)
            
            self.edbo_bot.talk('Include your results column at the right and save the file.')
            self.edbo_bot.get_response('Let me know when you are done...')
            results = pd.read_csv('results.csv', index_col=0).dropna(axis=0)
        
        result_descriptors = self.obj.domain.iloc[results.index.values]
        results = pd.concat([result_descriptors, results.iloc[:,[-1]]], axis=1)
        
        # Initialize data container
        self.obj = objective(domain=self.obj.domain,
                             results=results, 
                             exindex=self.obj.exindex,
                             gpu=self.obj.gpu,
                             computational_objective=self.obj.computational_objective)
        
    def export_proposed(self, path=None):
        """Export proposed experiments.
        
        edbo.BO works directly with a standardized encoded reaction space. This 
        method exports proposed experiments as the origional smiles strings, 
        categories, or numerical values. If a path to the results is not 
        specified, a CSV file entitled 'experiments.csv' will be exported to 
        the current working directory.
        
        Parameters
        ----------
        path : str
            Export a CSV file to path.
        
        Returns
        ----------
        None
        """
            
        index = self.proposed_experiments.index.values
        proposed = self.reaction.base_data[self.reaction.index_headers].iloc[index]
        target = pd.DataFrame([['<Enter Response>']] * len(proposed), 
                              columns=[self.obj.target],
                              index=index)
        proposed = pd.concat([proposed, target], axis=1)
            
        if path == None:
            proposed.to_csv('experiments.csv')
                
        else:
            proposed.to_csv(path)
    
    def help(self):
        """Spawn an edbo bot to help with tasks.
        
        If you are not familiar with edbo commands BO_express.help() will spawn
        an edbo bot to help with tasks. Natural language can be used to interact 
        with edbo bot in the terminal to accomplish tasks such as: initializing 
        (selecting initial experiments using chosen init method), optimizing 
        (run BO algorithm with availible data to choose next experiments), getting 
        proposed experiments, adding experimental results, checking the underlying
        models regression performance, saving the BO instance so you can load it 
        for use later, and exporting proposed experiments to a CSV file.

        """
        
        # Keywords which trigger responses
        trigger_dict = {'exit':['exit', 'stop'],
                'initialize':['init', 'start'],
                'optimize':['opt', 'run', 'bo', 'next'],
                'print proposed':['print', 'next', 'choic', 'choice', 'exper'],
                'add results':['load', 'add', 'results', 'data'],
                'check model':['regres', 'fit', 'pred', 'model'],
                'pickle BO object for later':['save', 'pickle'],
                'export proposed':['expo', 'save', 'exper']
                }
        
        # Response functions
        def bot_exit():
            return 'exit'
        
        def bot_init():
            self.init_sample()
            print(self.get_experiments())
        
        def bot_opt():
            if len(self.obj.results) > 0:
                self.run()
                print(self.get_experiments())
            else:
                self.edbo_bot.talk('No experimental data are loaded.')
        
        def bot_next():
            self.get_experiments()
            print(self.get_experiments())
        
        def bot_data():
            self.add_results()
        
        def bot_model():
            self.model.regression()
        
        def bot_save():
            self.save()
        
        def bot_export():
            self.export_proposed()
        
        response_dict = {'exit':bot_exit,
                'initialize':bot_init,
                'optimize':bot_opt,
                'print proposed':bot_next,
                'add results':bot_data,
                'check model':bot_model,
                'pickle BO object for later':bot_save,
                'export proposed':bot_export
                }
        
        # Messages related to different triggers
        print_dict = {'exit':'Exiting...',
                'initialize':'Initializing via ' + self.init_seq.method + ' method...\n',
                'optimize': 'Fitting model... Optimizing acquisition function...\n',
                'print proposed':'The next proposed experiments are:\n',
                'check model':'This is how ' + str(self.base_model) + ' fits the available data:\n',
                'pickle BO object for later':'Saving edbo.BO instance...\n',
                'export proposed':'Exporting proposed experiments to CSV file...\n'
                }
        
        # Responses which require confirmaiton
        confirm_dict = {
                'initialize':'Choose initial experiments via ' + self.init_seq.method + ' method? (yes or no)',
                'optimize':'Run Bayesian optimization with the avialable data? (yes or no)',
                'add results':'Update experimental results from a new CSV file? (yes or no)',
                'pickle BO object for later':'Save instace? (yes or no) You can load instance later with edbo.BO_express.load().'
                }
        
        # Help loop
        control = 'run'
        while control != 'exit':
            control = self.edbo_bot.resolve_direct('What can I help you with?',
                                         trigger_dict,
                                         response_dict,
                                         print_dict,
                                         confirm_dict)
        
    