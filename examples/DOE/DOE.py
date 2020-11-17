# -*- coding: utf-8 -*-
"""
###################################################################
Design of experiments module for initializing EDBO
###################################################################

Methods are designed to work with the edbo.utils.Data objects. 
These objects are created when for example edbo.BO_express is
used to auto-generate an encoded reaction space. The modified
Data object can be accessed via edbo.BO_express.reaction.
"""

# Imports

from pyDOE2 import ccdesign, bbdesign, lhs, pbdesign, gsd
from edbo.math_utils import pca
from edbo.utils import Data
import pandas as pd
import numpy as np

# Get the submatrices associated with different components

def component_submatrix(rxn, components):
    """
    Get the submatrices associated with different components
    from the reaction data container.
    """
    
    # Get descriptors associated with each submatrix
    columns = {}
    for key in components:
        component_descriptors = []
        for col in rxn.data.columns.values:
            if key in col:
                component_descriptors.append(col)
        columns[key] = component_descriptors
        
    # Obtain submatrices
    descriptors = {}
    for key in columns:
        descriptors[key] = rxn.data.copy()[columns[key]]
        
        index = ''
        for name in rxn.index_headers:
            if key in name:
                index = name       
        descriptors[key].insert(0, index, rxn.base_data[index])
    
    return descriptors

# Principal components anlaysi 

def principles(descriptors, scale='minmax'):
    """
    Get principle components associated with each reaction
    component. Use the first PC for DOE.
    """
    
    coordinates = {}
    for key in descriptors:
        
        unique = descriptors[key].drop_duplicates()
        if len(descriptors[key].columns.values) > 2:
            col = pca(unique.iloc[:,1:], n_components=1)
        else:
            col = unique.copy().iloc[:,[1]]
        
        # Scale
        if scale == 'minmax':
            col = Data(col)
            col.standardize(scaler='minmax', target=None)
            col = col.data
        
        col.insert(0, unique.columns.values[0], unique.iloc[:,0].values)
        
        coordinates[key] = col
    
    return coordinates

# Initial design for BO uisng PCA-DOE approach

class init_design:
    """
    Use DOE over the PCs of the reaction components to select initial
    experiments. This is an attempt to deal with the ordering of
    multi-level categorical variables.
    """
    
    def __init__(self, reaction, components):
        
        self.reaction = reaction
        
        # Get component submatrices
        self.subs = component_submatrix(reaction, components)
        
        # Scaled principle components
        self.pcs = principles(self.subs)
        
        # Count component levels
        self.levels = [len(self.pcs[key]) for key in self.pcs]
        
        # Design information
        self.N = len(self.levels)
        self.names = [key for key in self.pcs]
    
    def get_closest(self, pc, value):
        """
        Fill a numerical design by getting reaction component
        which corresponda to the closest PC value.
        """
        
        diff = (self.pcs[pc].iloc[:,1].copy() - value).abs().sort_values()
        closest = self.pcs[pc].iloc[diff.index.values[0]]
        
        return closest.values
    
    def get_closest_principles(self):
        """
        Loop get_closest over all principal components.
        """
        
        experiments = []
        for i in range(len(self.design)):
            experiment = []
            for col in self.design.columns.values:
                closest = self.get_closest(col, self.design[col].iloc[i])[0]
                experiment.append(closest)
            experiments.append(experiment)
        
        self.experiment_design = pd.DataFrame(experiments, 
                                              columns=self.design.columns.values)
        
    def encoded(self):
        """
        Get encoded domain points corresponding to an experimental
        design.
        """
        
        index = self.reaction.base_data[self.reaction.index_headers]

        indices = []
        for experiment in self.experiment_design.values:
            entry = index[(index.values == experiment).all(1)]
            indices.append(entry.index.values[0])

        self.encoded_design = self.reaction.data.iloc[indices]
    
    def lhs(self, samples, seed=None):
        """
        Get experiments corresponding to a latin hypercube design.
        """
        
        lh = lhs(self.N, 
                 samples=samples, 
                 criterion='center', 
                 random_state=seed)
        
        self.design = pd.DataFrame(lh, columns=self.names)
        
    def pbd(self):
        """
        Get experiments corresponding to a Plackett-Burman design.
        """
        
        pb = pd.DataFrame(pbdesign(len(self.levels)), columns=self.names)
        pb = Data(pb)
        pb.standardize(scaler='minmax', target=None)
        
        self.design = pb.data
    
    def ccd_lhs(self, ccd_factors, lhs_factors, add_samples=5, seed=None,
                center_fill=False):
        """
        Get experiments corresponding to a hybrid central composit
        and latin hypercube design.
        """
        
        # Run cc design
        cc = pd.DataFrame(ccdesign(len(ccd_factors), (0,1), 'o', 'cci'),
                          columns=ccd_factors)
        cc = Data(cc)
        cc.standardize(target=None, scaler='minmax')
        
        # Either fill lh design or a single number
        if center_fill:
            lh = pd.DataFrame(np.ones((len(cc.data), len(lhs_factors))) * 0.5,
                              columns=lhs_factors)
        else:        
            lh = pd.DataFrame(lhs(len(lhs_factors),
                              samples=len(cc.data),
                              criterion='center',
                              random_state=seed),
                              columns=lhs_factors)
        
        lh2 = pd.DataFrame(lhs(self.N, 
                               samples=add_samples, 
                               criterion='center', 
                               random_state=seed),
                            columns=ccd_factors + lhs_factors)
        
        # Concatenate
        design = pd.concat([cc.data, lh], axis=1)
        design = pd.concat([design, lh2], axis=0).reset_index(drop=True)
        
        # Reorder
        self.design = design.copy()[self.names]
        
    def visualize(self):
        """
        Visualize the selected experiments.
        """
        
        for i in self.encoded_design.index.values:
            self.reaction.visualize(i)
        
# Initial design for BO uisng combinatorial design

class generalized_subset_design:
    """
    Get experiments corresponding to a generalized subset design.
    """
    
    def __init__(self, reaction, components):
        
        self.reaction = reaction
        
        # Get component submatrices
        self.subs = component_submatrix(reaction, components)
        
        # Get unique sets
        self.unique = {}
        for key in self.subs:
            self.unique[key] = self.subs[key].drop_duplicates().reset_index(drop=True).iloc[:,0]
        
        # Count component levels
        self.levels = [len(self.unique[key]) for key in self.unique]
        
        # Design information
        self.N = len(self.levels)
        self.names = [key for key in self.unique]
        
    def build(self, reduction=20):
        """
        Generate the design.
        """
        
        gs = gsd(self.levels, reduction)
        self.design = pd.DataFrame(gs, columns=self.names)
    
    def get_experiments(self):
        """
        Get experiments corresponding to the design.
        """
            
        # Fill in experiments
        experiments = []
        for i in range(len(self.design)):
            experiment = []
            row = self.design.iloc[i]
            for col in self.design.columns.values:
                experiment.append(self.unique[col].iloc[row[col]])
            experiments.append(experiment)
        
        self.experiment_design = pd.DataFrame(experiments, 
                                              columns=self.design.columns.values)
        
    def encoded(self):
        """
        Get encoded experiments corresponding to the design.
        """
        
        index = self.reaction.base_data[self.reaction.index_headers]

        indices = []
        for experiment in self.experiment_design.values:
            entry = index[(index.values == experiment).all(1)]
            indices.append(entry.index.values[0])

        self.encoded_design = self.reaction.data.iloc[indices]

class external_design:
    """
    Get experiments corresponding to an external design. Utilized to
    import D-optimal designs from R.
    """
    
    def __init__(self, reaction, components, design):
        
        self.reaction = reaction
        self.design = design
        
        # Get component submatrices
        self.subs = component_submatrix(reaction, components)
        
        # Get unique sets
        self.unique = {}
        for key in self.subs:
            self.unique[key] = self.subs[key].drop_duplicates().reset_index(drop=True).iloc[:,0]
        
        # Count component levels
        self.levels = [len(self.unique[key]) for key in self.unique]
        
        # Design information
        self.N = len(self.levels)
        self.names = [key for key in self.unique]
    
    def get_experiments(self):
        """
        Get experiments corresponding to the design.
        """
            
        # Fill in experiments
        experiments = []
        for i in range(len(self.design)):
            experiment = []
            row = self.design.iloc[i]
            for col in self.design.columns.values:
                experiment.append(self.unique[col].iloc[row[col]])
            experiments.append(experiment)
        
        self.experiment_design = pd.DataFrame(experiments, 
                                              columns=self.design.columns.values)
        
    def encoded(self):
        """
        Get encoded experiments corresponding to the design.
        """
        
        index = self.reaction.base_data[self.reaction.index_headers]

        indices = []
        for experiment in self.experiment_design.values:
            entry = index[(index.values == experiment).all(1)]
            indices.append(entry.index.values[0])

        self.encoded_design = self.reaction.data.iloc[indices]
