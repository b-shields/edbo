# -*- coding: utf-8 -*-

# Imports

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Permutation importance feature selection using a random forest model

class rf_permutation_importance:
    """Feature selection via random forest permutation importance
    
    Addapted from: "Permutation Importance with Multicollinear or Correlated 
    Features" in sklearn.
    """
    def __init__(self, use_data='all'):
        """        
        Parameters
        ----------
        use_data : str
            If 'all' use all availible data when training RF model. Else use
            an 80/20 split of the data.
        
        Returns
        ----------
        None
        """
        
        self.importance_type = use_data
        self.model = RandomForestRegressor(n_jobs=-1, 
                                           random_state=10, 
                                           n_estimators=500, 
                                           max_features='auto',
                                           max_depth=None,
                                           min_samples_leaf=1,
                                           min_samples_split=2)
    
    # Run model and compute permutation importance
    
    def run(self, df, target, n_repeats=5, random_state=1):
        """Fit model and compute permutation importance.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing experimental data.
        target : 'str'
            Column name for target (e.g., 'yield').
        n_repeats : int
            Number of times to permuate in order to get statistics.
        random_state : int
            Random seed used when using a training/test split.
        
        Returns
        ----------
        None
        """
        
        # Select training and test data
        
        if self.importance_type == 'all':
            X_train = df.drop(target, axis=1)
            y_train = df[target]
        else:
            X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1),
                                                                df[target],
                                                                test_size=0.2,
                                                                random_state=random_state)
        
        # Fit RF model and compute permutation importance
        
        self.model.fit(X_train, y_train)
        
        result = permutation_importance(self.model, 
                                         X_train,
                                         y_train, 
                                         n_repeats=n_repeats,
                                         random_state=random_state)
        
        # Set results to class variables
        
        self.result = result
        self.features = df.drop(target, axis=1).columns.values
        self.importances = pd.DataFrame(self.result.importances, 
                                        index=self.features)
        self.importances['mean'] = self.importances.mean(axis=1)
    
    def plot_importances(self, top_k=10, export_path=None):
        """Plot a importances as a box plot.
        
        Parameters
        ----------
        top_k : int
            Show top_k features according to permutation importance.
        export_path : None, str
            Export impotance plot to export_path as an SVG.
        
        Returns
        ----------
        matplotlib.pyplot
            Box plot of results.
        """
        
        perm_sorted_idx = self.result.importances_mean.argsort()
        
        plot_data = self.result.importances[perm_sorted_idx[-top_k:]].T
        plot_labels = self.features[perm_sorted_idx[-top_k:]]

        fig, ax = plt.subplots(1, figsize=(8, 16))
        ax.boxplot(plot_data, 
                   vert=False,
                   labels=plot_labels)
        
        if export_path != None:
            plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        
        return plt.show()
    
    def get_best(self, threshold):
        """Return descriptors with importance above threshold.
        
        Parameters
        ----------
        threshold : float
            Return a list of descriptors with importance above the specified
            threshold.
        
        Returns
        ----------
        numpy.array
            Array of descriptors with importance above the threshold.
        """
        
        best = self.importances[self.importances['mean'] > threshold]
        s = best['mean'].sum()
        tot = self.importances['mean'].sum()
        n = len(best)
        
        print('N features = ' + str(n) + ' | % Importance = ' + str(s/tot*100))
        
        return best.index.values
        
