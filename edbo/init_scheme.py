# -*- coding: utf-8 -*-

# Imports

import random
import pandas as pd
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score

from .plot_utils import tsne_plot, embedding_plot, scatter_overlay

# Init schemes

class Init:
    """Class represents different initialization schemes.
    
    Methods for selecting initial points on a user defined grid.
    """

    def __init__(self, method, batch_size, distance='gower', visualize=True):
        """        
        Parameters
        ----------
        method : str
            Sampling method. Opions include: 'rand', 'pam', 'kmeans', and 
            'external'.
        batch_size : int 
            Number of points to select.
        distance_metric : str 
            Distance metric to be used with PAM. Options include: 'gower', 
            'euclidean', and 'euclidean_square'.
        visualize : bool 
            Visualize color coded clusters selected via PAM or k-Means.
        
        """
        
        self.method = method
        self.batch_size = batch_size
        self.distance_metric = distance
        self.visualize = visualize

    def run(self, obj, seed=None, export_path=None):
        """Run initialization algorithm on user defined domain.
        
        Parameters
        ----------
        obj : edbo.objective
            Objective data container.
        seed : None, int
            Random seed for random selection and initial choice of medoids 
            or centroids.
        export_path : None, str
            Path to export visualization if applicable.
        
        Returns
        ----------
        pandas.DataFrame
            Selected domain points.
        """
        
        if self.method == 'rand':
            self.experiments = rand(obj, self.batch_size, seed=seed)
            
        elif self.method == 'pam':
            self.experiments = PAM(obj, 
                                   self.batch_size, 
                                   distance=self.distance_metric,
                                   visualize=self.visualize,
                                   seed=seed,
                                   export_path=export_path)
            
        elif self.method == 'kmeans':
            self.experiments = k_means(obj, 
                                   self.batch_size, 
                                   visualize=self.visualize,
                                   seed=seed,
                                   export_path=export_path)
            
        elif self.method == 'external':
            self.experiments = external_data(obj)
        else:
            print('Init: Error specify valid method.')
        
        return self.experiments
    
    def plot_choices(self, obj, export_path=None):
        """Plot low dimensional embeddingd of initialization points in domain.
        
        Parameters
        ----------
        obj : edbo.objective
            Objective data container.
        export_path : None, str
            Path to export visualization if applicable.
        
        Returns
        ----------
        pandas.DataFrame
            Selected domain points.
        """
        
        X = pd.concat([obj.domain.drop(self.experiments.index.values, axis=0), self.experiments])
        domain = ['Domain' for i in range(len(obj.domain.drop(self.experiments.index.values, axis=0)))]
        init = ['Initialization' for i in range(len(self.experiments))]
        labels = domain + init
        
        if len(X.iloc[0]) > 2:        
            tsne_plot(X,
                      y=labels,
                      label='Key',
                      colors='hls', 
                      legend='full',
                      export_path=export_path)
        
        elif len(X.iloc[0]) == 2:
            scatter_overlay(X,
                            y=labels,
                            label='Key',
                            colors='hls', 
                            legend='full',
                            export_path=export_path)
        
        elif len(X.iloc[0]) == 1:
            rep = pd.DataFrame()
            rep[X.columns.values[0]] = X.iloc[:,0]
            rep[' '] = [0 for i in range(len(X))]
            scatter_overlay(rep,
                            y=labels,
                            label='Key',
                            colors='hls', 
                            legend='full',
                            export_path=export_path)

# Random selection of domain points
        
def rand(obj, batch_size, seed=None):
    """Random selection of points.
        
    Parameters
    ----------
    obj : edbo.objective
            Objective data container.
    batch_size : int
        Number of points to be selected.
    seed : None, int
        Random seed.
        
    Returns
    ----------
    pandas.DataFrame 
        Selected domain points.
    """
        
    batch = obj.domain.sample(
            n=batch_size, 
            random_state=seed)
        
    return batch

# External data  
    
def external_data(obj):
    """External data reader.
        
    Parameters
    ----------
    obj : edbo.objective
            Objective data container.
               
    Returns
    ----------
    pandas.DataFrame
        Selected domain points.
    """
    
    print('\nUsing external results for initializaiton...\n')   
    
    return obj.results.drop(obj.target, axis=1)
    
    
def PAM(obj, batch_size, distance='gower', visualize=True, 
        seed=None, export_path=None):
    """Partitioning around medoids algorithm. 
    
    PAM function returns medoids of learned clusters.
           
    PAM implimentation from: https://pypi.org/project/pyclustering/
        
    Parameters
    ----------
    obj : edbo.objective
            Objective data container.
    batch_size : int
        Number of points to be selected. Batch size also determins the number 
        of clusters. PAM returns the medoids.
    distance : str 
        Distance metric to be used in the PAM algorithm. Options include: 
        'gower', 'euclidean', and 'euclidean_square'.
    visualize : bool 
        Visualize the learned clusters.
    seed : None, int 
        Random seed.
    export_path : None, str
        Path to export cluster visualization SVG image.
        
    Returns
    ----------
    pandas.DataFrame
        Selected domain points.
    """
        
    # print('\nInitializing using PAM...\n')
        
    # Set random initial medoids
    if type(seed) == type(1):
        random.seed(a=seed)
    initial_medoids = random.sample(range(len(obj.domain)), batch_size)

    # Load list of points for cluster analysis
    sample = obj.domain.values.tolist()

    # Create instance of K-Medoids algorithm
    if distance == 'gower':
        max_range = (obj.domain.max() - obj.domain.min()).values
        metric = distance_metric(type_metric.GOWER, max_range=max_range)
    elif distance == 'euclidean':
        metric = distance_metric(type_metric.EUCLIDEAN)
    elif distance == 'euclidean_square':
        metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)
            
    kmedoids_instance = kmedoids(sample, initial_medoids, metric=metric, tolerance=0.0001, itermax=300)
        
    # Run cluster analysis and obtain results
    kmedoids_instance.process()
    medoids = kmedoids_instance.get_medoids()
    medoids = obj.domain.iloc[medoids]
        
    # Display clusters
    if visualize == True:
        # Get clusters
        clusters = kmedoids_instance.get_clusters()
            
        # If low d use built in visualization
        if len(sample[0]) < 4:
            visualizer = cluster_visualizer()
            visualizer.append_clusters(clusters, sample)
            visualizer.show()
        else:
            columns = obj.domain.columns.values.tolist()
            columns.append('label')
            tsne_data = pd.DataFrame(columns=columns)
            for i in range(len(clusters)):
                data = obj.domain.iloc[clusters[i]].values.tolist()
                data = pd.DataFrame(data=data, columns=columns[:-1])
                data['label'] = [i] * len(clusters[i])
                tsne_data = pd.concat([tsne_data,data])
            embedding_plot(
                    tsne_data.drop('label',axis=1),
                    labels=tsne_data['label'].values.tolist(),
                    export_path=export_path
                    )
                    
    return medoids

def k_means(obj, batch_size, visualize=True, seed=None, export_path=None,
            n_init=1, return_clusters=False, return_centroids=False):
    """K-Means algorithm. 
    
    k_means function returns domain points closest to the means of learned clusters.
    
    Implementation from sklearn.
        
    Parameters
    ----------
    obj : edbo.objective
            Objective data container.
    batch_size : int 
        Number of points to be selected. Batch size also determins the number 
        of clusters. PAM returns the medoids.
    visualize : bool 
        Visualize the learned clusters.
    seed : None, int
        Random seed.
    export_path : None, str
        Path to export cluster visualization SVG image.
        
    Returns
    ----------
    pandas.DataFrame 
        Selected domain points.
    """
        
    # Run cluster analysis and choose best via silhouette
    cluster_sizes = [n for n in range(batch_size, batch_size+10)]
    
    scores = []
    for n_clusters in cluster_sizes:
        clusterer = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
        cluster_labels = clusterer.fit_predict(obj.domain)
        silhouette_avg = silhouette_score(obj.domain, cluster_labels)
        scores.append(silhouette_avg)
    best = cluster_sizes[np.argmax(scores)]
    
    print(best, 'clusters selected by silhouette score...')
    
    # Refit with best value
    clusterer = KMeans(n_clusters=best, random_state=seed, n_init=n_init)
    cluster_labels = clusterer.fit_predict(obj.domain)
        
    # Get points closes to the cluster means
    closest = pd.DataFrame(columns=obj.domain.columns)
    for i in range(best):
        cluster_i = obj.domain.iloc[np.where(clusterer.labels_ == i)]
        closest_i, _ = pairwise_distances_argmin_min(clusterer.cluster_centers_[[i]], cluster_i)
        closest = pd.concat([closest, cluster_i.iloc[closest_i]], sort=False)
    
    if return_centroids == True:
        return closest
    
    if len(closest) > batch_size:
        closest = closest.sample(batch_size, random_state=seed)
        
    # Display clusters
    if visualize == True:
        # Get clusters
        labels = clusterer.labels_   
        tsne_data = obj.domain.copy()
        tsne_data['label'] = labels
        
        if len(tsne_data.iloc[0]) > 2:
        
            embedding_plot(
                        tsne_data.drop('label',axis=1),
                        labels=tsne_data['label'].values.tolist(),
                        export_path=export_path
                        )
            
    if return_clusters == True:
        return cluster_labels
                    
    return closest







    
