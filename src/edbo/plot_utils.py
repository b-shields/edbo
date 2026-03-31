# -*- coding: utf-8 -*-

# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from .math_utils import model_performance

from .pd_utils import to_torch

# Scatter plot for pred-obs

def plot2d(x,y,export_path=None):
    """
    Scatter plot with y=x line.
    """

    plt.cla()    
    
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (6, 6)
    plt.scatter(np.array(x), np.array(y), color='black', alpha=0.4)
    plt.xlabel('x')
    plt.ylabel('y')
    
    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

# Scatter plot for pred-obs

def scatter(pred,obs,plot_label,export_path=None):
    """
    Scatter plot with y=x line.
    """
    plt.cla()    
    upper = np.array([pred.max(),obs.max()]).max()
    lower = np.array([pred.min(),obs.min()]).min()
    
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (6, 6)
    plt.scatter(np.array(pred), np.array(obs), color='black', alpha=0.4)
    plt.xlabel('predicted')
    plt.ylabel('observed')
    plt.title(plot_label)
    plt.plot([lower,upper], [lower,upper], 'k-', alpha=0.75, zorder=0)
    
    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
              
# t-SNE plot

def tsne_plot(data,y=[],label='y',colors='hls', export_path=None, legend=None):
    """
    t-SNE plot for domain and progress visualization.
    """
    
    from sklearn.manifold import TSNE
    import seaborn as sns
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (5, 5)
    
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=10)
    tsne_results = tsne.fit_transform(data)
    
    df = pd.DataFrame()
    df['t-SNE1'] = tsne_results[:,0]
    df['t-SNE2'] = tsne_results[:,1]
    
    if len(y) == len(data):
        df[label] = y
        hue = label
    else:
        hue = None
    ax = sns.scatterplot(
            x="t-SNE1", y="t-SNE2",
            hue=hue,
            palette=sns.color_palette(colors, len(df[label].drop_duplicates())),
            data=df,
            legend=legend,
            alpha=0.8
            )
    
    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
    else:
        plt.show()

# scatter plot

def scatter_overlay(df, y=[], label='y', colors='hls', export_path=None, legend=None):
    """
    Scatter for 2D domain and progress visualization.
    """
    
    import seaborn as sns
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (6, 6)
    
    if len(y) == len(df):
        df[label] = y
        hue = label
    else:
        hue = None
    sns.scatterplot(
            x=df.columns.values[0], 
            y=df.columns.values[1],
            hue=hue,
            palette=sns.color_palette(colors, len(df[label].drop_duplicates())),
            data=df,
            legend=legend,
            alpha=0.8
            )
    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
    else:
        plt.show()

# Plot convergence 

def max_observed(points, batch_size):
    """
    Compute max observed.
    """
    
    index = []
    max_obs = []
    for i in range(round(len(points)/batch_size)):
        current_max = points[:batch_size*(i+1)].max()
        max_obs.append(current_max)
        index.append(i+1)
        
    return index, max_obs

def rate(seq):
    """
    Rate of convergence in time.
    """
    
    sequence = []
    for i in range(len(seq)-1):
        r = (seq[i+1] - seq[i])
        sequence.append(r)
        
    return sequence

def plot_convergence(data, batch_size, avg=False, export_path=None):
    """
    Plot optimizer convergence. 
    """
    
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (10, 5)

    points = np.array(data)
    index, max_obs = max_observed(points, batch_size)
    conv_rate = rate(max_obs)
        
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set(xlabel='Batch', ylabel='Max Observed')
    ax1.set_title('Convergence')
    ax1.plot(index, max_obs, 'o-', color="r")
    ax2.plot(index[:-1], conv_rate, '-o', color='b')
    ax2.set(xlabel='d/Batch', ylabel='d/MaxObserved')
    ax2.set_title('Rate')
    fig.tight_layout()
    
    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
    else:
        return plt.show()
    
# Average convergence with standard deviation bars
        
def average_convergence(data, partition):
    """
    Average convergence output for plots.
    """

    max_obs_list = []
    for data_i in np.array(data):
        
        points = np.array(data_i)
        max_obs = []
        index = []

        for i in range(round(len(points)/partition)):
            current_max = points[0:partition*(i+1)].max()
            max_obs.append(current_max)
            index.append(i+1)
        
        max_obs_list.append(max_obs)
    
    mean = np.mean(max_obs_list, axis=0)
    std = np.std(max_obs_list, axis=0)
    
    return index, mean, std
        
def plot_avg_convergence(data, batch_size, export_path=None):
    """
    Plot average optimizer convergence. 
    """
    
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (10, 5)
    
    index, mean, std = average_convergence(data, batch_size)
    conv_rate = rate(mean)
        
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.set(xlabel='Batch', ylabel='Max Observed')
    ax1.set_title('Convergence (N:' + str(len(data)) + ', Batch Size:' + str(batch_size) + ')')
    ax1.plot(index, mean, 'o-', color="r")
    ax1.fill_between(index, mean-std, mean+std)
    ax2.plot(index[:-1], conv_rate, '-o', color='b')
    ax2.set(xlabel='d/Batch', ylabel='d/MaxObserved')
    ax2.set_title('Rate (N:' + str(len(data)) + ', Batch Size:' + str(batch_size) + ')')
    fig.tight_layout()

    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
    else:
        return plt.show()
    
# Plot average convergence for multiple cases with legend
    
def compare_convergence(data_list, batch_sizes, legend_list=None, xlabel='Batch' ,export_path=None):
    """
    Plot average optimizer convergence for a list of runs. 
    """
    
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (10, 5)
    
    if type(batch_sizes) == type(1):
        batch_sizes = [batch_sizes for i in range(len(data_list))]
    
    if legend_list == None:
        legend_list = ['result' + str(n) for n in range(len(data_list))]
    
    ndata = [len(data) for data in data_list]
    if len(set(ndata)) == 1:
        ndata = ndata[0] 
    
    if len(set(batch_sizes)) == 1:
        bs = batch_sizes[0]
    else:
        bs = batch_sizes
    
    # Define a color mapping
    colormap = plt.cm.viridis
    
    mean_list = []
    rate_list = []
    index_list = []
    for i in range(len(data_list)):
        index, mean, std = average_convergence(data_list[i], batch_sizes[i])
        index_list.append(index)
        mean_list.append(mean)
        rate_list.append(rate(mean))
        
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, len(data_list))])
    ax1.set(xlabel=xlabel, ylabel='Max Observed')
    ax1.set_title('Convergence (N:' + str(ndata) + ', Batch Size:' + str(bs) + ')')
    for i in range(len(mean_list)):
        ax1.plot(index_list[i], mean_list[i], 'o-', label=legend_list[i])
    ax1.legend(loc='lower right', shadow=True)   
    
    ax2.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, len(data_list))])
    ax2.set(xlabel='d/'+xlabel, ylabel='d/MaxObserved')
    ax2.set_title('Rate (N:' + str(ndata) + ', Batch Size:' + str(bs) + ')')
    for i in range(len(rate_list)):
        ax2.plot(index_list[i][:-1], rate_list[i], 'o-', label=legend_list[i])
    ax2.legend(loc='upper right', shadow=True)
    fig.tight_layout()
    
    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
    else:
        return plt.show()

# Regression results
        
def pred_obs(pred, obs, title='Fit', return_data=False, export_path=None, return_scores=False):
    """
    Run a regression using the trained GP and return
    pred-obs plot for known data. return_data = True
    gives pred-obs data.
    """
    
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (6, 6)

    predobs = pd.DataFrame()
    predobs['pred'] = np.array(pred)
    predobs['obs'] = np.array(obs)
    rmse, r2 = model_performance(predobs['pred'],predobs['obs'])
    rmse, r2 = np.round(rmse,2), np.round(r2,2)
    
    if return_data == True:
        return predobs
    elif return_scores == True:
        return rmse, r2
    else:
        return scatter(
            predobs['pred'],
            predobs['obs'],
            title + '(RMSE = ' + str(rmse) + ', R^2 = ' + str(r2) + ')',
            export_path=export_path)      
    
# Spearman correlation map

def spearman_map(df, export_path=None):
    """
    Plot a spearman correlation dendrogram and heat map for a dataframe.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    corr = spearmanr(df).correlation
    corr_linkage = hierarchy.ward(corr)

    dendro = hierarchy.dendrogram(corr_linkage, 
                              labels=df.columns.values, 
                              ax=ax1,
                              leaf_rotation=90)

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    fig.tight_layout()
    
    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
   
    return plt.show()

# Plot a horizontal bar chart 

def hor_bar(values, names=[], size=(10,20), title='', xlabel='', ylabel='', sort=True, export_path=None, color='gray'):
    """
    Horizontal bar bar chart.
    """
    
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = size
    
    if len(names) != len(values):
        names = np.array([str(i) for i in range(len(values))])
    
    if sort == True:
        sort_index = np.array(values).argsort()
        names = names[sort_index]
        values = values[sort_index]
    
    # Plot
    
    fig, ax = plt.subplots() 
    width = 0.75 # the width of the bars 
    ind = np.arange(len(values))  # the x locations for the groups
    ax.barh(ind, values, width, color="gray")
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(names, minor=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      
    
    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

def prior_plot(prior_list, X, legends, title='', xlabel='x', ylabel='density', 
               export_path=None, legend_position='lower left', log=False):
    """
    Plot priors on X.
    """

    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['figure.figsize'] = (6, 6)
    
    # Get log probs
    logprobs = []
    for prior in prior_list:
        logprob = []
        for x in X:
            if log:
                logprob.append(float(prior.log_prob(x)))
            else:
                logprob.append(float(10**prior.log_prob(x))) 
        logprobs.append(logprob)
    
    # Plot
    for i in range(len(logprobs)):
        plt.plot(X, logprobs[i], label=legends[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_position, shadow=True)
    
    # Return
    if type(export_path) == type(''):
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
        
    return plt.show()    
        
# Plot choices in a lower dimensional space

def plot_choices(obj, proposed, export_path=None):
    """!
    @brief Plot low dimensional embedding (t-SNE) of initialization
           points over user specified domain.
        
    Parameters
    ----------
    @param[in] obj (class): Objective object with methods defined in 
               bro.objective.
    @param[in] proposed (DataFrame): Proposed experiments.
    @param[in] export_path (None, str): Path to export visualization.
        
    Returns
    ----------
    (DataFrame) Selected domain points.
    """
        
    X = pd.concat([obj.domain, obj.results.drop('yield', axis=1), proposed])
    domain = ['Domain' for i in range(len(obj.domain))]
    results = ['Experiments' for i in range(len(obj.results))]
    init = ['Selected' for i in range(len(proposed))]
    labels = domain + results + init
        
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
    
# Partial Dependence Plots

def pdp_points(bo_object, descriptor, config='mean', grid=100, seed=1):
    """
    Partial dependence of a given descriptor on the outcome of model
    predictions. Acts on a BO object after model training. Note: config='mean'
    sets inactive dimensions to their mean value and config='sample' randomly
    samples the domain and sets the inactive dimensions to the sample values.
    """
    
    # Descriptors
    columns = bo_object.obj.domain.columns.values
    
    # Set inactive dimensions
    if config == 'mean':
        base_list = list(bo_object.obj.domain.drop(descriptor, axis=1).mean())
    elif 'sample' in config:
        base_list = list(bo_object.obj.domain.drop(descriptor, axis=1).sample(1, random_state=seed).values[0])
    
    # Build partial dependence domain
    index = np.argwhere(columns == descriptor)[0][0]
    grid_points = np.linspace(bo_object.obj.domain[descriptor].min(), 
                              bo_object.obj.domain[descriptor].max(), 
                              grid)
    pdp_domain = []
    for i in grid_points:
        row = list(base_list)
        row.insert(index, i)
        pdp_domain.append(row)
    
    # Make predictions over domain
    pred = bo_object.model.predict(to_torch(pdp_domain, gpu=bo_object.obj.gpu))
    pred = bo_object.obj.scaler.unstandardize(pred)
    
    return grid_points, pred

def dependence_plot(bo_object, descriptors, samples=100, export_path=None):
    """
    Plot partial dependence of a given dimension with all other 
    dimensions set to the domain mean. Plot N samples of other 
    descriptor configurations for inactive dimensions drawn from
    the optimization domain.
    """
    
    matplotlib.rcParams['font.size'] = 12
    
    descriptors = list(descriptors)
    indices = [i for i in range(len(descriptors))]
    
    # Matplotlib subplots for each descriptor
    fig, axs = plt.subplots(1, len(descriptors), figsize=(len(descriptors)*4, 5), constrained_layout=True)
    
    # Generate plots
    for descriptor, ax in zip(descriptors, axs):
        
        # Samples
        for sample in range(samples):
            domain, pdp = pdp_points(bo_object, descriptor, config='sample', seed=sample)
            if sample == 0:
                ax.plot(domain, pdp, color='black', alpha=0.2, label='Constant=Sample')
            else:
                ax.plot(domain, pdp, color='black', alpha=0.2)
        
        # Mean
        domain, pdp = pdp_points(bo_object, descriptor, config='mean')
        ax.plot(domain, pdp, linewidth=5, label='Constant=Mean')
        
        obj_min = bo_object.obj.results_input()[bo_object.obj.target].min()
        obj_max = bo_object.obj.results_input()[bo_object.obj.target].max()
        
        ax.set_xlabel(descriptor)
        ax.set_ylabel('predicted ' + bo_object.obj.target)
        ax.set_xlim(min(domain), max(domain))
        ax.set_ylim(obj_min - abs(obj_max)*0.1, obj_max + abs(obj_max)*0.1)
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
    
    # Return
    if type(export_path) == type(''):
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
    
    return plt.show()

# PCA and t-SNE embeddings

def embedding_plot(data, labels=[], export_path=None):
    """
    PCA and t-SNE plots.
    """
    
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # PCA
    pca = PCA(n_components=2, copy=True)
    pca.fit(data)
    pca_results = pca.transform(data)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=10)
    tsne_results = tsne.fit_transform(data)
    
    # Unique labels
    clusters = list(set(labels))
    labels = np.array(labels)

    # Plot with matplotlib    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    
    if len(clusters) < 2:
        ax1.scatter(pca_results[:,0],
                    pca_results[:,1],
                    s=30,
                    lw=0,
                    alpha=0.5,
                    color='black')
        ax2.scatter(tsne_results[:,0],
                    tsne_results[:,1],
                    s=30,
                    lw=0,
                    alpha=0.5,
                    color='black')
    
    else:
        for cluster_i in clusters:
            index_i = np.argwhere(labels == cluster_i).flatten()
            ax1.scatter(pca_results[:,0][index_i],
                        pca_results[:,1][index_i],
                        s=30,
                        lw=0,
                        alpha=0.5)
            ax2.scatter(tsne_results[:,0][index_i],
                        tsne_results[:,1][index_i],
                        s=30,
                        lw=0,
                        alpha=0.5)
    
    ax1.set(xlabel='PC1', ylabel='PC2')
    ax1.set_title('Principal Components Analysis')
    ax2.set(xlabel='t-SNE1', ylabel='t-SNE2')
    ax2.set_title('Stochastic Neighbor Embedding')
    fig.tight_layout()
    
    if export_path != None:
        plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
    else:
        plt.show()
