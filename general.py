# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:54:49 2024

@author: danie
"""
import os
import sys
import pathlib
import pickle as pkl
import numpy as np
from numpy import matlib
from simulation_whiskers.simulate_task import simulate_session, session2feature_array, session2labels
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False
    


def pca_trials(sim_params, n=None, sum_bins=False, omit_angle=False, center=True, plot=True, scale=False, face_task=None, edge_task=None, face_cmap='cool', edge_cmap='binary', alpha=0.4, size=5.0, jitter=0.1, linewidth=1.0, save=True, output_directory=None):
    """
    Run whisker simulation then project data on principal components.

    Parameters
    ----------
    sim_params : dict
        Dict of whisker simulation parameters.Same as for simulate_session().

    n : int
        Number of principal components to keep. If None, keep all. 
    
    sum_bins : bool, optional
        Whether to sum features across time bins before running principal   
        component analysis (PCA) on. 
    
    center : bool, optional
        Whether to mean-subtract before running PCA. 
    
    scale : bool, optional
        Whether to Z-score data before running PCA. Note that `center` must
        also be set to True in order to use this option. 
    
    save : bool, optional
        Whether to save results. 
    
    output_directory : str, optional
        Where to save results if `save` is True. If None, saves results to 
        current directory.

    Returns
    -------
    Session : pandas.core.frame.DataFrame
        Same format as output of simulate_session() function, except with
        additional `feature_PCs` field. For each trial, `feature_PCs` is an n-
        element array, where n is the number of principal components retained.
        The i-th element of each trial's `feature_PCs` array represents the 
        projection of that trial's whisker data onto the i-th PC. Note that 
        PCs are computed across all trials combined, i.e., are the eigenvectors
        of the covariance matrix of the t-by-f feature matrix, where t is the
        number of trials and f is the number of features per trial. 

    """
    
    # Define field to run PC on:
    if not sum_bins:
        field='features'
    else:
        field='features_bins_summed'
    
    # Simulate session:
    Session=simulate_session(sim_params, sum_bins=True)
        
    # Extract features from session:
    F=session2feature_array(Session, field=field, omit_angle=omit_angle) # trials-by-features
    
    # Preprocess data for PCA: 
    if center==True and scale==False:
        mu=np.mean(F,0)
        mu=np.expand_dims(mu,0)
        n_obs=F.shape[0]
        mu=matlib.repmat(mu,n_obs,1)
        F=F-mu
    elif center==True and scale==True:
        F=StandardScaler().fit_transform(F)
    elif center==False and scale==True:
        raise AssertionError('Invalid standardization parameters center=False and scale=True; data must be centered before being scaled.')
    
    # Run PCA: 
    pca=PCA(n_components=n)
    F_hat=pca.fit_transform(F)
    
    # Write PCs back into Sessions dataframe:
    Session['feature_PCs']=[x for x in F_hat]
    
    # Plot PCs:
    if plot:
        fig=plot_trial_PCs(Session, field='feature_PCs', face_task=face_task, 
                        edge_task=edge_task, face_cmap=face_cmap, edge_cmap=edge_cmap, alpha=alpha, size=size, jitter=jitter, linewidth=linewidth)
    
    # Save output if requested:
    if save:
        
        # Set default output directory:
        if output_directory is None:
            output_directory=os.getcwd()
            
        # If requested output directory does not exist, create it:        
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Save scores: 
        scores_path=os.path.join(output_directory,'whisker_PCs.pickle')
        with open(scores_path, 'wb') as p:
            pkl.dump(F_hat, p)
            
        # Save plot:
        if plot:
            fig_path=os.path.join(output_directory,'whisker_PCs.png')
            fig.savefig(fig_path,dpi=500,bbox_inches='tight')
        
        # Write metadata if analysis_metadata module successfully imported:
        if 'analysis_metadata' in sys.modules:
            M=Metadata()
            M.parameters=sim_params
            M.add_param('center',center)
            M.add_param('scale',scale)
            M.add_param('sum_bins',sum_bins)
            M.add_param('omit_angle',omit_angle)
            M.add_param('face_task',face_task)
            M.add_param('edge_task',edge_task)
            M.add_param('jitter',jitter)
            M.add_output(scores_path)
            if plot:
                M.add_output(fig_path)
            metadata_path = os.path.join(output_directory, 'whisker_PCs_metadata.json')
            write_metadata(M, metadata_path)
        
    return Session
    


def plot_trial_PCs(Session, field='feature_PCs', face_task=None, edge_task=None, face_cmap='winter', edge_cmap='binary', size=0.5, alpha=0.25, jitter=0.1, linewidth=1):
    
    # Extract features from session:
    F=session2feature_array(Session, field=field)
    
    # Add jitter:
    F = F + jitter*np.random.standard_normal(size=F.shape)
    
    # Get number of of features:
    n_features=F.shape[-1]
    if n_features>=3:
        proj='3d'
    elif n_features==2:
        proj=None
    
    #Initialize plot:
    fig=plt.figure()
    ax=plt.axes(projection=proj)
    
    if face_task is not None:
        face_labels=session2labels(Session, face_task)
    else:
        face_labels='gray'
    
    # If partitioning data and plotting different markers for each:
    if edge_task is not None:    
        edge_labels=session2labels(Session, edge_task)
        
        # Define RGB triples for edges:
        scalarMap=cmx.ScalarMappable(norm=None, cmap=edge_cmap)
        edgecolors=scalarMap.to_rgba(edge_labels)
        edgecolors=edgecolors[:,0:3] #remove alpha, extra random dimension
    else:
        edgecolors=None
    
    # 3D case:    
    if n_features>=3:

        # Plot 0-th partition:
        scatter=ax.scatter3D(F[:,0], F[:,1], F[:,2], c=face_labels, cmap=face_cmap, edgecolors=edgecolors, alpha=alpha, s=size, linewidths=linewidth)
        
    # 2D case:        
    elif n_features==2:
        scatter=ax.scatter(F[:,0], F[:,1], c=face_labels, cmap=face_cmap, edgecolors=edgecolors, s=size, alpha=alpha, linewidths=linewidth)

        
    """
        # Split data into partitions to be plotted with different markers:
        partition0_data=F[edge_labels.astype('bool')]
        partition1_data=F[~edge_labels.astype('bool')]
        
        # Split color labels into partitions if necessary:
        if face_task is not None:
            partition0_face_labels=face_labels[edge_labels.astype('bool')]
            partition1_face_labels=face_labels[~edge_labels.astype('bool')]
        else:
            partition0_face_labels='gray'
            partition1_face_labels='gray'            
        
        # 3D case:    
        if n_features>=3:

            # Plot 0-th partition:
            scatter0=ax.scatter3D(partition0_data[:,0], partition0_data[:,1], partition0_data[:,2], c=partition0_face_labels, cmap=face_cmap, alpha=alpha, s=size, marker=markers[0])
        
            # Plot 1st partition:
            scatter1=ax.scatter3D(partition1_data[:,0], partition1_data[:,1], partition1_data[:,2], c=partition1_face_labels, cmap=face_cmap, s=size, marker=markers[1])
        
        #2D case:
        elif n_features==2:
            
            # Plot 0-th partition:
            scatter0=ax.scatter(partition0_data[:,0], partition0_data[:,1], c=partition0_face_labels, cmap=face_cmap, alpha=alpha, s=size, marker=markers[0])
        
            # Plot 1st partition:
            scatter1=ax.scatter(partition1_data[:,0], partition1_data[:,1], c=partition1_face_labels, cmap=face_cmap, alpha=alpha, s=size, marker=markers[1])
    
        
    # If plotting only one marker: 
    else:

        # 3D case:        
        if n_features>=3:
            scatter=ax.scatter3D(F[:,0], F[:,1], F[:,2], c=face_labels, cmap=face_cmap, alpha=alpha, s=size)
            leg_elements=scatter.legend_elements()
    
        # 2D case:        
        if n_features==2:
            scatter=ax.scatter(F[:,0], F[:,1], c=face_labels, cmap=face_cmap, s=size, alpha=alpha)
            leg_elements=scatter.legend_elements()
    """
    
    # Set axis labels:
    ax.set_xlabel('PC 0')
    ax.set_ylabel('PC 1')
    if n_features>=3:
        ax.set_zlabel('PC 2')
    
    return fig