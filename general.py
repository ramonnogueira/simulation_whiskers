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
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
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



def proj_code_axis(sim_params, base_task, proj_task, classifier='LogisticRegression', 
   sum_bins=False, save_output=False, output_directory=None):
    
    fig, ax = plt.subplots()
    
    # Define classifier:
    if classifier == 'LogisticRegression':
        clf = LogisticRegression()
    elif classifier == 'SVM':
        clf = LinearSVC()
    
    # Run simulation:
    session = simulate_session(sim_params, save_output=False, sum_bins=sum_bins)
    
    # Extract features for current session:
    X = session2feature_array(session, field='features')
    
    # Compute labels for base and projection tasks:
    base_labels = session2labels(session, base_task)
    proj_labels = session2labels(session, proj_task)
    
    # Train decoder (option of SVM or Logistic?) on base task:
    clf.fit(X, base_labels)
    
    # Project all trials down onto coding axis:
    base_coding_axis = np.transpose(clf.coef_)
    Xhat = np.matmul(X, base_coding_axis)
    Xhat = Xhat - np.mean(Xhat)
    
    # Iterate over conditions of projection task:
    handles = []
    proj_task_key = list(proj_task[0].keys())[0] # Hack-y and won't work for more complex tasks but will do for now
    for cx, condition in enumerate(np.unique(proj_labels)):
        
        # Extract trials of current condition:
        curr_cond_indices = np.where(proj_labels==condition)
        curr_cond_trials = Xhat[curr_cond_indices]
        
        # Plot histogram of current projection task condition trials:
        hs = ax.hist(curr_cond_trials, label='{}={}'.format(proj_task_key, proj_task[cx][proj_task_key]))
        handles.append(hs[2])
    
    # Define axis labels, legend:    
    base_task_key = list(base_task[0].keys())[0] 
    # ^ Hack-y and won't work for multiclass tasks or tasks defined over multiple
    # variables, but will do for now
    plt.xlabel('Projection onto {} coding axis'.format(base_task_key))
    plt.ylabel('Trial count')
    # ax.legend(handles=handles)
    ax.legend()
    
    # Save output:
    if save_output:
        
        # Set default output directory if necessary:
        if output_directory is None:
            output_directory = os.getcwd()
            
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(output_directory, 'proj_hist.png')
        
        fig.savefig(output_path, bbox_inches='tight')
    
        # Define and save metadata:
        if 'analysis_metadata' in sys.modules:
            M=Metadata()
            params=dict()
            params['sim_params']=sim_params
            params['base_task']=base_task
            params['proj_task']=proj_task
            params['classifier']=classifier
            params['sum_bins']=sum_bins
            M.parameters=params
            M.add_output(output_path)
            metadata_path = os.path.join(output_directory, 'simulation_metadata.json')
            write_metadata(M, metadata_path)
    
    return Xhat, base_labels, proj_labels

    

def self_v_other_proj(sim_params, task0, task1, n_iterations=5, classifier='LogisticRegression', 
   distance_measure='normalized', sum_bins=False, save_output=False, output_directory=None):
    
    tasks = [task0, task1]
    df = pd.DataFrame(columns=['base_task', 'base_task_name', 'proj_task', 'proj_task_name', 'diffs'])
    
    # Iterate over base tasks:
    for base_task in tasks:

        # Iterate over tasks to be projected onto base task:
        for proj_task in tasks:        

            # Iterate over simulations:
            curr_dict = {}
            curr_diffs = []
            for it in np.arange(n_iterations):
            
                # Run simulation:
                Xhat, base_labels, proj_labels = proj_code_axis(sim_params, 
                    base_task=base_task, proj_task=proj_task, classifier=classifier,
                    save_output=False)
                
                # Split data to be projected by class:
                proj_cond0_dat = Xhat[proj_labels.astype(bool)]
                proj_cond1_dat = Xhat[~proj_labels.astype(bool)]
                
                # Compute distance between centroids of different projected 
                # task classes along base task coding axis: 
                if distance_measure == 'normalized':
                    
                    mu0 = np.mean(proj_cond0_dat)
                    mu1 = np.mean(proj_cond1_dat)
                    
                    sig0 = np.var(proj_cond0_dat)
                    sig1 = np.var(proj_cond1_dat)
                    
                    #distance = np.abs( mu0 - mu1 ) / np.sqrt( (sig0+sig1) / 2 )
                    distance = np.abs( mu0 - mu1 ) 
                
                # TODO: Add support for other ways of measuring distance
                curr_diffs.append(distance)
            
            # Get convenient readable names for different tasks (hacky and won't
            # work for more complex tasks but will do for now):
            base_task_key = list(base_task[0].keys())[0] 
            proj_task_key = list(proj_task[0].keys())[0]
                
            curr_dict['base_task'] = base_task
            curr_dict['base_task_name'] = base_task_key
            curr_dict['proj_task'] = proj_task
            curr_dict['proj_task_name'] = proj_task_key
            curr_dict['diffs'] = np.array(curr_diffs)
            
            # Update dataframe:
            df = df.append(curr_dict, ignore_index=True)
            
    return df



def plot_self_other_proj(df):
    
    fig, ax = plt.subplots()
    base_task_names = np.unique(df.base_task_name) # < Get all base tasks included in input dataframe
    proj_task_names = np.unique(df.proj_task_name)
    all_names = np.unique(list(base_task_names) + list(proj_task_names))

    # Define color code:
    colors = ['green', 'orange', 'red', 'blue'] # < Hacky, will not work for >4 tasks overall, but will do for now
    color_dict = dict()
    for nx, name in enumerate(all_names):
        color_dict[name] = colors[nx]
        
    # Iterate over base tasks:
    horiz = 0
    x = []
    y = []
    yerrs = []
    facecolors = []
    edgecolors = []
    legend_labels = []
    for base_task in base_task_names:
        
        # Get names of other tasks projected down onto current base task:
        curr_base_task_rows = df[df.base_task_name==base_task]
        curr_proj_task_names = np.unique(curr_base_task_rows.proj_task_name)
        
        # Iterate over tasks projected down onto current base task coding axis:
        for proj_task in curr_proj_task_names:
            
            # Get row (presumably one) corresponding to current task projected onto base task:
            curr_row = curr_base_task_rows[curr_base_task_rows.proj_task_name==proj_task]
            curr_mu = np.mean(curr_row.iloc[0]['diffs'])
            curr_std = np.std(curr_row.iloc[0]['diffs'])
            
            # Get colors:
            curr_fill_color = color_dict[base_task]
            curr_edge_color = color_dict[proj_task]
            
            horiz = horiz + 1
            
            # Define label:
            curr_label = '{} projected onto {}'.format(proj_task, base_task)
            
            x.append(horiz)
            y.append(curr_mu)
            yerrs.append(curr_std)
            facecolors.append(curr_fill_color)
            edgecolors.append(curr_edge_color)
            legend_labels.append(curr_label)

        # Add space between bars between base tasks:
        horiz = horiz + 1

    # Plot:
    br = plt.bar(np.array(x), np.array(y), yerr=np.array(yerrs), color=facecolors, edgecolor=edgecolors, linewidth=2)
    #br = ax.bar(np.array(horiz), np.array(curr_mu), edgecolor=curr_edge_color, linewidth=10)
    #br.set_edgecolor(curr_edge_color)
    #br.set_linewidth(15)
    plt.ylabel('distance between projected task class centroids')
    plt.legend(br, legend_labels)
            

        
    
    return