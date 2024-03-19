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
import seaborn as sns
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


def plot_coding_axes(sim_params, task0, task1, classifier='LogisticRegression', 
   face_cmap='winter', edge_cmap='binary', sum_bins=False, save_output=False, 
   output_directory=None):
    
    fig, ax = plt.subplots()
    
    # Define classifier:
    if classifier == 'LogisticRegression':
        clf0 = LogisticRegression()
        clf1 = LogisticRegression()
    elif classifier == 'SVM':
        clf0 = LinearSVC()
        clf1 = LinearSVC()
    
    # Run simulation:
    session = simulate_session(sim_params, save_output=False, sum_bins=sum_bins)
    
    # Extract features for current session:
    X = session2feature_array(session, field='features')
    
    # Compute labels for base and projection tasks:
    task0_labels = session2labels(session, task0)
    task1_labels = session2labels(session, task1)
    
    # Train decoders (option of SVM or Logistic?) on base task:
    clf0.fit(X, task0_labels)
    clf1.fit(X, task1_labels)
    
    # Project all trials down onto coding axes:
    task0_coding_axis = np.transpose(clf0.coef_)
    Xhat0 = np.matmul(X, task0_coding_axis)
    Xhat0 = Xhat0 - np.mean(Xhat0)
    
    task1_coding_axis = np.transpose(clf1.coef_)
    Xhat1 = np.matmul(X, task1_coding_axis)
    Xhat1 = Xhat1 - np.mean(Xhat1)
    
    # Define edge colors:
    scalarMap=cmx.ScalarMappable(norm=None, cmap=edge_cmap)
    edgecolors=scalarMap.to_rgba(task1_labels)
    edgecolors=edgecolors[:,0:3] #remove alpha, extra random dimension
    
    # Scatter data: 
    ax.scatter(Xhat0, Xhat1, c=task0_labels, cmap=face_cmap, edgecolors=edgecolors)    
    
    # Define axis labels, legend:    
    task0_key = list(task0[0].keys())[0] 
    # ^ Hack-y and won't work for multiclass tasks or tasks defined over multiple
    # variables, but will do for now
    plt.xlabel('{} coding axis'.format(task0_key))

    task1_key = list(task1[0].keys())[0] 
    plt.ylabel('{} coding axis'.format(task1_key))
    # ax.legend(handles=handles)
    
    # Save output:
    if save_output:
        
        # Set default output directory if necessary:
        if output_directory is None:
            output_directory = os.getcwd()
            
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(output_directory, 'plot_coding_axes.png')
        
        fig.savefig(output_path, bbox_inches='tight')
    
        # Define and save metadata:
        if 'analysis_metadata' in sys.modules:
            M=Metadata()
            params=dict()
            params['sim_params']=sim_params
            params['task0']=task0
            params['task1']=task1
            params['classifier']=classifier
            params['sum_bins']=sum_bins
            M.parameters=params
            M.add_output(output_path)
            metadata_path = os.path.join(output_directory, 'plot_coding_axes_metadata.json')
            write_metadata(M, metadata_path)
    
    return 



def plot_weight_heatmap(sim_params, task0, task1, classifier='LogisticRegression', 
   sum_bins=False, cmap='cool', save_output=False, output_directory=None):

    fig, ax = plt.subplots()
    
    # Define classifier:
    if classifier == 'LogisticRegression':
        clf0 = LogisticRegression()
        clf1 = LogisticRegression()
    elif classifier == 'SVM':
        clf0 = LinearSVC()
        clf1 = LinearSVC()
    
    # Run simulation:
    session = simulate_session(sim_params, save_output=False, sum_bins=sum_bins)
    
    # Extract features for current session:
    X = session2feature_array(session, field='features')
    
    # Compute labels for base and projection tasks:
    task0_labels = session2labels(session, task0)
    task1_labels = session2labels(session, task1)
    
    # Train decoders (option of SVM or Logistic?) on base task:
    clf0.fit(X, task0_labels)
    clf1.fit(X, task1_labels)    
    task0_coding_axis = np.transpose(clf0.coef_)    
    task1_coding_axis = np.transpose(clf1.coef_)
    
    # Sort weights by coding axis 0:
    order = np.argsort(np.squeeze(task0_coding_axis))
    task0_coding_axis_ordered = task0_coding_axis[order]
    task1_coding_axis_ordered = task1_coding_axis[order]        
    D = np.vstack([task0_coding_axis_ordered.T, task1_coding_axis_ordered.T])
    
    # Define labels:
    task0_key = list(task0[0].keys())[0] # < Hack, won't necessarily work for more complex tasks, but will work for now
    task1_key = list(task1[0].keys())[0] # < Hack, won't necessarily work for more complex tasks, but will work for now
    ylabels = [task0_key + ' weights', task1_key + ' weights']    

    # Plot data:
    sns.heatmap(D, cmap=cmap, yticklabels=ylabels, xticklabels=False)

    # Save output:
    if save_output:
        
        # Set default output directory if necessary:
        if output_directory is None:
            output_directory = os.getcwd()
            
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(output_directory, 'weight_heatmap.png')
        
        fig.savefig(output_path, bbox_inches='tight')
    
        # Define and save metadata:
        if 'analysis_metadata' in sys.modules:
            M=Metadata()
            params=dict()
            params['sim_params']=sim_params
            params['task0']=task0
            params['task1']=task1
            params['classifier']=classifier
            params['sum_bins']=sum_bins
            M.parameters=params
            M.add_output(output_path)
            metadata_path = os.path.join(output_directory, 'weight_heatmap_metadata.json')
            write_metadata(M, metadata_path)    

    return



def proj_code_plane(sim_params, base_task, proj_task, classifier='LogisticRegression', 
   sum_bins=False, face_cmap='winter', edge_cmap='binary', plot_coding_axes=True, save_output=False, 
   output_directory=None):
    
    fig, ax = plt.subplots()
    
    # Define classifier:
    if classifier == 'LogisticRegression':
        clf_base = LogisticRegression()
        clf_proj = LogisticRegression()
    elif classifier == 'SVM':
        clf_proj = LinearSVC()
        clf_proj = LinearSVC()
    
    # Run simulation:
    session = simulate_session(sim_params, save_output=False, sum_bins=sum_bins)
    
    # Extract features for current session:
    X = session2feature_array(session, field='features')
    
    # Compute labels for base and projection tasks:
    base_labels = session2labels(session, base_task)
    proj_labels = session2labels(session, proj_task)
    
    # Train decoder (option of SVM or Logistic?) on base task:
    clf_base.fit(X, base_labels)
    base_coding_axis = np.transpose(clf_base.coef_)
    
    clf_proj.fit(X, proj_labels)
    proj_coding_axis = np.transpose(clf_proj.coef_)
    
    # Orthogonalize projection coding axis v0 and base coding axis v1 to 
    # orthonormal vectors u0 and u1 using Gram-Schmidt algorithm as follows:
        
    # Step 0: u0 = v0
    # Step 1: u1 = v1 - proj_{u0}(v1) = ( <v1, u0>/<u0, u0> )u0
    # Step 2: normalize u0 and u1 (i.e. e0 = u0/||u0||, e1 = u1/||u1||)
    
    # Where e0 just ends up being a normalized version of v0 and e1 is an orthonormal
    # vector in the same plane as v0 and v1.
    
    # Define base axis:
    v0 = np.squeeze(base_coding_axis)
    u0 = v0
    
    # Orthogonalize projection coding axis:
    v1 = np.squeeze(proj_coding_axis)
    u1 = v1 - ( np.dot(v1,u0)/np.dot(u0,u0) )*u0    
    
    # Normalize:
    e0 = u0/np.linalg.norm(u0)
    e1 = u1/np.linalg.norm(u1)
    
    # Project data onto new axes:
    X_e0 = np.dot(X, e0)
    X_e1 = np.dot(X, e1)
    
    # Mean-subtract data:
    X_e0 = X_e0 - np.mean(X_e0)
    X_e1 = X_e1 - np.mean(X_e1)
    
    # Define edge colors:
    scalarMap=cmx.ScalarMappable(norm=None, cmap=edge_cmap)
    edgecolors=scalarMap.to_rgba(proj_labels)
    edgecolors=edgecolors[:,0:3] #remove alpha, extra random dimension
    
    # Scatter data: 
    ax.scatter(X_e0, X_e1, c=base_labels, cmap=face_cmap, edgecolors=edgecolors)    
    
    # Get some task labels:
    base_name = list(base_task[0].keys())[0] 
    proj_name = list(proj_task[0].keys())[0] 
    # ^ Hack-y and won't work for multiclass tasks or tasks defined over multiple
    # variables, but will do for now
    
    # Compute coding axes:
    base_coding_axis_e0 = np.dot(v0, e0)
    base_coding_axis_e1 = np.dot(v0, e1)    
    b_norm = np.linalg.norm([base_coding_axis_e0, base_coding_axis_e1])

    proj_coding_axis_e0 = np.dot(v1, e0)
    proj_coding_axis_e1 = np.dot(v1, e1)        
    p_norm = np.linalg.norm([proj_coding_axis_e0, proj_coding_axis_e1])
    
    # Compute angle between coding axes:
    s = np.dot([proj_coding_axis_e0, proj_coding_axis_e1]/p_norm, [base_coding_axis_e0, base_coding_axis_e1]/b_norm)
    deg = np.degrees(np.arccos(s))
    
    """
    plt.plot([0,base_coding_axis_e0/b_norm], [0, base_coding_axis_e1/b_norm], label='{} coding axis'.format(base_name))
    plt.plot([0,proj_coding_axis_e0/p_norm], [0, proj_coding_axis_e1/p_norm], label='{} coding axis'.format(proj_name))
    """
    
    # Plot coding axes:
    if plot_coding_axes:
        plt.arrow(0, 0, base_coding_axis_e0/b_norm, base_coding_axis_e1/b_norm, color='blue', head_width=0.05, label='{} coding axis'.format(base_name))
        plt.arrow(0, 0, proj_coding_axis_e0/p_norm, proj_coding_axis_e1/p_norm, color='orange', head_width=0.05, label='{} coding axis'.format(proj_name))
    
    # Compute means:
    means = np.empty((2,2,2))
    means[:] = np.nan
    
    # Task 0, condition 0:
    means[0,0,0] = np.mean(X_e0[base_labels.astype(bool) & proj_labels.astype(bool)]) # Dimension 0
    means[0,0,1] = np.mean(X_e1[base_labels.astype(bool) & proj_labels.astype(bool)]) # Dimension 1    
    
    # Task 0, condition 1:
    means[0,1,0] = np.mean(X_e0[base_labels.astype(bool) & ~proj_labels.astype(bool)]) 
    means[0,1,1] = np.mean(X_e1[base_labels.astype(bool) & ~proj_labels.astype(bool)])     
        
    # Task 1, condition 0:
    means[1,0,0] = np.mean(X_e0[~base_labels.astype(bool) & proj_labels.astype(bool)]) 
    means[1,0,1] = np.mean(X_e1[~base_labels.astype(bool) & proj_labels.astype(bool)])     
        
    # Taks 1, condition 1:
    means[1,1,0] = np.mean(X_e0[~base_labels.astype(bool) & ~proj_labels.astype(bool)]) 
    means[1,1,1] = np.mean(X_e1[~base_labels.astype(bool) & ~proj_labels.astype(bool)])     
        
    # Define some colors:
    scalarMap_base=cmx.ScalarMappable(norm=None, cmap=face_cmap)
    facecolors=scalarMap_base.to_rgba(base_labels)
    facecolors=facecolors[:,0:3]
    first_base1 = np.where(base_labels)[0][0]
    first_base0 = np.where(~base_labels.astype(bool))[0][0]
    base0_color = facecolors[first_base0]
    base1_color = facecolors[first_base1]
    
    # Scatter means:
    plt.scatter([means[0,0,0]], [means[0,0,1]], s=300, marker='P', c=base1_color, edgecolor='red', linewidths=3)
    plt.scatter([means[0,1,0]], [means[0,1,1]], s=300, marker='P', c=base1_color, edgecolor='pink', linewidths=3)
    plt.scatter([means[1,0,0]], [means[1,0,1]], s=300, marker='P', c=base0_color, edgecolor='red', linewidths=3)
    plt.scatter([means[1,1,0]], [means[1,1,1]], s=300, marker='P', c=base0_color, edgecolor='pink', linewidths=3)
    
    yl = plt.ylim()
    xl = plt.xlim()
    minmin = min([min(yl), min(xl)])
    maxmax = max([max(yl), max(xl)])
    plt.xlim([minmin, maxmax])
    plt.ylim([minmin, maxmax])
    
    dimstr = '{} - {} plane dim '.format(base_name, proj_name)    
    plt.xlabel(dimstr + '0')
    plt.ylabel(dimstr + '1')
    plt.legend()
    ax.set_aspect('equal', adjustable='box')
    
    # Save output:
    if save_output:
        
        # Set default output directory if necessary:
        if output_directory is None:
            output_directory = os.getcwd()
            
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(output_directory, 'proj_code_plane.png')
        
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
            metadata_path = os.path.join(output_directory, 'proj_code_plane_metadata.json')
            write_metadata(M, metadata_path)   
    
    return X_e0, X_e1, deg