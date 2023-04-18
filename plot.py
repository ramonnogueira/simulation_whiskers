# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:01:05 2023

@author: danie
"""

import sys
import os
import pathlib
import numpy as np
from scipy.stats import sem
import h5py
import matplotlib.pyplot as plt
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False
    
    
def plot_iterate_autoencoder_results(inpt, plot_train=False, save_output=False, output_directory=None):
    """
    Plot loss vs training epoch and decoder performance vs training epoch for 
    results of iterate_fit_autoencoder() function.
    

    Parameters
    ----------
    inpt : dict | str 
        Results from iterate_fit_autoencoder() function. If dict, then should 
        define following keys:
        
            loss_epochs : numpy.ndarray
                f-by-p matrix of overall loss across training epochs, where f 
                is the number of files (repetitions) and p is the number of 
                training epochs per repetition.
                
            perf_hidden : numpy.ndarray
                f-by-p-by-2 matrix of classifier performance based on hidden
                layer activity. Each p-by-2 slice corresponding to a single 
                file includes both training and test performance (hence 2 
                columns).
            
            perf_orig : numpy.ndarray
                f-by-2 matrix of classifier performance based on original input
                data.
        
            perf_out : numpy.ndarray
                f-by-p-by-2 matrix of classidier performance based on
                reconstructed input. 
        
        If str, then should be path to HDF5 file with datasets corresponding to 
        dict keys defined above.
    
    save_output : bool, optional
        Whether to save generates figures to disk. The default is False.
    
    output_directory : str, optional
        Directory where figures should be saved if `save_output` is True. Set
        to current directory by defaulty.

    Returns
    -------
    loss_plot : matplotlib.figure.Figure
        Figure of overall loss versus training epoch, averaged across files 
        (repetitions).
        
    perf_plot : matplotlib.figure.Figure
        Figure of classifier performance for original inputs, hidden layer
        activity, and reconstructed input vs training epoch.

    """
    
    # Load results:
    if type(inpt)==str:
        file=h5py.File(inpt,'r')
        loss_epochs=np.array(file['loss_epochs'])
        perf_hidden=np.array(file['perf_hidden'])
        perf_orig=np.array(file['perf_orig'])
        perf_out=np.array(file['perf_out'])
        if 'perf_orig_mlp' in file.keys():
            perf_orig_mlp=np.array(file['perf_orig_mlp'])
        file.close()
    elif type(inpt)==dict:
        loss_epochs=inpt['loss_epochs']
        perf_hidden=inpt['perf_hidden']
        perf_orig=inpt['perf_orig']
        perf_out=inpt['perf_out']
        if 'perf_orig_mlp' in inpt:
            perf_orig_mlp=inpt['perf_orig_mlp']
        
    n_epochs=loss_epochs.shape[1]
    
    # Plot Loss
    loss_plot, ax = plt.subplots(1)
    loss_m=np.mean(loss_epochs,axis=0) # Average across files
    loss_m=np.mean(loss_m,axis=0) # Average again, this time across cross-validations    
    plt.plot(loss_m)
    plt.ylabel('Training Loss')
    plt.xlabel('Epochs')
    plt.show()
    
    # Plot performance
    perf_plot, ax = plt.subplots(1)
    perf_m=np.mean(perf_orig,axis=0)
    perf_out_m=np.mean(perf_out,axis=0) # Average across files
    #perf_out_m=np.mean(perf_out_m,axis=0) # Average again, this time across cross-validations
    perf_hidden_m=np.mean(perf_hidden,axis=0) # Average across filess
    #perf_hidden_m=np.mean(perf_hidden_m,axis=0) # Average again, this time across cross-validations

    plt.plot(perf_out_m[:,1],color='blue',label='Out')
    plt.plot(perf_hidden_m[:,1],color='red',label='Hidden')
    plt.plot(perf_m[1]*np.ones(n_epochs),color='grey',label='Input')

    if 'perf_orig_mlp' in locals():
        perf_mlp_m=np.mean(perf_orig_mlp,axis=0)    
        plt.plot(perf_mlp_m[1]*np.ones(n_epochs),color='green',label='MLP')
    
    if plot_train:
        plt.plot(perf_out_m[:,0],color='blue',linestyle='--', label='Out train')
        plt.plot(perf_hidden_m[:,0],color='red',linestyle='--', label='Hidden train')
        plt.plot(perf_m[0]*np.ones(n_epochs),color='grey',linestyle='--', label='Input train')
        if 'perf_orig_mlp' in locals():
            plt.plot(perf_mlp_m[0]*np.ones(n_epochs),color='green',linestyle='--', label='MLP train')
    
    plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
    plt.ylim([0,1.1])
    plt.ylabel('Decoding Performance')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.show()
    
    if save_output:
        
        # Make current folder default:
        if output_directory==None:
            output_directory=os.getcwd()
            
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
            
        # Save SVGs of figures:
        loss_plot_path=os.path.join(output_directory, 'loss.svg')
        loss_plot.savefig(loss_plot_path)
        
        perf_plot_path=os.path.join(output_directory, 'perf.svg')
        perf_plot.savefig(perf_plot_path)

        # Save metadata if analysis_metadata successfully imported:
        if 'analysis_metadata' in sys.modules:
            M=Metadata()         
            if type(inpt)==str:
                M.add_input(inpt)
            M.add_output(loss_plot_path)
            M.add_output(perf_plot_path)
            metadata_path=os.path.join(output_directory, 'plot_autoencoder_results_metadata.json')
            write_metadata(M, metadata_path)
    
    return loss_plot, perf_plot



def plot_geometry_results(task_rec_in, ccgp_rec_in, task_hidden_in, ccgp_hidden_in):
    
    n_files=task_rec_in.shape[0] # assuming same for reconstructed output and hidden layer
    
    # Average across linear classification tasks:
    acc_rec=np.zeros((n_files, 2,2))
    acc_rec[:,1,:]=task_rec_in[:,2,:]
    acc_rec[:,0,:]=np.mean(task_rec_in[:,0:2,:],axis=0) # dim0: n_files; dim1: linear vs XOR; dim2: train vs test
    acc_rec_m=np.mean(acc_rec,axis=0)
    acc_rec_sem=sem(acc_rec,axis=0)
    
    # Average across CCGP tasks:
    ccgp_rec=np.mean(ccgp_rec_in,axis=1)
    ccgp_rec=np.mean(ccgp_rec,axis=1) # dim0: n_files; dim1: train vs test
    ccgp_rec_m=np.mean(ccgp_rec,axis=0)
    ccgp_rec_sem=sem(ccgp_rec,axis=0)

    # Average across linear classification tasks:
    acc_hidden=np.zeros((n_files, 2,2))
    acc_hidden[:,1,:]=task_hidden_in[:,2,:]
    acc_hidden[:,0,:]=np.mean(task_hidden_in[:,0:2,:],axis=0) # dim0: n_files; dim1: linear vs XOR; dim2: train vs test
    acc_hidden_m=np.mean(acc_hidden,axis=0)
    acc_hidden_sem=sem(acc_hidden,axis=0)
    
    # Average across CCGP tasks:
    ccgp_hidden=np.mean(ccgp_hidden_in,axis=1)
    ccgp_hidden=np.mean(ccgp_hidden,axis=1) # dim0: n_files; dim1: train vs test
    ccgp_hidden_m=np.mean(ccgp_hidden,axis=0)
    ccgp_hidden_sem=sem(ccgp_hidden,axis=0)
    
    # Define some plotting params:
    width=0.15
    min_alph=0.4
    max_alph=1.0
    alph_step = (max_alph-min_alph)/(3-1)
    alpha_vec=np.arange(min_alph, max_alph+alph_step, alph_step)
    
    # Init axes:
    fig=plt.figure(figsize=(2,2))
    ax=fig.add_subplot(111)

    # Plot geometry results for reconstructed output:
    ax.bar(0*width-1.5*width,acc_rec_m[0,1],yerr=acc_rec_sem[0,1],color='blue',width=width,alpha=alpha_vec[0]) # plot linear performance
    ax.bar(1*width-1.5*width,acc_rec_m[1,1],yerr=acc_rec_sem[1,1],color='blue',width=width,alpha=alpha_vec[1]) # plot XOR performance
    ax.bar(2*width-1.5*width,ccgp_rec_m[1],yerr=ccgp_rec_sem[1],color='blue',width=width,alpha=alpha_vec[2]) # plot CCGP

    # Plot geometry results for hidden layer representation:
    ax.bar(4*width-1.5*width,acc_hidden_m[0,1],yerr=acc_hidden_sem[0,1],color='red',width=width,alpha=alpha_vec[0]) # plot linear performance
    ax.bar(5*width-1.5*width,acc_hidden_m[1,1],yerr=acc_hidden_sem[1,1],color='red',width=width,alpha=alpha_vec[1]) # plot XOR performance
    ax.bar(6*width-1.5*width,ccgp_hidden_m[1],yerr=ccgp_hidden_sem[1],color='red',width=width,alpha=alpha_vec[2]) # plot CCGP    

