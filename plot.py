# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:01:05 2023

@author: danie
"""

import sys
import os
import pathlib
import numpy as np
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
        file.close()
    elif type(inpt)==dict:
        loss_epochs=inpt['loss_epochs']
        perf_hidden=inpt['perf_hidden']
        perf_orig=inpt['perf_orig']
        perf_out=inpt['perf_out']
        
    n_epochs=loss_epochs.shape[1]
    
    # Plot Loss
    loss_plot, ax = plt.subplots(1)
    loss_m=np.mean(loss_epochs,axis=0)
    plt.plot(loss_m)
    plt.ylabel('Training Loss')
    plt.xlabel('Epochs')
    plt.show()
    
    # Plot performance
    perf_plot, ax = plt.subplots(1)
    perf_m=np.mean(perf_orig,axis=0)
    perf_out_m=np.mean(perf_out,axis=0)
    perf_hidden_m=np.mean(perf_hidden,axis=0)
    
    plt.plot(perf_out_m[:,1],color='blue',label='Out')
    plt.plot(perf_hidden_m[:,1],color='red',label='Hidden')
    plt.plot(perf_m[1]*np.ones(n_epochs),color='grey',label='Input')
    
    if plot_train:
        plt.plot(perf_out_m[:,0],color='blue',linestyle='--', label='Out train')
        plt.plot(perf_hidden_m[:,0],color='red',linestyle='--', label='Hidden train')
        plt.plot(perf_m[0]*np.ones(n_epochs),color='grey',linestyle='--', label='Input train')
    
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