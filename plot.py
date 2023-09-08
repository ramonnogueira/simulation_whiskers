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



def plot_autoencoder_geometry(hidden_lr, hidden_ccgp, hidden_par, rec_lr=None, rec_ccgp=None, rec_par=None, inpt_lr=None, inpt_ccgp=None, inpt_par=None, pre_lr=None, pre_ccgp=None, pre_par=None, plot_train=False, save_output=False, output_directory=None):

    
    # Init:
    fig=plt.figure(figsize=(4,4))
    ax=fig.add_subplot(111)
    offset=0
    
    # Plot geometry of input if requested:
    if inpt_lr is not None and inpt_ccgp is not None:
        plot_ccgp(inpt_lr, inpt_ccgp, color='green', plot_train=plot_train, h_offset=offset, ax=ax)
        offset+=4
    
    # Plot geometry of hidden layer before training if requested:
    if pre_lr is not None and pre_ccgp is not None:
        plot_ccgp(pre_lr, pre_ccgp, color='orange', plot_train=plot_train, h_offset=offset, ax=ax)
        offset+=4
    
    # Plot geometry of hidden layer representation:
    plot_ccgp(hidden_lr, hidden_ccgp, color='red', plot_train=plot_train, h_offset=offset, ax=ax)
    offset+=4    

    # Plot geometry of reconstructed output if requested:
    if rec_lr is not None and rec_ccgp is not None:
        plot_ccgp(rec_lr, rec_ccgp, color='blue', plot_train=plot_train, h_offset=offset, ax=ax)
    
    xl=ax.get_xlim()
    ax.plot([xl[0],xl[1]],0.5*np.ones(2),color='black',linestyle='--')
    ax.set_ylim([0.4,1.0])
    ax.set_ylabel('Decoding Performance')
    
    # Save figure if requested:
    if save_output:
        
        if output_directory==None:
            output_directory=os.getcwd()
        
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
    
        output_path=os.path.join(output_directory, 'autoencoeder_geometry.png')
        fig.savefig(output_path,dpi=500)
        
        if 'analysis_metadata' in sys.modules:
            M=Metadata()         
            M.add_output(output_path)
            metadata_path=os.path.join(output_directory, 'plot_autoencoder_geometry_metadata.json')
            write_metadata(M,metadata_path)
    
    return fig
    
    
    
def plot_ccgps_by_layer(hidden_lr, hidden_ccgp, rec_lr=None, rec_ccgp=None, inpt_lr=None, inpt_ccgp=None, pre_lr=None, pre_ccgp=None, plot_train=False, save_output=False, output_directory=None):
    """
    Plot results of decoder-based geometry analysis (logistic regression, CCGP)
    for autoencoder.

    Parameters
    ----------
    hidden_lr : array-like
        Hidden layer decoder performance. Should be same format as 'task_hidden'
        field of `results` output of iterate_fit_autoencoder().
        
    hidden_ccgp : array-like
        Hidden layer CCGP. Should be same format as 'ccgp_hidden'
        field of `results` output of iterate_fit_autoencoder().
        
    rec_lr : array-like, optional
        Reconstructed output decoder performance. Should be same format as 
        'task_rec' field of `results` output of iterate_fit_autoencoder().
        
    rec_ccgp : array-like, optional
        Reconstructed output CCGP. Should be same format as 'ccgp_rec' field of 
        `results` output of iterate_fit_autoencoder().
        
    inpt_lr : array-like, optional
        Input decoder performance.
        
    inpt_ccgp : TYPE, optional
        Input CCGP performance.
        
    save_output : bool, optional
        Whether to save figures. 
        
    output_directory : str, optional
        Directory where figures should be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.

    """
    
    # Init:
    fig=plt.figure(figsize=(4,4))
    ax=fig.add_subplot(111)
    offset=0
    
    # Plot geometry of input if requested:
    if inpt_lr is not None and inpt_ccgp is not None:
        plot_ccgp(inpt_lr, inpt_ccgp, color='green', plot_train=plot_train, h_offset=offset, ax=ax)
        offset+=7
    
    # Plot geometry of hidden layer before training if requested:
    if pre_lr is not None and pre_ccgp is not None:
        plot_ccgp(pre_lr, pre_ccgp, color='orange', plot_train=plot_train, h_offset=offset, ax=ax)
        offset+=7
    
    # Plot geometry of hidden layer representation:
    plot_ccgp(hidden_lr, hidden_ccgp, color='red', plot_train=plot_train, h_offset=offset, ax=ax)
    offset+=7    

    # Plot geometry of reconstructed output if requested:
    if rec_lr is not None and rec_ccgp is not None:
        plot_ccgp(rec_lr, rec_ccgp, color='blue', plot_train=plot_train, h_offset=offset, ax=ax)
    
    xl=ax.get_xlim()
    ax.plot([xl[0],xl[1]],0.5*np.ones(2),color='black',linestyle='--')
    ax.set_ylim([0.4,1.0])
    ax.set_ylabel('Decoding Performance')
    
    # Save figure if requested:
    if save_output:
        
        if output_directory==None:
            output_directory=os.getcwd()
        
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
    
        output_path=os.path.join(output_directory, 'autoencoeder_ccgp.png')
        fig.savefig(output_path,dpi=500)
        
        if 'analysis_metadata' in sys.modules:
            M=Metadata()         
            M.add_output(output_path)
            metadata_path=os.path.join(output_directory, 'plot_autoencoder_geometry_metadata.json')
            write_metadata(M,metadata_path)
    
    return fig



def plot_ccgp(task_in, ccgp_in, plot_train=False, color='blue', h_offset=0, ax=None):
    """
    Plot results of geometry_2D() function.

    Parameters
    ----------
    task_in : array-like
        n_files-by-3-by-2 array of linear decoder results. Format should be
        same as perf_tasks output of geometry_2D(), except with additional 0th
        dimension for n_files (i.e. repetitions). 
        
    
    ccgp_in : array-like
        n_files-by-2-by-2-by-2-by-2 array of of CCGP results. Format should be 
        same as perf_ccgp output of geometry_2D(), except with additional 0th
        dimension for n_files (i.e. repetitions). 
    
    color : str, optional
        Bar plot base color. The default is 'blue'.
    
    h_offset : int, optional
        Horizontal offset for bars. Useful if calling plot_geometry_results() 
        multiple times on same axes.
    
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axes object to plot bar graph on. If None, creates a new axes object.


    Returns
    -------
    None.

    """
    
    # Initialize axes if necessary:
    if ax==None:
        fig=plt.figure(figsize=(6,6))
        ax=fig.add_subplot(111)
        
    n_files=task_in.shape[0] # assuming same for reconstructed output and hidden layer
    
    # Average across linear classification tasks:
    task0=task_in[:,0,:]
    task0_m=np.mean(task0,axis=0)
    task0_sem=sem(task0,axis=0)

    task1=task_in[:,1,:]
    task1_m=np.mean(task1,axis=0)
    task1_sem=sem(task1,axis=0)
      
    xor_perf=task_in[:,2,:]
    xor_m=np.mean(xor_perf,axis=0)
    xor_sem=sem(xor_perf,axis=0)
    """
    acc=np.zeros((n_files, 2,2))
    acc[:,1,:]=task_in[:,2,:]
    acc[:,0,:]=np.mean(task_in[:,0:2,:],axis=1) # dim0: n_files; dim1: linear vs XOR; dim2: train vs test
    acc_m=np.mean(acc,axis=0)
    acc_sem=sem(acc,axis=0)
    """
    
    # Average across CCGP tasks:
    ccgp0=ccgp_in[:,0,:,:] # ccgp0: n_files-by-2-by-2       
    ccgp0=np.mean(ccgp0,axis=0) # average across n_files; so ccgp0_m is 2-by-2
    ccgp0_m=np.mean(ccgp0,axis=0) # average across different values of non-decoded feature; so ccgp0_m now just 2 elements (train and test)
    ccgp0_sem=sem(ccgp0,axis=0)

    ccgp1=ccgp_in[:,1,:,:] # ccgp0: n_files-by-2-by-2       
    ccgp1=np.mean(ccgp1,axis=0) # average across n_files; so ccgp0_m is 2-by-2
    ccgp1_m=np.mean(ccgp1,axis=0) # average across different values of non-decoded feature; so ccgp0_m now just 2 elements (train and test)
    ccgp1_sem=sem(ccgp1,axis=0)
    
    # Define some plotting params:
    width=1
    min_alph=0.4
    max_alph=1.0
    alph_step = (max_alph-min_alph)/(4-1)
    alpha_vec=np.arange(min_alph, max_alph+alph_step, alph_step)

    # Plot geometry results for reconstructed output:
    
    #ax.bar(0*width-1.5*width+h_offset,acc_m[0,1],yerr=acc_sem[0,1],color=color,width=width,alpha=alpha_vec[0]) # plot linear performance
    ax.bar(0*width-1.5*width+h_offset,task0_m[1],yerr=task0_sem[1],color=color,width=width,alpha=alpha_vec[0]) # plot task 0 performance
    ax.bar(1*width-1.5*width+h_offset,task1_m[1],yerr=task1_sem[1],color=color,width=width,alpha=alpha_vec[0]) # plot task 1 performance
    ax.bar(2*width-1.5*width+h_offset,xor_m[1],yerr=xor_sem[1],color=color,width=width,alpha=alpha_vec[1]) # plot XOR performance
    ax.bar(3*width-1.5*width+h_offset,ccgp0_m[1],yerr=ccgp0_sem[1],color=color,width=width,alpha=alpha_vec[2]) # plot CCGP
    ax.bar(4*width-1.5*width+h_offset,ccgp1_m[1],yerr=ccgp1_sem[1],color=color,width=width,alpha=alpha_vec[2]) # plot CCGP

    if plot_train:
        #ax.scatter(0*width-1.5*width+h_offset,acc_m[0,0],color=color,alpha=alpha_vec[0])
        ax.scatter(0*width-1.5*width+h_offset,task0_m[0],color=color,alpha=alpha_vec[0])
        ax.scatter(1*width-1.5*width+h_offset,task1_m[0],color=color,alpha=alpha_vec[0])
        ax.scatter(2*width-1.5*width+h_offset,xor_m[0],color=color,alpha=alpha_vec[1])
        ax.scatter(3*width-1.5*width+h_offset,ccgp0_m[0],color=color,alpha=alpha_vec[2])
        ax.scatter(4*width-1.5*width+h_offset,ccgp1_m[0],color=color,alpha=alpha_vec[2])
    
    

def plot_pars_by_layer(inpt_par, pre_par, hidden_par, rec_par, save_output=False, output_directory=None):

    # Init:
    fig=plt.figure(figsize=(4,4))
    ax=fig.add_subplot(111)
    offset=0
    
    # Plot parallelism score in input:
    plot_parallelism(inpt_par, color='green', h_offset=offset, ax=ax)
    offset+=3
    
    # Plot parallelism score in hidden layer before training:
    plot_parallelism(pre_par, color='orange', h_offset=offset, ax=ax)
    offset+=3
    
    # Plot parallelism score in hidden layer:
    plot_parallelism(hidden_par, color='red', h_offset=offset, ax=ax)
    offset+=3    

    # Plot geometry of reconstructed output if requested:
    plot_parallelism(rec_par, color='blue', h_offset=offset, ax=ax)
    
    xl=ax.get_xlim()
    #ax.plot([xl[0],xl[1]],0.5*np.ones(2),color='black',linestyle='--')
    ax.set_ylim([0.0,1.0])
    ax.set_ylabel('Parallelism score')
    
    # Save figure if requested:
    if save_output:
        
        if output_directory==None:
            output_directory=os.getcwd()
        
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
    
        output_path=os.path.join(output_directory, 'autoencoeder_parallelism.png')
        fig.savefig(output_path,dpi=500)
        
        if 'analysis_metadata' in sys.modules:
            M=Metadata()         
            M.add_output(output_path)
            metadata_path=os.path.join(output_directory, 'plot_autoencoder_geometry_metadata.json')
            write_metadata(M,metadata_path)
    
    return fig    
    

    
def plot_parallelism(parallelism_in, color='blue', h_offset=0, ax=None):
    
    # Initialize axes if necessary:
    if ax==None:
        fig=plt.figure(figsize=(6,6))
        ax=fig.add_subplot(111)
    
    # Average paralellism scores across files:
    par0=parallelism_in[:,0]
    par0_m=np.mean(par0)
    par0_m=np.abs(par0_m)
    par0_sem=sem(par0)
    
    par1=parallelism_in[:,1]
    par1_m=np.mean(par1)
    par1_m=np.abs(par1_m)
    par1_sem=sem(par1)

    # Define some plotting params:
    width=1
    
    # Plot parallelism scores
    ax.bar(0*width-1.5*width+h_offset,par0_m,yerr=par0_sem,color=color,width=width) # plot parallelism
    ax.bar(1*width-1.5*width+h_offset,par1_m,yerr=par1_sem,color=color,width=width) # plot parallelism