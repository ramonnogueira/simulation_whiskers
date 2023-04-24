# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:19:40 2023

@author: danie
"""
import numpy as np
import json
import itertools
from simulation_whiskers.simulate_task import load_sim_params, load_task_def
from simulation_whiskers.miscellaneous_sparseauto import iterate_fit_autoencoder
from simulation_whiskers.plot import plot_iterate_autoencoder_results, plot_autoencoder_geometry
from analysis_metadata.analysis_metadata import increment_dir_name


# Define paths to parameters files:
sim_params_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\hyperparams\\example_sim_params_laptop.json'
ae_params_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\hyperparams\\example_autoencoder_hparams.json'
task_def_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\task_defs\\convex_concave.json'

# Define general variables:
n_files = 1

# Output directory:
base_output_directory='C:\\Users\\danie\\Documents\\simulation_whiskers\\results\\'
run_base_name='run'
sv=True
    
# Load simulation hyperparameters, task definition:
sim_params=load_sim_params(sim_params_path)
task=load_task_def(task_def_path)
autoencoder_params=json.load(open(ae_params_path,'r'))

# Define a bunch of different hyperparamter combinations to try:
beta=np.arange(0,1.2,0.2)
n_hiddens=[10, 100, 300]
params=[t for t in itertools.product(beta, n_hiddens)]
           
# Iterate over dicts of hyperparamter combos:
for d in params:
    
    # Define new output directory:
    curr_output_directory=increment_dir_name(base_output_directory, run_base_name)
    
    # Write new parameter values to sim_params:
    autoencoder_params['beta']=d[0]
    autoencoder_params['n_hidden']=d[1]
    
    # Fit autoencoder, test classifier performance:
    results=iterate_fit_autoencoder(sim_params, autoencoder_params, task, n_files, save_perf=sv, save_learning=False, save_sessions=False, output_directory=curr_output_directory, verbose=True)
    
    # Plot results: 
    loss_plot, perf_plot=plot_iterate_autoencoder_results(results, save_output=sv, output_directory=curr_output_directory)
    geo_plot=plot_autoencoder_geometry(results['task_hidden'], results['ccgp_hidden'], rec_lr=results['task_rec'], rec_ccgp=results['ccgp_rec'], inpt_lr=results['task_inpt'], inpt_ccgp=results['ccgp_inpt'], plot_train=True, save_output=sv, output_directory=curr_output_directory)