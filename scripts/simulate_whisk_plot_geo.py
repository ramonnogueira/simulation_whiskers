# -*- coding: utf-8 -*-
"""
Simulate whisker data and plot input space task performance, CCGP, and 
parallelism score.

Created on Mon Sep 30 01:57:15 2024

@author: danie
"""

import sys
import os
import pathlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import inspect
from simulation_whiskers.miscellaneous_sparseauto import iterate_fit_autoencoder, fmt_ae_metadata
from simulation_whiskers.simulate_task import load_sim_params, load_task_def
from simulation_whiskers.plot import plot_ccgps_by_layer, plot_pars_by_layer
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, increment_dir_name, seconds_2_full_time_str
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False


# Define simulation parameters:
sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv2.json'
n_repeats = 1
sum_bins = False    

# Define tasks:
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\convex_concave.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\time_mov.json'

# Define output parameters:
save_output = False
base_output_directory = 'E:\\simulation_whiskers\\results\\input_geometry'
run_base_name = 'run'

    
#%% Simulate whikser data:

# Load simulation parameters, tasks:
sim_params=load_sim_params(sim_params_path)
task0=load_task_def(task0_def_path)
task1=load_task_def(task1_def_path)
tasks = [task0, task1]

results = iterate_fit_autoencoder(sim_params, tasks, n_files=n_repeats, sum_inpt=sum_bins, test_geometry=True, n_geo_subsamples=1)


#%% Plot task performance, CCGP, and parallelism score:

geo_df = results['geo_df']
perf_df = results['perf_df']

ccgp_fig = plot_ccgps_by_layer(perf_df, geo_df)
pars_fig = plot_pars_by_layer(geo_df)


#%% Save output if requested:
    
if save_output:
    
    # Define, create output directory:
    if not os.path.exists(base_output_directory):
        pathlib.Path(base_output_directory).mkdir(parents=True, exist_ok=True)    
    
    if 'analysis_metadata' in sys.modules:
        curr_output_directory=increment_dir_name(base_output_directory, run_base_name)
    else:
        curr_output_directory=base_output_directory
        
    if not os.path.exists(curr_output_directory):
        pathlib.Path(curr_output_directory).mkdir(parents=True, exist_ok=True)
        
    plt.figure(ccgp_fig)
    ccgp_fig_basename = 'input_ccgp'
    ccgp_png_path = os.path.join(curr_output_directory, ccgp_fig_basename+'.png')
    plt.savefig(ccgp_png_path)
    ccgp_svg_path = os.path.join(curr_output_directory, ccgp_fig_basename+'.svg')
    plt.savefig(ccgp_svg_path)       
    
    plt.figure(pars_fig)
    pars_fig_basename = 'input_ccgp'
    pars_png_path = os.path.join(curr_output_directory, pars_fig_basename+'.png')
    plt.savefig(pars_png_path)
    pars_svg_path = os.path.join(curr_output_directory, pars_fig_basename+'.svg')
    plt.savefig(pars_svg_path)           
    
    N = fmt_ae_metadata(sim_params, autoencoder_params=None)
    
    # Save metadata:
    if 'analysis_metadata' in sys.modules:
        M = Metadata()
        M.add_param('simulation_parameters', N.parameters['sim_params'])
        M.add_param('task0', task0)
        M.add_param('task1', task1)
        M.add_output(ccgp_png_path)
        M.add_output(ccgp_svg_path)
        M.add_output(pars_png_path)
        M.add_output(pars_svg_path)
        metadata_path = os.path.join(curr_output_directory, 'input_geometry.json')
        write_metadata(M, metadata_path)