# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:24:45 2024

@author: danie
"""

import sys
import os
import pathlib
import numpy as np
from numpy import matlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from simulation_whiskers.simulate_task import load_sim_params, simulate_session, session2feature_array, session2labels, load_task_def
from simulation_whiskers.general import proj_code_plane
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, increment_dir_name
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False

# Define path to, load simulation parameters:
#sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv.json'    
#sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv2.json'


# Define parameters, tasks:
"""
# Near/far vs. rough/smooth
sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv3.json'
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\freq_sh.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\time_mov.json'
"""

#"""
# Near/far vs. convex/concave
sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv2.json'
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\convex_concave.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\time_mov.json'
#"""

"""
# Convex/concave vs. rough/smooth
sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv4.json'
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\freq_sh.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\convex_concave.json'
"""

"""
# Convex/concave vs. curvature
sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv5.json'
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\flat_v_curved.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\convex_concave.json'
"""

"""
# Rough/smooth vs. curvature
sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv6.json'
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\flat_v_curved.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\freq_sh.json'
"""

# Define general options:
zscore_data = True

# Define decoder options:
clf_type = 'LogisticRegression'
sum_bins = False

# Define output parameters:
save_output=False
base_output_directory='E:\\simulation_whiskers\\results\\proj_code_plane'
run_base_name='run'


#%% Deal with preliminaries (loading simulation params, etc.):
    
sim_params=load_sim_params(sim_params_path)

"""
base_task=load_task_def(task0_def_path)
#base_task=load_task_def(task1_def_path)
proj_task=load_task_def(task1_def_path)
#proj_task=load_task_def(task0_def_path)
"""

task0=load_task_def(task0_def_path)
#base_task=load_task_def(task1_def_path)
task1=load_task_def(task1_def_path)
#proj_task=load_task_def(task0_def_path)



#%% Run whisker simulation:

session = simulate_session(sim_params, save_output=False, sum_bins=sum_bins)

# Extract features for current session:
X = session2feature_array(session, field='features')

# Compute labels for base and projection tasks:
base_labels = session2labels(session, task0)
proj_labels = session2labels(session, task1)



#%% Project input space representation onto coding axes:
    
xhat, yhat, angle = proj_code_plane(sim_params, base_labels=base_labels, 
    proj_labels=proj_labels, zscore_data=zscore_data, classifier=clf_type, 
    sum_bins=sum_bins, plot_coding_axes=True, save_output=False)


#%% Save figure if requested:

if save_output:
    
    # Define new output directory:
    if 'analysis_metadata' in sys.modules:
        curr_output_directory=increment_dir_name(base_output_directory, run_base_name)
    else:
        curr_output_directory=base_output_directory
        
    # Create output directory if necessary:
    if not os.path.exists(curr_output_directory):
        pathlib.Path(curr_output_directory).mkdir(parents=True, exist_ok=True)

    png_path = os.path.join(curr_output_directory, 'proj_inpt_code_plane.png')
    plt.savefig(png_path)
        
    svg_path = os.path.join(curr_output_directory, 'proj_inpt_code_plane.svg')
    plt.savefig(svg_path)
        
    # Define and save metadata:
    if 'analysis_metadata' in sys.modules:
        M=Metadata()
        params=dict()
        params['base_task']=task0
        params['proj_task']=task1
        params['classifier']=clf_type
        params['sum_bins']=sum_bins
        params['zscore_data']=zscore_data
        M.parameters=params
        M.add_output(png_path)
        M.add_output(svg_path)
        metadata_path = os.path.join(curr_output_directory, 'proj_code_plane_metadata.json')
        write_metadata(M, metadata_path)   