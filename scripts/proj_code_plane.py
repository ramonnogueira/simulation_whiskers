# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:24:45 2024

@author: danie
"""

import numpy as np
from numpy import matlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from simulation_whiskers.simulate_task import load_sim_params, simulate_session, session2feature_array, session2labels, load_task_def
from simulation_whiskers.general import proj_code_plane
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from analysis_metadata.analysis_metadata import increment_dir_name

# Define path to, load simulation parameters:
#sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv.json'    
#sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv2.json'


# Define tasks:
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

# Define general options:
zscore_data = True

# Define decoder options:
clf_type = 'LogisticRegression'
sum_bins = False

# Define where to save output:
save_output=False
base_output_directory='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\results\\proj_code_plane'
run_base_name='run'

# Define new output directory:
curr_output_directory=increment_dir_name(base_output_directory, run_base_name)

xhat, yhat, angle = proj_code_plane(sim_params, base_task=task0, proj_task=task1, zscore_data=zscore_data, classifier=clf_type, sum_bins=sum_bins, save_output=save_output, output_directory=curr_output_directory)