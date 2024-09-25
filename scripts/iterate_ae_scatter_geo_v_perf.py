# -*- coding: utf-8 -*-
"""
Plot CCGP/parallelism vs reconstruction loss. 

Created on Wed Jul 17 02:14:59 2024

@author: danie
"""

import sys
import os
import pathlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import inspect
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, increment_dir_name, seconds_2_full_time_str
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False

# Input parameters:
input_path = 'C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\results\\run651\\ae_iterate_beta_reconstruction.pickle'

# Define geometric measure to analyze:
Y = 'parallelism'

# Define task to measure performmance for on x-axis:
task = 'xor'

# Define additional custom filter:
flt = lambda x : round(x.beta_rec) == round(10**4.5)

#X = lambda x : x.beta0 + x.beta1

# Plotting parameters:
xscale = 'log' # 'linear' | 'log'
base_color = np.array([0, 0.5, 0])
ccgp_yl = [0.45, 1.0]
par_yl = [-0.4, 1.0]
xlim = None
#xlim = [-200, 5000]

ind_var_lbl = None
# ind_var_lbl = r'$\beta_{0} + \beta_{1}$'


# Output parameters:
save_output = False



#%% Preliminary stuff:

# Load results:
results = pickle.load(open(input_path, 'rb'))
geo_df = results['geo_df']
perf_df = results['perf_df']


# Select layer-specific results:
geo_df_hidden = geo_df[geo_df.layer=='hidden']
perf_df_hidden = perf_df[perf_df.layer=='hidden']


# Select only performance for requested task:
perf_df_hidden = perf_df_hidden[perf_df_hidden.task==task]


# Apply any additional filters:
geo_df_hidden = geo_df_hidden[geo_df_hidden.apply(flt, axis=1)]
perf_df_hidden = perf_df_hidden[perf_df_hidden.apply(flt, axis=1)]
    

# Compute independent variable if necessary:
if callable(X):
    ind_var_geo = geo_df.apply(X, axis=1)
    geo_df_hidden['ind_var'] = ind_var_geo
    
    ind_var_perf = perf_df.apply(X, axis=1)
    perf_df_hidden['ind_var'] = ind_var_perf
    
    if ind_var_lbl is None:
        ind_var_lbl = inspect.getsource(X)
        
elif type(X) == str:
    geo_df_hidden['ind_var'] = geo_df[X]
    perf_df_hidden['ind_var'] = perf_df[X]
    ind_var_lbl = X
    # Hack; replace 'reconstruction' with 'prediction' if applicable
    if ind_var_lbl=='beta_rec' and 'model_type' in geo_df and np.all(geo_df.model_type=='prediction'):
        ind_var_lbl = 'beta_pred'
        

# Get dichotomies:
dichotomy_strs = np.unique(geo_df_hidden.dichotomy)
dichotomy_lines = []
for i, d in enumerate(dichotomy_strs):
    curr_dichotomy_str = 'Task {} : {}'.format(i, d)
    dichotomy_lines.append(curr_dichotomy_str)
dichotomy_lines = '\n'.join(dichotomy_lines)


# Get number of subsamples/repeats:
n_subsamples = len(np.unique(geo_df_hidden.subsample))
n_repeats = len(np.unique(geo_df_hidden.repeat))
resamples_lines = 'n subsamples = {}, n repeats = {}'.format(n_subsamples, n_repeats)


# Try to get penalty:
if 'penalty' not in geo_df:
    inpt_dir = os.path.split(input_path)[0]
    metadata_path = os.path.join(inpt_dir, 'ae_iterate_hidden_size_metadata.json')
    inpt_metadata = json.load(open(metadata_path, 'rb'))
    try:
        penalty = inpt_metadata['parameters']['autoencoder_params']['p_norm']
    except KeyError:
        penalty = None
else:
    if len(np.unique(geo_df.penalty)) == 1:
        penalty = geo_df.iloc[0].penalty
    else:
        penalty = None
       
        
# Auto-assign dependent variable names:
if Y == 'CCGP':
    Y_str = 'test_accuracy'
elif Y == 'parallelism':
    Y_str = Y

if xscale == 'log':
    ind_var_lbl = 'log({})'.format(ind_var_lbl)       
    
        
# Define utility functions:
def grp_geo(df, dep_var):
    B = df[['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'repeat', 'subsample', 'beta_sp', 'ind_var', dep_var]]\
        .groupby(['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
        .groupby(['n_hidden', 'dichotomy_idx', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
        .groupby(['n_hidden', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
        .groupby(['n_hidden', 'subsample', 'beta_sp', 'ind_var']).mean()\
        .groupby(['n_hidden', 'beta_sp', 'ind_var']).mean().reset_index()
    return B

def grp_perf(df, dep_var):
    B = df[['n_hidden', 'task', 'repeat', 'subsample', 'beta_sp', 'ind_var', 'test']]\
        .groupby(['n_hidden', 'task', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
        .groupby(['n_hidden', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
        .groupby(['n_hidden', 'subsample', 'beta_sp', 'ind_var']).mean()\
        .groupby(['n_hidden', 'beta_sp', 'ind_var']).mean().reset_index()
    return B


#%% Compute average geometry metric:

Mu_geo = grp_geo(geo_df_hidden, Y_str)

if xscale == 'log':
    Mu_geo['ind_var'] = Mu_geo.apply(lambda x : np.log10(x.ind_var) if x.ind_var>0 else x.ind_var, axis=1)

configs_geo = Mu_geo[['n_hidden', 'beta_sp']].drop_duplicates()
configs_geo = configs_geo.reset_index()



#%% Compute task performance metrics:

Mu_perf = grp_perf(perf_df_hidden, Y_str)    

if xscale == 'log':
    Mu_perf['ind_var'] = Mu_perf.apply(lambda x : np.log10(x.ind_var) if x.ind_var>0 else x.ind_var, axis=1)

configs_perf = Mu_perf[['n_hidden', 'beta_sp']].drop_duplicates()
configs_perf = configs_perf.reset_index()



#%% Plot:
    
# Merge performance and geometry metrics:
Mu = pd.merge(Mu_geo, Mu_perf, on=['n_hidden', 'beta_sp', 'ind_var'], how='inner')

# Apply any additional filters:
Mu = Mu[Mu.apply(flt, axis=1)]




#%%
# Save output if requested:
    
if save_output:
    
    # Define current output directory:
    input_dir = os.path.split(input_path)[0]
    if 'analysis_metadata' in sys.modules:
        curr_output_dir = increment_dir_name(input_dir, 'geometry_vs_hidden_layer_size')
    else: 
        curr_output_dir = os.path.join(input_dir, 'geometry_vs_hidden_layer_size')
    
    # Create current output directory if necessary:
    if not os.path.exists(curr_output_dir):
        pathlib.Path(curr_output_dir).mkdir(parents=True,exist_ok=True)
    
    # Save figures:
    ccgp_fig_basename = 'ccgp_v_beta'
    if xscale == 'log':
        ccgp_fig_basename += '_log'
    plt.figure(ccgp_fig)
    ccgp_fig_path = os.path.join(curr_output_dir, ccgp_fig_basename+'.png')
    plt.savefig(ccgp_fig_path)
    
    par_fig_basename = 'parallelism_v_beta'
    if xscale == 'log':
        par_fig_basename += '_log'
    plt.figure(parallelism_fig)
    parallelism_fig_path = os.path.join(curr_output_dir, par_fig_basename+'.png')
    plt.savefig(parallelism_fig_path)
    
    # Save metadata:
    if 'analysis_metadata' in sys.modules:
        
        M = Metadata()
        M.add_input(input_path)
        M.add_output(ccgp_fig_path)
        M.add_output(parallelism_fig_path)
        metadata_path = os.path.join(curr_output_dir, 'geometry_vs_beta.json')
        write_metadata(M, metadata_path)