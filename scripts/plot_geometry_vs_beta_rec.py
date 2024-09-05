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
import inspect
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, increment_dir_name, seconds_2_full_time_str
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False

# Input parameters:
input_path = 'C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\results\\run609\\ae_iterate_beta_reconstruction.pickle'

# Define independent variable:
X = 'beta_rec'
#X = lambda x : x.beta0 + x.beta1

# Plotting parameters:
#ccgp_yl = None
#par_yl = None
ccgp_yl = [0.45, 1.0]
par_yl = [-0.2, 1.0]
ind_var_lbl = None
# ind_var_lbl = r'$\beta_{0} + \beta_{1}$'


# Output parameters:
save_output = False



###############################################################################
#Preliminary stuff:

# Load results:
results = pickle.load(open(input_path, 'rb'))
geo_df = results['geo_df']

# Compute independent variable if necessary:
if callable(X):
    ind_var = geo_df.apply(X, axis=1)
    geo_df['ind_var'] = ind_var
    if ind_var_lbl is None:
        ind_var_lbl = inspect.getsource(X)
elif type(X) == str:
    geo_df['ind_var'] = geo_df[X]
    ind_var_lbl = X

# Select layer-specific results:
geo_df_hidden = geo_df[geo_df.layer=='hidden'] 
geo_df_input = geo_df[geo_df.layer=='input'] 
    
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



###############################################################################
# CCGP analysis:

# Compute CCGP means:
Mu = geo_df_hidden[['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'repeat', 'subsample', 'beta_sp', 'ind_var', 'train_accuracy', 'test_accuracy']]\
    .groupby(['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'dichotomy_idx', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'beta_sp', 'ind_var']).mean().reset_index()


# Compute CCGP standard deviations:
Std = geo_df_hidden[['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'repeat', 'subsample', 'beta_sp', 'ind_var', 'train_accuracy', 'test_accuracy']]\
    .groupby(['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'dichotomy_idx', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'beta_sp', 'ind_var']).std().reset_index()

# Compute input CCGP for reference:
Mu_input = geo_df_input[['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'repeat', 'subsample', 'beta_sp', 'ind_var', 'train_accuracy', 'test_accuracy']]\
    .groupby(['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'dichotomy_idx', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'beta_sp', 'ind_var']).mean()\
    .groupby(['ind_var']).mean().reset_index()
Mu_input = Mu_input.mean(axis=0)

configs = Mu[['n_hidden', 'beta_sp']].drop_duplicates()

# Plot CCGP:
ccgp_fig, ax = plt.subplots(figsize=(6,6))
for i, cfg in configs.iterrows():
    
    is_hls = np.array(Mu.n_hidden == cfg.n_hidden)
    is_sp = np.array(Mu.beta_sp == cfg.beta_sp)
    curr_mu = Mu[is_hls & is_sp]
    curr_std = Std[is_hls & is_sp]
    
    label = 'm={}, beta_sp={}'.format(cfg.n_hidden, cfg.beta_sp)
    plt.errorbar(curr_mu['ind_var'], curr_mu.test_accuracy, yerr=curr_std.test_accuracy, label=label)

# Plot input CCGP for reference:
plt.axhline(y=Mu_input.test_accuracy, color='gray', linewidth=0.75, linestyle='--', label='input')


main_title_line = 'CCGP vs reconstruction weight'
title = '\n'.join([main_title_line, dichotomy_lines, resamples_lines])
plt.title(title)
plt.ylabel('CCGP')
plt.xlabel(ind_var_lbl)
plt.xscale('log')
plt.legend(frameon=False)

if ccgp_yl is not None:
    plt.ylim(ccgp_yl)

annotation_str = ''
if np.ptp(geo_df.beta_xor) == 0:
    annotation_str += r'$\beta_{\text{XOR}} = $' + '{}'.format(geo_df.iloc[0].beta_xor) + '\n'

if np.ptp(geo_df.beta_rec) == 0:
    annotation_str += r'$\beta_{\text{rec}} = $' + '{}'.format(geo_df.iloc[0].beta_rec) + '\n'
    
if np.ptp(geo_df.beta0) == 0:
    annotation_str += r'$\beta_{0} = $' + '{}'.format(geo_df.iloc[0].beta0) + '\n'

if np.ptp(geo_df.beta1) == 0:
    annotation_str += r'$\beta_{1} = $' + '{}'.format(geo_df.iloc[0].beta1) + '\n'

plt.annotate(annotation_str, [0.2, 0.6], xycoords='figure fraction')
plt.tight_layout()




###############################################################################
# Parallelism analysis:
    
# Compute parallelism means:
Mu = geo_df_hidden[['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'repeat', 'subsample', 'beta_sp', 'ind_var', 'parallelism']]\
    .groupby(['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'dichotomy_idx', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'beta_sp', 'ind_var']).mean().reset_index()


# Compute parallelism standard deviations:
Std = geo_df_hidden[['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'repeat', 'subsample', 'beta_sp', 'ind_var', 'parallelism']]\
    .groupby(['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'dichotomy_idx', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'beta_sp', 'ind_var']).std().reset_index()

# Compute input parallelism for reference:
Mu_input = geo_df_input[['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'repeat', 'subsample', 'beta_sp', 'ind_var', 'parallelism']]\
    .groupby(['n_hidden', 'dichotomy_idx', 'train_partition_inds', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'dichotomy_idx', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'repeat', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'subsample', 'beta_sp', 'ind_var']).mean()\
    .groupby(['n_hidden', 'beta_sp', 'ind_var']).mean()\
    .groupby(['ind_var']).mean().reset_index()
Mu_input = Mu_input.mean(axis=0)

configs = Mu[['n_hidden', 'beta_sp']].drop_duplicates()

# Plot parallelism:
parallelism_fig, ax = plt.subplots(figsize=(6,6))
for i, cfg in configs.iterrows():
    
    is_hls = np.array(Mu.n_hidden == cfg.n_hidden)  
    is_sp = np.array(Mu.beta_sp == cfg.beta_sp)
    curr_mu = Mu[is_hls & is_sp]
    curr_std = Std[is_hls & is_sp]
    
    label = 'm={}, beta_sp={}'.format(cfg.n_hidden, cfg.beta_sp)
    plt.errorbar(curr_mu['ind_var'], curr_mu.parallelism, yerr=curr_std.parallelism, label=label)

# Plot input parallelism for reference:
plt.axhline(y=Mu_input.parallelism, color='gray', linewidth=0.75, linestyle='--', label='input')


main_title_line = 'Parallelism vs hidden layer size'
title = '\n'.join([main_title_line, dichotomy_lines, resamples_lines])
plt.title(title)
plt.ylabel('Parallelism score')
plt.xlabel(ind_var_lbl)
plt.xscale('log')
plt.legend(frameon=False)

if par_yl is not None:
    plt.ylim(par_yl)

annotation_str = ''
if np.ptp(geo_df.beta_xor) == 0:
    annotation_str += r'$\beta_{\text{XOR}} = $' + '{}'.format(geo_df.iloc[0].beta_xor) + '\n'
    
if np.ptp(geo_df.beta_rec) == 0:
    annotation_str += r'$\beta_{\text{rec}} = $' + '{}'.format(geo_df.iloc[0].beta_rec) + '\n'
    
if np.ptp(geo_df.beta0) == 0:
    annotation_str += r'$\beta_{0} = $' + '{}'.format(geo_df.iloc[0].beta0) + '\n'

if np.ptp(geo_df.beta1) == 0:
    annotation_str += r'$\beta_{1} = $' + '{}'.format(geo_df.iloc[0].beta1) + '\n'

plt.annotate(annotation_str, [0.2, 0.6], xycoords='figure fraction')
plt.tight_layout()



###############################################################################
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
    plt.figure(ccgp_fig)
    ccgp_fig_path = os.path.join(curr_output_dir, 'ccgp_v_hidden_layer_size_beta_sp0.png')
    plt.savefig(ccgp_fig_path)
    
    plt.figure(parallelism_fig)
    parallelism_fig_path = os.path.join(curr_output_dir, 'parallelism_v_hidden_layer_size_beta_sp0.png')
    plt.savefig(parallelism_fig_path)
    
    # Save metadata:
    if 'analysis_metadata' in sys.modules:
        
        M = Metadata()
        M.add_input(input_path)
        M.add_output(ccgp_fig_path)
        M.add_output(parallelism_fig_path)
        metadata_path = os.path.join(curr_output_dir, 'geometry_vs_hidden_layer_size_metadata_beta_sp0.json')
        write_metadata(M, metadata_path)