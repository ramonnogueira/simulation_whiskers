# -*- coding: utf-8 -*-
"""
Plot task performance vs reconstruction loss. 

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
input_path = 'E:\\simulation_whiskers\\results\\run651\\ae_iterate_beta_reconstruction.pickle'

# Define independent variable:
X = 'beta_rec'
#X = lambda x : x.beta0 + x.beta1

# Plotting parameters:
#xscale = 'linear'
xscale = 'log'
    
#ccgp_yl = None
#par_yl = None
ccgp_yl = [0.45, 1.0]
par_yl = [0.5, 1.0]

xlim = None
#xlim = [-200, 5000]

ind_var_lbl = None
# ind_var_lbl = r'$\beta_{0} + \beta_{1}$'


# Define custom filter if desired:
#flt = lambda x : x.n_hidden==160
flt = lambda x : x.task=='xor'
#flt = None


# Define plotting parameters:
cmap = plt.get_cmap('Set1')
base_colors = [cmap(2), cmap(1), cmap(0)] # Cycle through green, blue, red for different tasks
color_var = 'n_hidden'
    

# Output parameters:
save_output = False



#%% Preliminary stuff:

# Load results:
results = pickle.load(open(input_path, 'rb'))
perf_df = results['perf_df']

# Apply any filters if requested:
if flt is not None:
    keep = perf_df.apply(flt, axis=1)
    perf_df = perf_df[keep]

# Compute independent variable if necessary:
if callable(X):
    ind_var = perf_df.apply(X, axis=1)
    perf_df['ind_var'] = ind_var
    if ind_var_lbl is None:
        ind_var_lbl = inspect.getsource(X)
elif type(X) == str:
    perf_df['ind_var'] = perf_df[X]
    ind_var_lbl = X
    # Hack; replace 'reconstruction' with 'prediction' if applicable
    if ind_var_lbl=='beta_rec' and 'model_type' in perf_df and np.all(perf_df.model_type=='prediction'):
        ind_var_lbl = 'beta_pred'

if xscale == 'log':
    ind_var_lbl = 'log({})'.format(ind_var_lbl)

# Select layer-specific results:
perf_df_hidden = perf_df[perf_df.layer=='hidden'] 
perf_df_input = perf_df[perf_df.layer=='input'] 

# Get number of subsamples/repeats:
n_subsamples = len(np.unique(perf_df_hidden.subsample))
n_repeats = len(np.unique(perf_df_hidden.repeat))
resamples_lines = 'n subsamples = {}, n repeats = {}'.format(n_subsamples, n_repeats)

if color_var == 'beta_sp':
    color_order = 'forward'
else:
    color_order = 'reverse'

# Define utility function:
def grp(df):
    B = df[['task', 'n_hidden', 'beta_sp', 'repeat', 'ind_var', 'test']]\
        .groupby(['task', 'n_hidden', 'beta_sp', 'repeat', 'ind_var']).mean()\
        .groupby(['task', 'n_hidden', 'beta_sp','ind_var'])    
    return B


#%% Task performance analysis:

# Compute parallelism means:
B_hidden = grp(
    perf_df_hidden)
Mu = B_hidden.mean().reset_index()
Mu = Mu.rename(columns={'test':'mu'})

Std = B_hidden.std().reset_index()
Std = Std.rename(columns={'test':'stdev'})

C = pd.merge(Mu, Std, on=['task', 'n_hidden', 'beta_sp', 'ind_var'], how='outer')

if xscale == 'log':
    C['ind_var'] = Mu.apply(lambda x : np.log10(x.ind_var) if x.ind_var>0 else x.ind_var, axis=1)

# Compute mean input:
B_input = grp(perf_df_input)
C_input = B_input.mean().reset_index()
C_input = C_input.rename(columns={'test':'mu'})


# Plot task performance:
perf_fig, ax = plt.subplots(figsize=(6,6))
for t, task in enumerate(np.unique(perf_df.task)):

    curr_task_rows = C[C.task==task]
    
    # Get curves (n_hidden-beta_sp combos) to plot for current task:
    curr_curves = curr_task_rows.loc[:,curr_task_rows.columns[:-1]][['n_hidden', 'beta_sp']].drop_duplicates()
    curr_curves = curr_curves.reset_index()
    
    # Compute colors to plot current curves in:
    curr_base_color = np.array(base_colors[t][:-1]) # exclude final color coordinate (alpha)
    mx = np.max(curr_task_rows[color_var])
    if curr_curves.shape[0] > 1:
        compute_shade = lambda x : curr_base_color + (x/mx)*0.75*(np.ones(3) - curr_base_color)
        curr_colors = list(map(compute_shade, curr_curves[color_var]))
    else:
        curr_colors = [curr_base_color]
    if color_order == 'forward':        
        curr_colors = np.flip(curr_colors)
    
    # Iterate over curves for current task:
    for i, curve in curr_curves.iterrows():
        curr_curve_rows = curr_task_rows[np.array(curr_task_rows.n_hidden==curve.n_hidden) & np.array(curr_task_rows.beta_sp==curve.beta_sp)]

        # Define label for current curves:
        curr_label = task
        if np.ptp(perf_df.n_hidden) > 0:
            curr_label += ', n hidden={}'.format(curve.n_hidden)
        if np.ptp(perf_df.beta_sp) > 0:
            curr_label += ', beta_sp={}'.format(curve.n_hidden)
            
        # Plot curves:
        p = plt.plot(curr_curve_rows['ind_var'], curr_curve_rows.mu, color=curr_colors[i])
        plt.errorbar(curr_curve_rows['ind_var'], curr_curve_rows.mu, yerr=curr_curve_rows.stdev, color=curr_colors[i], label=curr_label)

    # Plot input for reference:
    curr_input_rows = C_input[C_input.task==task]
    plt.axhline(y=np.mean(curr_input_rows.mu), color=p[0].get_color(), linewidth=0.75, linestyle='--', label='{} input'.format(task))


main_title_line = 'Task performance vs {}'.format(ind_var_lbl)
title = '\n'.join([main_title_line, resamples_lines])

if np.ptp(perf_df.n_hidden) == 0:
    title += '\nn_hidden = {}'.format(perf_df.iloc[0].n_hidden)
    
if np.ptp(perf_df.beta_sp) == 0:
    title += ', beta_sp = {}'.format(perf_df.iloc[0].beta_sp)

if np.ptp(perf_df.penalty) == 0:
    title += ', L{}'.format(perf_df.iloc[0].penalty)

plt.title(title)
plt.ylabel('Task performance')
plt.xlabel(ind_var_lbl)
plt.legend(frameon=False)
plt.legend(frameon=False)

if xlim is not None:
    plt.xlim(xlim)

if par_yl is not None:
    plt.ylim(par_yl)

annotation_str = ''
if np.ptp(perf_df.beta_xor) == 0:
    annotation_str += r'$\beta_{\text{XOR}} = $' + '{}'.format(perf_df.iloc[0].beta_xor) + '\n'
    
if np.ptp(perf_df.beta_rec) == 0:
    annotation_str += r'$\beta_{\text{rec}} = $' + '{}'.format(perf_df.iloc[0].beta_rec) + '\n'
    
if np.ptp(perf_df.beta0) == 0:
    annotation_str += r'$\beta_{0} = $' + '{}'.format(perf_df.iloc[0].beta0) + '\n'

if np.ptp(perf_df.beta1) == 0:
    annotation_str += r'$\beta_{1} = $' + '{}'.format(perf_df.iloc[0].beta1) + '\n'

plt.annotate(annotation_str, [0.2, 0.6], xycoords='figure fraction')
plt.tight_layout()



#%% Save output if requested:
    
if save_output:
    
    # Define current output directory:
    input_dir = os.path.split(input_path)[0]
    if 'analysis_metadata' in sys.modules:
        curr_output_dir = increment_dir_name(input_dir, 'task_perf_vs_beta')
    else: 
        curr_output_dir = os.path.join(input_dir, 'task_perf_vs_beta')
    
    # Create current output directory if necessary:
    if not os.path.exists(curr_output_dir):
        pathlib.Path(curr_output_dir).mkdir(parents=True,exist_ok=True)
    
    # Save figures:    
    perf_fig_basename = 'task_perf_v_beta'
    if xscale == 'log':
        perf_fig_basename += '_log'
    plt.figure(perf_fig)
    perf_fig_path = os.path.join(curr_output_dir, perf_fig_basename+'.png')
    plt.savefig(perf_fig_path)
    
    # Save metadata:
    if 'analysis_metadata' in sys.modules:
        
        M = Metadata()
        M.add_input(input_path)
        M.add_output(perf_fig_path)
        metadata_path = os.path.join(curr_output_dir, 'geometry_vs_beta.json')
        write_metadata(M, metadata_path)