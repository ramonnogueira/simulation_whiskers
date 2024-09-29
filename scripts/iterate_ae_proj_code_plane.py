# -*- coding: utf-8 -*-
"""
Project autoencoder/prediction model hidden layer representations onto plane
defined by coding axes for pair of tasks used to train model. 

Created on Sun Sep 29 18:06:25 2024

@author: danie
"""
import sys
import os
import pathlib 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from simulation_whiskers.general import proj_code_plane
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, increment_dir_name, seconds_2_full_time_str
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False

# Define inputs:
input_path = 'E:\\simulation_whiskers\\results\\run681\\ae_iterate_beta_reconstruction.pickle'

# Define custom filters:
flt = lambda x : x.n_hidden==160 and round(x.beta_rec, ndigits=2) == round(10**4.5, ndigits=2)

# Define analysis parameters:
clf_type = 'LogisticRegression'
sum_bins = False
zscore_data = False

# Define output parameters:
save_output = False


#%% Preliminaries:

# Load previously saved model outputs:    
results = pickle.load(open(input_path, 'rb'))
ae_df = results['ae_df']    

# Apply custom filters:
ae_df = ae_df[ae_df.apply(flt, axis=1)]
    
# Get task names:
geo_df = results['geo_df']
tasks = np.unique(geo_df.dichotomy)
if len(tasks) == 2:

    # Get name of task coding plane will be defined around (task0 by convention):    
    task0_df = geo_df[geo_df.dichotomy_idx==0]
    if len(np.unique(task0_df.dichotomy))==1:
        base_name = task0_df.iloc[0].dichotomy
        field_names = [x.split(':')[0][2:-1] for x in base_name.split(' vs ')]
        if len(np.unique(field_names)) == 1:
            base_name = field_names[0]
    else:
        raise AssertionError('More than one task name associated with task index 0.')

    # Get name of task to be projected onto coding plane (task1 by convention):    
    task1_df = geo_df[geo_df.dichotomy_idx==1]
    if len(np.unique(task1_df.dichotomy))==1:
        proj_name = task1_df.iloc[0].dichotomy
        field_names = [x.split(':')[0][2:-1] for x in base_name.split(' vs ')]
        if len(np.unique(field_names)) == 1:
            proj_name = field_names[0]
    else:
        raise AssertionError('More than one task name associated with task index 1.')
    
else:
    base_name = None
    proj_name = None
    
# Try to get number of training epochs:
inpt_metadata_path = os.path.join(os.path.split(input_path)[0], 'ae_iterate_hidden_size_metadata.json')
try:
    saved_metadata = json.load(open(inpt_metadata_path,'rb'))
    n_epochs = saved_metadata['parameters']['autoencoder_params']['n_epochs']
except FileNotFoundError():
    n_epochs = None
except KeyError():
    n_epochs = None



#%% Project representations onto coding plane: 
    
# Sample random run: # TODO: come up with better way of specifically selecting run?
# For time being can add more conditions to flt function (including specific repeat number);
# would be better to ultiamtely have more principled way of doing this though,
# e.g., choose run with highest (or median) parallelism; for that though need 
# to match runs between ae_df and geo_df; should have some way of doing this. 
curr_row = ae_df.sample(1)
X = curr_row.iloc[0].hidden_rep
base_labels = curr_row.iloc[0].labels[:,0]
proj_labels = curr_row.iloc[0].labels[:,1]

# Project data onto coding plane, generate plot:
xhat, yhat, angle = proj_code_plane(X, base_labels=base_labels, 
    proj_labels=proj_labels, zscore_data=zscore_data, classifier=clf_type, 
    sum_bins=sum_bins, base_name=base_name, proj_name=proj_name, 
    plot_coding_axes=True, save_output=False)

# Define title:
title_str0 = 'Hidden layer representation projected onto coding plane'

# Assuming for the time being that the 'hidden_rep' column of ae_df is the final 
# representation after all training epochs:
title_str1 = []
if n_epochs is not None:
    title_str1 = 'Representation after {} epochs training'.format(n_epochs)

title_str2 = []
if np.ptp(ae_df.n_hidden) == 0:
    title_str2.append('n_hidden={}'.format(ae_df.iloc[0].n_hidden))

if np.ptp(ae_df.beta_rec) == 0:
    title_str2.append('beta_rec='+r'$10^{{{}}}$'.format(round(np.log10(ae_df.iloc[0].beta_rec), ndigits=2)))    

if np.ptp(ae_df.beta_sp) == 0:
    title_str2.append('beta_sp={}'.format(ae_df.iloc[0].beta_sp))
title_str2 = ', '.join(title_str2)

title_str = '\n'.join([title_str0, title_str1, title_str2])
plt.title(title_str)
plt.tight_layout()



#%% Save output if requested:

if save_output: 
    
    # Define output directory
    input_dir = os.path.split(input_path)[0]
    if 'analysis_metadata' in sys.modules:
        curr_output_dir = increment_dir_name(input_dir, 'proj_hidden_code_plane')
    else: 
        curr_output_dir = os.path.join(input_dir, 'proj_hidden_code_plane')
        
    # Create current output directory if necessary:
    if not os.path.exists(curr_output_dir):
        pathlib.Path(curr_output_dir).mkdir(parents=True,exist_ok=True)
    
    fig_basename = 'proj_hidden_code_plane'
    
    png_path = os.path.join(curr_output_dir, fig_basename+'.png')
    plt.savefig(png_path)

    svg_path = os.path.join(curr_output_dir, fig_basename+'.svg')
    plt.savefig(svg_path)
    
    # Save metadata:
    if 'analysis_metadata' in sys.modules:
        
        M = Metadata()
        M.add_input(input_path)
        M.add_output(png_path)
        M.add_output(svg_path)
        metadata_path = os.path.join(curr_output_dir, 'proj_hidden_code_plane.json')
        write_metadata(M, metadata_path)