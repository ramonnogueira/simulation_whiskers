# -*- coding: utf-8 -*-
"""
Plot various loss terms vs. training epoch for autoencoder/prediction model.

Created on Sun Sep  8 09:15:45 2024

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

# Define inputs/parameters:

# Input parameters:
input_path = 'E:\\simulation_whiskers\\results\\run667\\ae_iterate_beta_reconstruction.pickle'

# Define independent variable:
loss = 'loss_rec_epochs' # 'loss_rec_epochs' | 'loss_rec_binned' | 'loss_ce_epochs' | 'loss_sp_epochs' | 'loss_epochs' | 'loss_xor_epochs'

# Define custom filter if desired:
flt = lambda x : round(x.beta_rec) == round(10**1)
#flt = lambda x : round(x.beta_sp) == 150 
#flt = None

# Output parameters:
save_output = False


###############################################################################
# Preliminary stuff:

# Load results:
results = pickle.load(open(input_path, 'rb'))
ae_df = results['ae_df']    

# Apply any filters if requested:
if flt is not None:
    keep = ae_df.apply(flt, axis=1)
    ae_df = ae_df[keep]
    

###############################################################################
# Generate figure:

if loss == 'loss_rec_binned':
    loss_terms = [x for x in ae_df.keys() if 'loss_rec' in x]
else:
    loss_terms = [loss]
    
grouping_variables = ['beta_sp', 'beta0', 'beta1', 'beta_xor', 'beta_rec', 'n_hidden']

B = ae_df[grouping_variables + loss_terms]\
    .groupby(grouping_variables)

Mu = B.mean().reset_index()
Mu['n_repeats'] = B.count().reset_index()[loss_terms[0]]

#L = np.array(list(ae_df[loss])) # repeats-by-training epochs
#mu = np.mean(L, axis=0)

# Define labels:
Mu['log_beta_rec'] = np.round(np.log10(Mu.beta_rec), decimals=1)
labels = Mu.apply(lambda x : 'beta0={}, beta1={}, beta_xor={}, beta_sp={}, log(beta_rec)={}, n_repeats={}'\
         .format(x.beta0, x.beta1, x.beta_xor, x.beta_sp, x.log_beta_rec, x.n_repeats), axis=1)
    
# Plot:
loss_fig = plt.figure()
for loss in loss_terms:
    curr_labels = [loss + ', ' + x for x in labels] 
    if len(curr_labels) == 1:
        curr_labels = curr_labels[0]
    plt.plot(np.array(list(Mu[loss])).T, label=curr_labels)

# Create title, axis labels, etc:
if loss == 'loss_epochs':
    loss_str = 'Total_loss'
elif loss == 'loss_rec_epochs':
    if 'model_type' in ae_df:
        if len(np.unique(ae_df.model_type)) == 1:
            if ae_df.iloc[0].model_type == 'autoencoder':
                loss_str = 'Reconstruction loss'
            elif ae_df.iloc[0].model_type == 'prediction':
                loss_str = 'Prediction loss'
        else:
            loss_str = 'Reconstruction/prediction loss'
    else:
        loss_str = 'Reconstruction loss'
elif loss == 'loss_ce_epochs':
    loss_str = 'Linear task loss'
elif loss == 'loss_xor_epochs':
    loss_str = 'XOR loss'
elif loss == 'loss_sp_epochs':
    loss_str = 'Sparsity loss'



title = '{} vs training epoch'.format(loss_str)
plt.xlabel('Training epoch')
plt.ylabel(loss_str)
plt.title(title)
plt.legend(frameon=False, prop={'size':6})



###############################################################################
# Save output if requested:
    
if save_output:
    
    # Define current output directory:
    input_dir = os.path.split(input_path)[0]
    if 'analysis_metadata' in sys.modules:
        curr_output_dir = increment_dir_name(input_dir, 'loss_plot')
    else: 
        curr_output_dir = os.path.join(input_dir, 'loss_plot')
    
    # Create current output directory if necessary:
    if not os.path.exists(curr_output_dir):
        pathlib.Path(curr_output_dir).mkdir(parents=True,exist_ok=True)
    
    # Save figures:
    fname_loss = loss.lower().replace(' ', '_')
    plt.figure(loss_fig)
    png_path = os.path.join(curr_output_dir, fname_loss+'.png')
    plt.savefig(png_path)
    
    svg_path = os.path.join(curr_output_dir, fname_loss+'.svg')
    plt.savefig(svg_path)
    
    # Save metadata:
    if 'analysis_metadata' in sys.modules:
        
        M = Metadata()
        M.add_input(input_path)
        M.add_output(png_path)
        M.add_output(svg_path)
        metadata_path = os.path.join(curr_output_dir, 'plot_{}.json'.format(fname_loss))
        write_metadata(M, metadata_path)