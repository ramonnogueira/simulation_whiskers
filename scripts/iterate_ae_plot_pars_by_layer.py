# -*- coding: utf-8 -*-
"""
Plot parallelism by layer for results generated in iterate_ae.py script. 

Created on Sat Sep 28 18:34:11 2024

@author: danie
"""
import sys
import os
import pathlib
import pickle
import numpy as np
from simulation_whiskers.plot import plot_pars_by_layer
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, increment_dir_name, seconds_2_full_time_str
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False
import matplotlib.pyplot as plt

# Define input file:
input_path = 'E:\\simulation_whiskers\\results\\run651\\ae_iterate_beta_reconstruction.pickle'
    
# Define custom filters
flt = lambda x : x.n_hidden==160 and round(x.beta_rec)==round(10**4.5)

# Output settings:
save_output = False


#%% Plot results:

# Load:    
results = pickle.load(open(input_path, 'rb'))
geo_df = results['geo_df']

# Apply custom filters:
geo_df = geo_df[geo_df.apply(flt, axis=1)]

# Pass results to plotting function:
fig = plot_pars_by_layer(geo_df, avg_tasks=True)

# Title, axis labels, etc.:
title_line0 = 'Parallelism by layer\n'

title_line1 = []
if len(np.unique(geo_df.n_hidden)) == 1:
    n_hidden = geo_df.iloc[0].n_hidden
    title_line1.append('n_hidden={}'.format(n_hidden))

if np.ptp(geo_df.beta_sp) == 0:
    beta_sp = geo_df.iloc[0].beta_sp
    title_line1.append('beta_sp={}'.format(beta_sp))

if np.ptp(geo_df.beta_rec) == 0:
    beta_rec = geo_df.iloc[0].beta_rec
    title_line1.append('beta_rec='+r'$10^{{{}}}$'.format(round(np.log10(beta_rec), ndigits=2)))
    
title_line1 = ', '.join(title_line1)
title_str = '\n'.join([title_line0, title_line1])
plt.title(title_str)
plt.tight_layout()
    
    
#%% Save figures, metadata if requested:
if save_output:
    
    input_dir = os.path.split(input_path)[0]
    if 'analysis_metadata' in sys.modules:
        curr_output_dir = increment_dir_name(input_dir, 'par_by_layer')
    else: 
        curr_output_dir = os.path.join(input_dir, 'par_by_layer')
    
    # Create current output directory if necessary:
    if not os.path.exists(curr_output_dir):
        pathlib.Path(curr_output_dir).mkdir(parents=True,exist_ok=True)
        
    # Save plots:
    png_path = os.path.join(curr_output_dir, 'par_by_layer.png')
    plt.savefig(png_path)
    
    svg_path = os.path.join(curr_output_dir, 'par_by_layer.svg')
    plt.savefig(svg_path)
    
    # Save metadata:
    if 'analysis_metadata' in sys.modules:
        
        M = Metadata()
        M.add_input(input_path)
        M.add_output(png_path)
        M.add_output(svg_path)
        metadata_path = os.path.join(curr_output_dir, 'par_by_layer.json')
        write_metadata(M, metadata_path)
    
    