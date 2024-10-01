# -*- coding: utf-8 -*-
"""
Plot bar graphs of task0 performance, task1 performance, XOR performance, task0
CCGP w.r.t task1, and task 1 CCGP w.r.t task0, in that order, for all model layers
(including input) in previously-saved results. 

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
from simulation_whiskers.plot import plot_ccgps_by_layer
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, increment_dir_name, seconds_2_full_time_str
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False

# Input parameters:
input_path = 'E:\\simulation_whiskers\\results\\run651\\ae_iterate_beta_reconstruction.pickle'

# Output parameters:
save_output = False



#%% Preliminary stuff:

# Load results:
results = pickle.load(open(input_path, 'rb'))
perf_df = results['perf_df']
geo_df = results['geo_df']

# Plot:
pl = plot_ccgps_by_layer(perf_df, geo_df)



#%% Save output if requested:
    
if save_output:
    
    # Define current output directory:
    input_dir = os.path.split(input_path)[0]
    if 'analysis_metadata' in sys.modules:
        curr_output_dir = increment_dir_name(input_dir, 'task_perf_bars')
    else: 
        curr_output_dir = os.path.join(input_dir, 'task_perf_bars')
    
    # Create current output directory if necessary:
    if not os.path.exists(curr_output_dir):
        pathlib.Path(curr_output_dir).mkdir(parents=True,exist_ok=True)
    
    # Save figures:    
    perf_fig_basename = 'task_perf_bars'
    
    perf_png_path = os.path.join(curr_output_dir, perf_fig_basename+'.png')
    plt.savefig(perf_png_path)

    perf_svg_path = os.path.join(curr_output_dir, perf_fig_basename+'.svg')
    plt.savefig(perf_svg_path)
    
    # Save metadata:
    if 'analysis_metadata' in sys.modules:
        
        M = Metadata()
        M.add_input(input_path)
        M.add_output(perf_png_path)
        M.add_output(perf_svg_path)
        metadata_path = os.path.join(curr_output_dir, 'task_perf_bars.json')
        write_metadata(M, metadata_path)