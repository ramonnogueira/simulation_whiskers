# -*- coding: utf-8 -*-
"""
Utility script to stitch together results of different runs of iterate_ae.py.

Created on Fri Sep 27 20:19:45 2024

@author: danie
"""

import os
import pathlib
import pandas as pd
import numpy as np
import pickle
from analysis_metadata.analysis_metadata import Metadata, increment_dir_name, write_metadata

# Define input paths:
input_paths = [
    'E:\\simulation_whiskers\\results\\run666\\ae_iterate_beta_reconstruction.pickle',
    'E:\\simulation_whiskers\\results\\run667\\ae_iterate_beta_reconstruction.pickle'
    ]
    
# Define output settings:
save_output = True
base_output_directory='E:\\simulation_whiskers\\results\\'
run_base_name='run'

    

#%% Merge together input dataframes:

geo_df = pd.DataFrame()
perf_df = pd.DataFrame()
ae_df = pd.DataFrame()

for inpt in input_paths:
    
    # Load:
    curr_result = pickle.load(open(inpt,'rb'))
    curr_geo_df = curr_result['geo_df']
    curr_perf_df = curr_result['perf_df']
    curr_ae_df = curr_result['ae_df']
    
    # Merge:
    geo_df = pd.concat([geo_df, curr_geo_df])
    perf_df = pd.concat([perf_df, curr_perf_df])
    ae_df = pd.concat([ae_df, curr_ae_df])

results = dict()
results['geo_df'] = geo_df
results['perf_df'] = perf_df
results['ae_df'] = ae_df


#%% Save results:

if save_output: 
    
    # Save results dataframe:
    curr_output_directory=increment_dir_name(base_output_directory, run_base_name)
    if not os.path.exists(curr_output_directory):
        pathlib.Path(curr_output_directory).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(curr_output_directory, 'ae_iterate_beta_reconstruction.pickle')
    pickle.dump(results, open(results_path, 'wb'))
    
    # Save metadata:
    M = Metadata()
    for inpt in input_paths:
        M.add_input(inpt)
    metadata_path = os.path.join(curr_output_directory, 'stitch_ae_results_metadata.json')
    write_metadata(M, metadata_path)