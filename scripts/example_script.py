# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:16:32 2022

@author: DDK
"""
import os
import sys
from datetime import datetime
from simulation_whiskers.simulate_task import compare_stim_decoders
try:
    from analysis_metadata.analysis_metadata import find_max_dir_suffix
except:
    pass


sim_params_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\hyperparams\\example_sim_params.json'
mlp_hparams_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\hyperparams\\example_mlp_hparams.json'
task_def_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\task_defs\\convex_concave.json'
save_figs=False
base_output_directory='C:\\Users\\danie\\Documents\\simulation_whiskers\\results\\compare_stim_decoders'
verbose=True

if 'analysis_metadata' in sys.modules:
    curr_max_dir_suffix = find_max_dir_suffix(base_output_directory, 'run')
    curr_output_dir = 'run'+str(curr_max_dir_suffix+1).zfill(3)
    curr_output_dir_full = os.path.join(base_output_directory, curr_output_dir)
else:
    curr_output_dir_full = base_output_directory

start = datetime.now()
compare_stim_decoders(sim_params=sim_params_path, mlp_hparams=mlp_hparams_path, task=task_def_path, save_figs=save_figs, output_directory=curr_output_dir_full, verbose=verbose)
stop = datetime.now()
duration = stop - start
print('Duration = {} seconds'.format(duration.seconds))