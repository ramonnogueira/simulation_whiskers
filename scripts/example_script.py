# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:16:32 2022

@author: DDK
"""

from datetime import datetime
from simulation_whiskers.simulate_task import compare_stim_decoders


sim_params_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\hyperparams\\example_sim_params.json'
mlp_hparams_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\hyperparams\\example_mlp_hparams.json'
task_def_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\task_defs\\convex_concave.json'
save_figs=False
output_directory='C:\\Users\\danie\\Documents\\simulation_whiskers\\results\\'
verbose=True

start = datetime.now()
compare_stim_decoders(sim_params=sim_params_path, mlp_hparams=mlp_hparams_path, task=task_def_path, save_figs=save_figs, output_directory=output_directory, verbose=verbose)
stop = datetime.now()
duration = stop - start
print('Duration = {} seconds'.format(duration.seconds))