# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:16:32 2022

@author: DDK
"""

from datetime import datetime
from simulation_whiskers.simulate_task import compare_stim_decoders


hparams_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\simulate_task_hparams.json'
save_figs=True
output_directory='C:\\Users\\danie\\Documents\\simulation_whiskers\\results\\'
verbose=True

start = datetime.now()
compare_stim_decoders(hparams=hparams_path, save_figs=save_figs, output_directory=output_directory, verbose=verbose)
stop = datetime.now()
duration = stop - start
print('Duration = {} seconds'.format(duration.seconds))