# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:16:32 2022

@author: danie
"""

from simulation_whiskers.simulate_task import compare_stim_decoders

hparams_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\simulate_task_hparams.json'
save_figs=True
output_directory='C:\\Users\\danie\\Documents\\simulation_whiskers\\results'
verbose=True
compare_stim_decoders(hparams=hparams_path, save_figs=save_figs, output_directory=output_directory, verbose=verbose)
