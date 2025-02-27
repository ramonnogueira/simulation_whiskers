# -*- coding: utf-8 -*-
"""
Fit autoencoders for different values of beta reconstruction/beta prediction/beta 
XOR/beta sparsity/n hidden and measure representational geometry.

Created 2024-06-04

@author: danie
"""
import os
import pathlib
import inspect
import pickle
import numpy as np
import pandas as pd
import json
import itertools
from simulation_whiskers.simulate_task import load_sim_params, load_task_def
from simulation_whiskers.miscellaneous_sparseauto import iterate_fit_autoencoder, fmt_ae_metadata, generate_hparams_df
#from simulation_whiskers.plot import plot_iterate_autoencoder_results, plot_autoencoder_geometry
from simulation_whiskers.plot import plot_iterate_autoencoder_results, plot_ccgps_by_layer, plot_pars_by_layer
from analysis_metadata.analysis_metadata import Metadata, increment_dir_name, write_metadata
import time

# Define paths to parameters files:
#sim_params_paths=['C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv.json', 'C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv_aug5.json']
ae_params_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\hyperparams\\example_autoencoder_hparams.json'

"""
# Convex/concave, curvature (good entangled task) [Done]
sim_params_paths=['C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv5.json']
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\flat_v_curved.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\convex_concave.json'
"""

"""
# Near/far, convex/concave (good entangled task) [Done]
sim_params_paths=['C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv2.json']
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\convex_concave.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\time_mov_rev.json'
"""

#"""
# Near/far, rough/smooth (good disentangled task) [Done]
sim_params_paths=['C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv3.json']
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\freq_sh.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\time_mov_rev.json'
#"""

"""
# Rough/smooth, curvature (good disentangled task) [Done]
sim_params_paths=['C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv6.json']
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\freq_sh.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\flat_v_curved.json'
"""



"""
# Near/far, curved/flat 
sim_params_paths=['C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv3.json']
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\flat_v_curved.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\time_mov.json'
"""

"""
# Convex/concave, rough/smooth
sim_params_paths=['C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\cvx_ccv4.json']
task0_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\freq_sh.json'
task1_def_path='C:\\Users\\danie\\Documents\\\\code_libraries\\simulation_whiskers\\task_defs\\convex_concave.json'
"""


# Define classifier tasks:
task_defs = [
    
    # Task 0:
    [
     lambda x : x.freq_sh==2, 
     lambda x : x.freq_sh==15
     ],
    
    # Task 1:
    [
     lambda x : x.time_mov==10, 
     lambda x : x.time_mov==17
     ]
    ]


# Define general variables:
n_files = 2
n_geo_subsamples = 1
sum_inpt=False
xor=True
zscore_data = False
sig_init = 1.0
save_learning = False
chunked_reconstruction_loss = False

# Define simulation parameters:
concavity = [0]
n_whisk = 2
prob_poiss = 1.01
noise_w = 0.3
spread = 'auto'
speed = 2.0
ini_phase_m = 0
ini_phase_spr = 100
delay_time = 0
freq_m = 3.0
freq_std = 0.1
std_reset = 0
t_total = 2
dt = 0.1
dx = 0.01
n_trials_pre = 50
n_files = 2
amp = 2
freq_sh = [2, 15]
z1 = [4]
max_rad = 50
n_rad = 4
disp = 4.5
theta = [0]
steps_mov = [10, 17]
rad_vec = [6]
init_position = 0

# Autoencoder parameters:
mdl_type = "prediction"
n_hidden = 10
sig_init = 1 
sig_neu = 0.1 
lr = 0.001
beta0 = 0
beta1 = 0
beta_rec = 0
beta_xor = 1.0
n_epochs = 10
batch_size = 10
beta_sp = 0
p_norm = 2
n_splits = 5
n_predictor_bins = 10
n_predicted_bins = 4
    
# Compute parameters:
gpu = False

# Output directory:
base_output_directory='E:\\simulation_whiskers\\results\\'
run_base_name='run'
sv=False
    
# Load simulation hyperparameters, task definition:
#sim_params=load_sim_params(sim_params_path)
#task=load_task_def(task_def_path)


# Define dicts of a bunch of different hyperparamter combinations to try:
"""
    dicts=[
       {'n_whisk' : 2,
        'noise_w' : 0.3, 
        'ini_phase_spr' : 9,
        'n_trials_pre' : 600,
        },
       
       {'n_whisk' : 3,
        'noise_w' : 0.5, 
        'ini_phase_spr' : 3,
        'n_trials_pre' : 20,
        },
       ]    
"""

#beta_lins=[0]
beta_lins=10**np.arange(0, 5, 0.5)
beta_lins = np.array([0] + list(beta_lins))
sig_inits=[1]
n_hiddens=[{'n_hidden':20, 'beta_sp':0.0}, {'n_hidden':80, 'beta_sp':0.0}]   
#n_hiddens=[{'n_hidden':20, 'beta_sp':0.0}]
params=[1]

autoencoder_params=json.load(open(ae_params_path,'r'))  



#%%

simulation_cols = ['concavity', 'n_whisk', 'prob_poiss', 'noise_w', 'spread',
     'speed', 'ini_phase_m', 'ini_phase_spr', 'delay_time', 'freq_m', 'freq_std',
     'std_reset', 't_total', 'dt', 'dx', 'n_trials_pre', 'n_files', 'amp', 'freq_sh',
     'z1', 'max_rad', 'n_rad', 'disp', 'theta', 'steps_mov', 'rad_vec', 'init_position']

autoencoder_cols = ['mdl_type', 'n_hidden', 'sig_init', 'sig_neu', 'lr', 'beta0',
    'beta1', 'beta_rec', 'beta_xor', 'n_epochs', 'batch_size', 'beta_sp', 'p_norm',
    'n_splits', 'n_predictor_bins', 'n_predicted_bins']

hparams_df = generate_hparams_df(hparams=None, task_defs=task_defs, n_files=n_files, 
     xor=xor, n_geo_subsamples=n_geo_subsamples, zscore_data=zscore_data, 
     save_perf=False, sum_inpt=sum_inpt, chunked_reconstruction_loss=False, 
     save_learning=save_learning, gpu=gpu, save_sessions=False, verbose=False, 
     concavity=concavity, n_whisk=n_whisk, prob_poiss=prob_poiss, noise_w=noise_w, 
     spread=spread, speed=speed, ini_phase_m=ini_phase_m, ini_phase_spr=ini_phase_spr, 
     delay_time=delay_time, freq_m=freq_m, freq_std=freq_std, std_reset=std_reset, 
     t_total=t_total, dt=dt, dx=dx, n_trials_pre=n_trials_pre, n_repeats=n_files, 
     amp=amp, freq_sh=freq_sh, z1=z1, max_rad=max_rad, n_rad=n_rad, disp=disp, 
     theta=theta, steps_mov=steps_mov, rad_vec=rad_vec, init_position=init_position, 
     mdl_type=mdl_type, n_hidden=n_hidden, sig_init=sig_init, sig_neu=sig_neu, 
     lr=lr, beta0=beta0, beta1=beta1, beta_rec=beta_rec, beta_xor=beta_xor, 
     beta_sp=beta_sp, n_epochs=n_epochs, batch_size=batch_size, p_norm=p_norm, 
     n_splits=n_splits, n_predictor_bins=n_predictor_bins, n_predicted_bins=n_predicted_bins)


# Verify parameters before executing:
archstrs = ['n_hidden={}, beta_sp={}'.format(x['n_hidden'], x['beta_sp']) for x in n_hiddens]
recstrs = ['beta_lin=10^{}'.format(round(np.log10(x), ndigits=2)) for x in beta_lins]
hparam_strs = list(itertools.product(['model={}'.format(autoencoder_params['type'])], archstrs, recstrs, ['n_epochs={}'.format(autoencoder_params['n_epochs'])]))
print('Running following hyperparameters:\n')
print(hparam_strs)
yn = input('\nProceed? (y/n)')
if yn == 'y':
    pass
else: 
    raise AssertionError('User aborted execution.')



# Initialize lists:
perf_orig = []
task_inpt = []
ccgp_inpt = []
parallelism_inpt = []
task_hidden_pre = []
ccgp_hidden_pre = []
parallelism_hidden_pre = []
task_hidden = []
ccgp_hidden = []
parallelism_hidden = []
task_rec = []
ccgp_rec = []
parallelism_rec = []

n_hidden_all = []
sig_inits_all = []
betas_all = []

# Iterate over dicts of hyperparamter combos:
all_geo_results = pd.DataFrame()
all_perf_results = pd.DataFrame()
all_ae_results = pd.DataFrame()
#for d in params:
start = time.time()

for hidx, hparams in hparams_df.iterrows():
    
    curr_sim_params = dict(hparams[simulation_cols])
    curr_autoencoder_params = dict(hparams[autoencoder_cols])
    
    curr_results=iterate_fit_autoencoder(curr_sim_params,  
        tasks=hparams.task_defs, autoencoder_params=curr_autoencoder_params, xor=hparams.xor, 
        n_geo_subsamples=hparams.n_geo_subsamples, zscore_data=hparams.zscore_data, 
        save_perf=False, sum_inpt=hparams.sum_inpt, chunked_reconstruction_lss=hparams.chunked_reconstruction_loss, 
        save_learning=hparams.save_learning, gpu=hparams.gpu, save_sessions=False, 
        verbose=True)

    curr_hparams_df = pd.DataFrame(hparams_df.iloc[0]).T

    # Extract geometry results, add metadata:
    curr_geo_results = curr_results['geo_df']
    geo_meta_cols = set(curr_hparams_df) - set(curr_geo_results.columns)
    geo_meta = pd.concat([curr_hparams_df[geo_meta_cols]]*curr_geo_results.shape[0],axis=0)
    curr_geo_results = pd.concat([curr_geo_results, geo_meta], axis=1)
    all_geo_results = pd.concat([all_geo_results, curr_geo_results], axis=0)
    
    # Extract classifier performance results, add metadata:
    curr_perf_results = curr_results['perf_df']
    perf_meta_cols = set(curr_hparams_df) - set(curr_perf_results.columns)
    perf_meta = pd.concat([curr_hparams_df[perf_meta_cols]]*curr_perf_results.shape[0],axis=0)
    curr_perf_results = pd.concat([curr_perf_results, perf_meta], axis=1)
    all_perf_results = pd.concat([all_perf_results, curr_perf_results], axis=0)

    # Extract autoencoder representations, add metadata:
    if curr_results['ae_df'] is not None:            
        curr_ae_results = curr_results['ae_df']
        ae_meta_cols = set(curr_hparams_df) - set(curr_ae_results.columns)
        ae_meta = pd.concat([curr_hparams_df[ae_meta_cols]]*curr_ae_results.shape[0],axis=0)
        curr_ae_results = pd.concat([curr_ae_results, perf_meta], axis=1)
        all_ae_results = pd.concat([all_ae_results, curr_ae_results])

all_results = dict()
all_results['geo_df'] = all_geo_results
all_results['perf_df'] = all_perf_results
all_results['ae_df'] = all_ae_results

stop = time.time()


#%% Save output:
if sv:
    
    # Save results dataframe:
    curr_output_directory=increment_dir_name(base_output_directory, run_base_name)
    if not os.path.exists(curr_output_directory):
        pathlib.Path(curr_output_directory).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(curr_output_directory, 'ae_iterate_beta_reconstruction.pickle')
    pickle.dump(all_results, open(results_path, 'wb'))
    
    # Save metadata:
    N = fmt_ae_metadata(sim_params, autoencoder_params)
    del N.parameters['autoencoder_params']['n_hidden']    
    del N.parameters['autoencoder_params']['sig_init']
    del N.parameters['autoencoder_params']['beta0']
    del N.parameters['autoencoder_params']['beta1']
    del N.parameters['autoencoder_params']['beta_xor']    
    
    M = Metadata()
    M.add_param('sim_params', N.parameters['sim_params'])  
    M.add_param('autoencoder_params', N.parameters['autoencoder_params'])  
    for tidx, task in enumerate(tasks):
        class_def_strs = [inspect.getsource(x).strip().replace(',','') for x in task]
        task_str = ' vs '.join(class_def_strs)
        M.add_param('task{}'.format(tidx), task_str)
    M.add_param('n_geo_subsamples', n_geo_subsamples)
    M.add_output(results_path)
    M.duration = stop - start
    metadata_path = os.path.join(curr_output_directory, 'ae_iterate_hidden_size_metadata.json')
    write_metadata(M, metadata_path)