# -*- coding: utf-8 -*-
"""
Fit autoencoders for different values of beta reconstruction/beta prediction/beta 
XOR/beta sparsity/n hidden and measure representational geometry.

Created 2024-06-04

@author: danie
"""
import os
import pathlib
import pickle
import numpy as np
import pandas as pd
import json
import itertools
from simulation_whiskers.simulate_task import load_sim_params, load_task_def
from simulation_whiskers.miscellaneous_sparseauto import iterate_fit_autoencoder, fmt_ae_metadata
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



# Define general variables:
n_files = 10
n_geo_subsamples = 1
sum_inpt=False
xor=True
zscore_data = False
sig_init = 1.0
save_learning = False
chunked_reconstruction_loss = False

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

#beta_lins=[10**4.5]
beta_lins=10**np.arange(0, 0.5, 5)
beta_lins = np.array([0] + list(beta_lins))
sig_inits=[1]
#n_hiddens=[{'n_hidden':40, 'beta_sp':0.0}]   
n_hiddens=[{'n_hidden':20, 'beta_sp':0.0}, {'n_hidden':40, 'beta_sp':0.0}, {'n_hidden':80, 'beta_sp':0.0}, {'n_hidden':120, 'beta_sp':0.0}]   
#n_hiddens=[{'n_hidden':240, 'beta_sp':100.0}, {'n_hidden':240, 'beta_sp':200.0}] 
params=[1]

autoencoder_params=json.load(open(ae_params_path,'r'))  

"""
# Iterate over dicts of hyperparamter combos:
for d in params:
    
    # Define new output directory:
    curr_output_directory=increment_dir_name(base_output_directory, run_base_name)
    #curr_output_directory='C:\\Users\\danie\\Documents\\simulation_whiskers\\results\\run015'    
    
    # Write new parameter values to sim_params:
    
    autoencoder_params['beta']=d[0]
    autoencoder_params['n_hidden']=d[1]
    sim_params['ini_phase_spr']=d[2]
    
    # Fit autoencoder, test classifier performance:
    results=iterate_fit_autoencoder(sim_params, autoencoder_params, task, n_files, save_perf=sv, save_learning=False, save_sessions=False, output_directory=curr_output_directory, verbose=True)
    
    # Plot results: 
    
    #results_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\results\\run015\\iterate_autoencoder_results.h5'
    #loss_plot, perf_plot=plot_iterate_autoencoder_results(results, save_output=sv, output_directory=curr_output_directory)
    geo_plot=plot_autoencoder_geometry(results['task_hidden'], results['ccgp_hidden'], rec_lr=results['task_rec'], rec_ccgp=results['ccgp_rec'], inpt_lr=results['task_inpt'], inpt_ccgp=results['ccgp_inpt'], plot_train=True, save_output=sv, output_directory=curr_output_directory)
"""



###########################################################


# Load simulation hyperparameters, task definition:

task0=load_task_def(task0_def_path)
task1=load_task_def(task1_def_path)
tasks=[task0,task1]

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
for spath in sim_params_paths:

    sim_params=load_sim_params(spath)
    
    for nh in n_hiddens:
            
        n_hidden=nh['n_hidden']
        beta_sp=nh['beta_sp']
        
        for beta_lin in beta_lins:
            
            """
            beta_lin = np.round(beta_lin, decimals=2)
            beta_xor = 1.0 - beta_lin
            
            autoencoder_params['n_hidden']=n_hidden
            autoencoder_params['beta0']=beta_lin/2
            autoencoder_params['beta1']=beta_lin/2
            autoencoder_params['beta_rec']=0
            autoencoder_params['beta_xor']=beta_xor
            autoencoder_params['beta_sp']=beta_sp
            autoencoder_params['sig_init']=sig_init
            """

            beta_lin = np.round(beta_lin, decimals=2)
            
            autoencoder_params['n_hidden']=n_hidden
            autoencoder_params['beta0']=0
            autoencoder_params['beta1']=0
            autoencoder_params['beta_rec']=beta_lin
            autoencoder_params['beta_xor']=1.0
            autoencoder_params['beta_sp']=beta_sp
            autoencoder_params['sig_init']=sig_init
    
            #print('beta_xor={}'.format(beta_xor))    
            #print('beta_task={}'.format(beta_task))
            #print('beta_rec={}\n'.format(autoencoder_params['beta_rec']))
            
            #"""
            # Fit autoencoder, test classifier performance:
            curr_results=iterate_fit_autoencoder(sim_params,  
                tasks, n_files, autoencoder_params=autoencoder_params, xor=xor, n_geo_subsamples=n_geo_subsamples, zscore_data=zscore_data, save_perf=False, sum_inpt=sum_inpt, 
                chunked_rec=chunked_reconstruction_loss, save_learning=save_learning, gpu=gpu, save_sessions=False, verbose=True)
            #all_results = pd.concat([all_results, results], axis=0)
            
            # Aggregate results:                
            curr_geo_results = curr_results['geo_df']
            curr_geo_results['beta0'] = [autoencoder_params['beta0']]*curr_geo_results.shape[0]    
            curr_geo_results['beta1'] = [autoencoder_params['beta1']]*curr_geo_results.shape[0]    
            curr_geo_results['beta_xor'] = [autoencoder_params['beta_xor']]*curr_geo_results.shape[0]
            curr_geo_results['beta_rec'] = [autoencoder_params['beta_rec']]*curr_geo_results.shape[0]
            curr_geo_results['beta_sp'] = [autoencoder_params['beta_sp']]*curr_geo_results.shape[0]
            curr_geo_results['n_hidden'] = [autoencoder_params['n_hidden']]*curr_geo_results.shape[0]    
            curr_geo_results['sig_init'] = [sig_init]*curr_geo_results.shape[0]    
            all_geo_results = pd.concat([all_geo_results, curr_geo_results], axis=0)
            
            curr_perf_results = curr_results['perf_df']
            curr_perf_results['beta0'] = [autoencoder_params['beta0']]*curr_perf_results.shape[0]    
            curr_perf_results['beta1'] = [autoencoder_params['beta1']]*curr_perf_results.shape[0]    
            curr_perf_results['beta_xor'] = [autoencoder_params['beta_xor']]*curr_perf_results.shape[0]
            curr_perf_results['beta_rec'] = [autoencoder_params['beta_rec']]*curr_perf_results.shape[0]
            curr_perf_results['beta_sp'] = [autoencoder_params['beta_sp']]*curr_perf_results.shape[0]
            curr_perf_results['n_hidden'] = [autoencoder_params['n_hidden']]*curr_perf_results.shape[0]    
            curr_perf_results['sig_init'] = [sig_init]*curr_perf_results.shape[0]    
            all_perf_results = pd.concat([all_perf_results, curr_perf_results], axis=0)
            
            if curr_results['ae_df'] is not None:            
                curr_ae_results = curr_results['ae_df']
                curr_ae_results['beta0'] = [autoencoder_params['beta0']]*curr_ae_results.shape[0]    
                curr_ae_results['beta1'] = [autoencoder_params['beta1']]*curr_ae_results.shape[0]    
                curr_ae_results['beta_xor'] = [autoencoder_params['beta_xor']]*curr_ae_results.shape[0]
                curr_ae_results['beta_rec'] = [autoencoder_params['beta_rec']]*curr_ae_results.shape[0]
                curr_ae_results['beta_sp'] = [autoencoder_params['beta_sp']]*curr_ae_results.shape[0]
                curr_ae_results['n_hidden'] = [autoencoder_params['n_hidden']]*curr_ae_results.shape[0]    
                curr_ae_results['sig_init'] = [sig_init]*curr_ae_results.shape[0]    
                all_ae_results = pd.concat([all_ae_results, curr_ae_results])
            #"""
            
            # Plot results: 
            
            #results_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\results\\run326\\iterate_autoencoder_results.h5'
            #loss_plot, perf_plot=plot_iterate_autoencoder_results(results, save_output=sv, output_directory=curr_output_directory)
            #geo_plot=plot_autoencoder_geometry(results['task_hidden'], results['ccgp_hidden'], rec_lr=results['task_rec'],1 rec_ccgp=results['ccgp_rec'], inpt_lr=results['task_inpt'], inpt_ccgp=results['ccgp_inpt'], pre_lr=results['task_hidden_pre'], pre_ccgp=results['ccgp_hidden_pre'], plot_train=True, save_output=sv, output_directory=curr_output_directory)
            #ccgp_plot=plot_ccgps_by_layer(results['task_hidden'], results['ccgp_hidden'], rec_lr=results['task_rec'], rec_ccgp=results['ccgp_rec'], inpt_lr=results['task_inpt'], inpt_ccgp=results['ccgp_inpt'], pre_lr=results['task_hidden_pre'], pre_ccgp=results['ccgp_hidden_pre'], plot_train=True, save_output=sv, output_directory=curr_output_directory)
            #par_plot=plot_pars_by_layer(results['parallelism_inpt'], results['parallelism_hidden_pre'], results['parallelism_hidden'], results['parallelism_rec'],  save_output=sv, output_directory=curr_output_directory)

all_results = dict()
all_results['geo_df'] = all_geo_results
all_results['perf_df'] = all_perf_results
all_results['ae_df'] = all_ae_results

stop = time.time()

# Save output:
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
    M.add_param('task0', task0)
    M.add_param('task1', task1)
    M.add_param('n_geo_subsamples', n_geo_subsamples)
    M.add_output(results_path)
    M.duration = stop - start
    metadata_path = os.path.join(curr_output_directory, 'ae_iterate_hidden_size_metadata.json')
    write_metadata(M, metadata_path)