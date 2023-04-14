# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:03:59 2023

@author: danie
"""
from simulation_whiskers.simulate_task import load_sim_params, load_task_def, simulate_session, session2feature_array, session2labels
from simulation_whiskers.miscellaneous_sparseauto import sparse_autoencoder_1, fit_autoencoder
from simulation_whiskers.functions_geometry import geometry_2D

# Define paths to parameters files:
sim_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\simulate_task_sim_params.json'
ae_params_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\hyperparams\\simulate_task_autoencoder_hparams.json'
task_def_path='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\task_defs\\convex_concave.json'

# Output directory:
base_output_directory='C:\\Users\\danie\\Documents\\code_libraries\\simulation_whiskers\\results\\'
run_base_name='run'

# Load simulation hyperparameters:
sim_params=load_sim_params(sim_params_path)
task=load_task_def(task_def_path)
autoencoder_params=json.load(open(ae_params_path,'r'))

# Run simulation:
session=simulate_session(sim_params, sum_bins=True)
F=session2feature_array(session,field='features')
F_summed=session2feature_array(session,field='features_bins_summed')
labels=session2labels(session, task)

# Fit autoencoder:
n_inp=F.shape[1]
n_hidden=autoencoder_params['n_hidden']
sigma_init=autoencoder_params['sig_init']
n_epochs=autoencoder_params['n_epochs']
batch_size=autoencoder_params['batch_size']
lr=autoencoder_params['lr']
sig_neu=autoencoder_params['sig_neu']
#beta=autoencoder_params['beta']
beta=0
beta_sp=autoencoder_params['beta_sp']
p_norm=autoencoder_params['p_norm']
model=sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init,k=len(np.unique(labels))) 
loss_rec_vec, loss_ce_vec, loss_sp_vec, loss_vec, data_epochs_test, data_hidden_test, data_epochs_train, data_hidden_train=fit_autoencoder(model, data_train=F, clase_train=labels, data_test=F, clase_test=labels, n_epochs=n_epochs, batch_size=batch_size, lr=lr, sigma_noise=sig_neu, beta=beta, p_norm=p_norm)    

