# -*- coding: utf-8 -*-
"""
Example script for generating simulated whisker contact and angle data then 
fitting task-optimized autoencoder. 

Works for DDK as of 2023-03-07. 

Created on Mon Mar  6 20:42:49 2023

@author: danie
"""
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from simulation_whiskers.simulate_task import simulate_session, session2feature_array, session2labels, load_hyperparams, load_task_def
from simulation_whiskers import miscellaneous_sparseauto 

# Define general variables:
hparams_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\simulate_task_hparams.json'
task_def_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\task_defs\\convex_concave.json'
n_files=5

# Define autoencoder-related variables:
n_hidden=10
sig_init=1
sig_neu=0.1
lr=1e-3
beta=0.5
beta_sp=0.25
p_norm=2

# Define batch parameters:
batch_size=10
n_epochs=500

# Load simulation hyperparameters, task definition:
hparams=load_hyperparams(hparams_path)
task=load_task_def(task_def_path)

# Initialize output arrays:
perf_orig=np.zeros((n_files,2))
perf_out=np.zeros((n_files,n_epochs,2))
perf_hidden=np.zeros((n_files,n_epochs,2))
loss_epochs=np.zeros((n_files,n_epochs))

# Iterate over files:
for k in range(n_files):
    
    # Simulate session:
    session=simulate_session(hparams)
    
    # Prepare simulated trial data for autoencoder:
    F=session2feature_array(session) # extract t-by-g matrix of feature data, where t is number of trials, g is total number of features (across all time bins)
    n_inp=F.shape[1]
    x_torch=Variable(torch.from_numpy(np.array(F,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor
    labels=session2labels(session, task) # generate vector of labels    
    labels_torch=Variable(torch.from_numpy(np.array(labels,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
    
    # Test logistic regression performance on original data:
    perf_orig[k]=miscellaneous_sparseauto.classifier(F,labels,1)
    
    # Create and fit task-optimized autoencoder:
    model=miscellaneous_sparseauto.sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init,k=np.unique(labels)) 
    loss_rec_vec, loss_ce_vec, loss_sp_vec, loss_vec, data_epochs, data_hidden=miscellaneous_sparseauto.fit_autoencoder(model=model,data=x_torch, clase=labels_torch, n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu, beta=beta, beta_sp=beta_sp, p_norm=p_norm)
    loss_epochs[k]=loss_vec
    
    # Test logistic regression performance on reconstructed data:
    for i in range(n_epochs):
        perf_out[k,i]=miscellaneous_sparseauto.classifier(data_epochs[i],labels,1)
        perf_hidden[k,i]=miscellaneous_sparseauto.classifier(data_hidden[i],labels,1)
    
    
# Plot loss:
loss_m=np.mean(loss_epochs,axis=0)
plt.plot(loss_m)
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.show()

# Plot performance
perf_m=np.mean(perf_orig,axis=0)
perf_out_m=np.mean(perf_out,axis=0)
perf_diff_m=np.mean(perf_diff,axis=0)
plt.plot(perf_out_m[:,1],color='blue',label='Out')
plt.plot(perf_diff_m[:,1],color='red',label='Diff')
plt.plot(perf_m[1]*np.ones(n_epochs),color='grey',label='Input')
plt.plot(0.5*np.ones(n_epochs),color='black',linestyle='--')
plt.ylim([0,1.1])
plt.ylabel('Decoding Performance')
plt.xlabel('Epochs')
plt.legend(loc='best')
plt.show()