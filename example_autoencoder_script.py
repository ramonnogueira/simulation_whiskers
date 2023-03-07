# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:00:10 2023

@author: danie
"""
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from simulation_whiskers.simulate_task import simulate_session, session2feature_array, load_hyperparams
from simulation_whiskers import miscellaneous_sparseauto 

# Define general variables:
#hparams_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\simulate_task_hparams.json'
hparams_path='C:\\Users\\danie\\Documents\\simulation_whiskers\\example_hparams.json'
n_files=5

# Define autoencoder-related variables:
n_hidden=10
sig_init=1
sig_neu=0.1
lr=1e-3

# Define batch parameters:
batch_size=10
n_epochs=500

# Load simulation hyperparameters, task definition:
hparams=load_hyperparams(hparams_path)

# Initialize output arrays:
loss_epochs=np.zeros((n_files,n_epochs))

# Iterate over files:
for k in range(n_files):
    
    # Simulate session:
    session=simulate_session(hparams, [6, 12, 24])
    
    # Prepare simulated trial data for autoencoder:
    F=session2feature_array(session) # extract t-by-g matrix of feature data, where t is number of trials, g is total number of features (across all time bins)
    n_inp=F.shape[1]
    x_torch=Variable(torch.from_numpy(np.array(F,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor  
    
    # Create and fit autoencoder:
    model=miscellaneous_sparseauto.sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init) 
    loss_vec,data_epochs=miscellaneous_sparseauto.fit_autoencoder(model=model,data=x_torch, n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu)
    loss_epochs[k]=loss_vec
    
# Plot loss:
loss_m=np.mean(loss_epochs,axis=0)
plt.plot(loss_m)
plt.ylabel('Training Loss')
plt.xlabel('Epochs')
plt.show()


