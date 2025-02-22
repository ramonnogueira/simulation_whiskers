import os
import sys
from datetime import datetime
import pathlib
import h5py
import pickle
import numpy as np
import pandas as pd
import re
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import zscore
def warn(*args, **kwargs):
    pass
import warnings
from simulation_whiskers.simulate_task import simulate_session, session2feature_array, session2labels, load_simulation, binarize_contacts
from simulation_whiskers.functions_geometry import geometry_2D, find_matching_2d_bin_trials, subsample_2d_bin
warnings.warn = warn
nan=float('nan')
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, seconds_2_full_time_str
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False
import time

# Standard classifier
def classifier(data,clase,reg,model='logistic', hidden_layer_sizes=(10), activation='relu', solver='adam', lr='constant', lr_init=1e-3):
    n_splits=5
    perf=nan*np.zeros((n_splits,2))
    cv=StratifiedKFold(n_splits=n_splits)
    g=-1
    for train_index, test_index in cv.split(data,clase):
        g=(g+1)
        if model=='logistic':
            clf = LogisticRegression(C=reg,class_weight='balanced')
        elif model=='mlp':
            clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation=activation,solver=solver,alpha=reg,learning_rate=lr, learning_rate_init=lr_init)            
        clf.fit(data[train_index],clase[train_index])
        perf[g,0]=clf.score(data[train_index],clase[train_index])
        perf[g,1]=clf.score(data[test_index],clase[test_index])
    return np.mean(perf,axis=0)


# Fit the autoencoder. The data needs to be in torch format
def fit_autoencoder(model,inpt_train,tgt_train, clase_train,inpt_test,clase_test,
    n_epochs,batch_size,lr,sigma_noise,beta0,beta1,beta_rec,beta_sp,p_norm,
    xor=False,beta_xor=0,chunked_rec=False, chunk_size=4,save_learning=True,
    gpu=False,verbose=False):
    """
    Fit task-optimized autoencoder to input data. 


    Parameters
    ----------
    model : simulation_whiskers.miscellaneous_sparseauto.sparse_autoencoder1
        Sparse autoencoder object (defined at bottom of file).

    data : torch.Tensor 
        Tensor encoding t-by-f array, where t is the number of trials and f is
        the number of features per trial.

    clase : torch.Tensor
        Tensor encoding t-element array of trial labels, where t is the number
        of trials.

    n_epochs : int
        Number of training epochs for autoencoder.

    batch_size : int
        Batch size used for training autoencoder.

    lr : float
        Learning rate for training autoencoder.

    sigma_noise : float
        Hidden layer unit noise.

    beta : [0,1]
        Weight assigned to cross-entropy term in loss function. Weight assigned
        to reconstruction term will be 1-beta.

    beta_sp : float
        Weight assigned to sparsity term in loss function.

    p_norm : int
        Exponent used in computing norm of weight vector for sparsity term of 
        loss function. E.g., setting p_norm=2 will use the L2 norm (Euclidean
        distance).


    Returns
    -------
    loss_rec_vec : numpy.ndarray
        Reconstruction loss term across all training epochs.

    loss_ce_vec : numpy.ndarray
        Cross-entropy loss term across all training epochs.

    loss_sp_vec : numpy.ndarray
        Sparsity loss term across all training epochs.
    
    loss_vec : numpy.ndarray
        Total loss function across all training epochs.

    data_epochs : numpy.ndarray
        p-by-t-by-f array of reconstructed input, where p is number of training 
        epochs, t is number of trials, and f is number of input features.
        
    data_hidden : numpy.ndarray
        p-by-t-by-h array of hidden layer activity, where h is the number of 
        hidden layer units.
    """
    
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    train_trial_indices=torch.Tensor(np.arange(len(clase_train)))
    train_loader=DataLoader(torch.utils.data.TensorDataset(inpt_train,tgt_train,train_trial_indices),batch_size=batch_size,shuffle=True)

    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    loss_rec=torch.nn.MSELoss()
    loss_ce0=torch.nn.CrossEntropyLoss()
    loss_ce1=torch.nn.CrossEntropyLoss()
    if xor:
        loss_xor=torch.nn.CrossEntropyLoss()
    model.train()

    outp_train=model(inpt_train,sigma_noise,gpu=gpu) # In case n_epochs = 0 
    outp_test=model(inpt_test,sigma_noise,gpu=gpu) # In case n_epochs = 0 
    
    # Initialize dataframe:
    columns = ['loss_rec', 'loss_ce', 'loss_sp', 'loss_xor', 'inpt_train', 'inpt_test',
       'hidden_train', 'hidden_test', 'rec_train', 'rec_test']
    if chunked_rec:
        n_chunks = int(np.floor(tgt_train.shape[1]/chunk_size))
        for i in np.arange(n_chunks):
            curr_chunk_name = 'loss_rec_chunk{}'.format(i)
            columns += [curr_chunk_name] 
    ae_df = pd.DataFrame(index=np.arange(n_epochs), columns=columns)
    
    # Iterate over training epochs:
    t=0
    while t<n_epochs: 
        #print (t)
        
        # Evaluate training loss, generate hidden and output representations using training trials:
        outp_train=model(inpt_train,sigma_noise,gpu=gpu)
            
        # Evaluate non-CE training loss:
        curr_loss_rec=loss_rec(outp_train[0],tgt_train).item()
        curr_loss_sp=sparsity_loss(outp_train[1],p_norm).item()
        
        # Evaluate CE terms of loss function:
        curr_loss_ce0=loss_ce0(outp_train[2],clase_train[:,0]).item()
        curr_loss_ce1=loss_ce1(outp_train[3],clase_train[:,1]).item()
        
        # Evaluate xor training loss:
        if xor:
            xor_labels=np.sum(np.array(torch.Tensor(clase_train).to('cpu')),axis=1)%2 # Define the XOR function wrt to the two variables
            xor_labels=Variable(torch.from_numpy(np.array(xor_labels,dtype=np.int64)),requires_grad=False)
            curr_loss_xor=loss_xor(outp_train[4],xor_labels.to(device)).item()
        else:
            curr_loss_xor=0
        
        # Add up training losses:
        curr_loss_ce_total=beta0*curr_loss_ce0+beta1*curr_loss_ce1
        curr_loss_total=(beta_rec*curr_loss_rec+curr_loss_ce_total+beta_xor*curr_loss_xor+beta_sp*curr_loss_sp)
        
        # Generate hidden and output layer representations of held-out trials: 
        outp_test=model(inpt_test,sigma_noise,gpu=gpu)

        #if verbose and t%10==0:
        #    print('Running autoencoder training epoch {} out of {}...'.format(t+1,n_epochs))
        if t==0 or t==(n_epochs-1):
            print (t,'rec ',curr_loss_rec,'ce ',curr_loss_ce_total,'sp ',curr_loss_sp,'total ',curr_loss_total)
            
        # Iterate over test batches, evaluate test loss, compute gradient, update weights:
        for batch_idx, (targ1, targ2, trial_indices) in enumerate(train_loader):
           
            optimizer.zero_grad()
            output=model(targ1,sigma_noise,gpu=gpu)

            loss_r=loss_rec(output[0],targ2) # reconstruction error
            
            trial_indices=trial_indices.type(torch.long) # need to do some annoying reformatting to get tensor to work as array of indices
            
            curr_task0_labels=clase_train[trial_indices,0]
            loss_cla0=loss_ce0(output[2],curr_task0_labels) # cross entropy error
            
            curr_task1_labels=clase_train[trial_indices,1]
            loss_cla1=loss_ce1(output[3],curr_task1_labels) # cross entropy error
            
            # compute xor cross-entropy if requested:
            if xor:
                curr_xor_labels=xor_labels[trial_indices]
                loss_x=loss_xor(output[4],curr_xor_labels.to(device))
            else:
                loss_x=0
            
            loss_s=sparsity_loss(output[1],p_norm)
            loss_t=(beta_rec*loss_r+beta0*loss_cla0+beta1*loss_cla1+beta_xor*loss_x+beta_sp*loss_s)

            loss_t.backward() # compute gradient
            optimizer.step() # weight update
            
        # Aggregate results:
        ae_df.loc[t, 'loss_rec'] = curr_loss_rec
        ae_df.loc[t, 'loss_ce'] = curr_loss_ce_total       
        ae_df.loc[t, 'loss_sp'] = curr_loss_sp           
        ae_df.loc[t, 'loss'] = curr_loss_total
        ae_df.loc[t, 'inpt_train'] = [inpt_train.to('cpu').numpy()] 
        ae_df.loc[t, 'inpt_test'] = [inpt_test.to('cpu').numpy()]
        ae_df.loc[t, 'hidden_train'] = [torch.Tensor(outp_train[1].detach()).to('cpu').numpy()]
        ae_df.loc[t, 'hidden_test'] = [torch.Tensor(outp_test[1].detach()).to('cpu').numpy()]
        ae_df.loc[t, 'rec_train'] = [torch.Tensor(outp_train[0].detach()).to('cpu').numpy()]
        ae_df.loc[t, 'rec_test'] = [torch.Tensor(outp_test[0].detach()).to('cpu').numpy()]
        if xor:
            ae_df.loc[t, 'loss_xor'] = curr_loss_xor
        if chunked_rec:
            for i in np.arange(n_chunks):
                curr_chunk_name = 'loss_rec_chunk{}'.format(i)
                curr_start_idx = i*chunk_size 
                curr_stop_idx = (i+1)*chunk_size
                ae_df.loc[curr_chunk_name, t] = loss_rec(outp_train[0][:, curr_start_idx:curr_stop_idx], tgt_train[:, curr_start_idx:curr_stop_idx]) 
        
        t=(t+1)
    model.eval()
    
    return ae_df



def iterate_fit_autoencoder(sim_params, tasks, n_files, autoencoder_params=None, 
    mlp_params=None, zscore_data=False, save_learning=True, test_geometry=True, 
    n_geo_subsamples=10, geo_reg=1.0, xor=False, sum_inpt=True, chunked_rec=False,
    sessions_in=None, save_perf=False, save_sessions=False, plot_xor=False, gpu=False, 
    output_directory=None, verbose=False):
    """
    Iterate fit_autoencoder() function one or more times and, for each iteration,
    capture overall loss vs training epoch as well as various metrics of 
    decoder performance vs training epoch. 

    Parameters
    ----------
    sim_params : dict
        Dict of simulation parameters. Should define same keys as `params` 
        argument to simulation_whiskers.simulate_task.simulate_session() 
        function.
    
    autoencoder_params : dict
        Dict of autoencoder hyperparameters. 
    
    task : dict
        Dict defining task autuoencoder should be jointly optimized to perform.
        Should be same format as `task` argument to 
        simulation_whiskers.simulate_task.session2labels() function.
    
    n_files : int
        Number of times to iterate fit_autoencoder() function. Will generate 
        one simulated session per iteration.
    
    save_output : bool, optional
        Whether to save output to disk.
    
    output_directory : str
        Directory where results should be saved if `save_output` is True. Set 
        to current working directory by default.


    Returns
    -------
    perf_orig : numpy.ndarray
        n_files-by-2 array of classifier performance on original input data. 
        First column is training performance, second is test.
    
    perf_out : numpy.ndarray
        n_files-by-p-by-2 array of classifier performance on reconstructed 
        input, where p is the number of training epochs specified in 
        `autoencoder_params`. For each p-by-2 slice, first column is training 
        performance, second is test.
    
    perf_hidden : numpy.ndarray
        n_files-by-p-by-2 array of classifier performance on hidden layer 
        activity. For each p-by-2 slice, first column is training performance, 
        second is test.
    
    loss_epochs : numpy.ndarray
        n_files-by-p array of total loss vs training epoch.

    """
    start_time=datetime.now()
    n_feat = sim_params['n_whisk']*2
    
    # Initialize dataframe of classifier performance and geometry results:
    perf_df = pd.DataFrame()
    geo_df = pd.DataFrame()
    perf_orig=np.zeros((n_files,2,2)) # Initialize array of classifier performance in input space for both tasks < Isn't this redundant with return from test_autoencoder_geometry??
    perf_orig_df = pd.DataFrame()
    
    # Define task strings:
    task_strs = []
    for t in tasks:
        curr_task_str = ' vs '.join([str(x) for x in t])
        task_strs.append(curr_task_str)
    
    # Unpack some autoencoder parameters:
    if autoencoder_params is not None:
        
        ae_df = pd.DataFrame()
        
        rec_network_type = autoencoder_params['type']
        n_hidden=autoencoder_params['n_hidden']
        if type(n_hidden)!=list and type(n_hidden)!=np.ndarray:
            n_hidden=int(n_hidden)
        sig_init=float(autoencoder_params['sig_init'])
        sig_neu=float(autoencoder_params['sig_neu'])
        lr=float(autoencoder_params['lr'])
        beta0=float(autoencoder_params['beta0'])
        beta1=float(autoencoder_params['beta1'])
        beta_rec=float(autoencoder_params['beta_rec'])
        penalty = autoencoder_params['p_norm']
        if xor:
            beta_xor=float(autoencoder_params['beta_xor'])
        else:
            beta_xor=0
        beta_sp=float(autoencoder_params['beta_sp'])
        p_norm=float(autoencoder_params['p_norm'])
        
        # Get some prediction network params if necessary:
        if rec_network_type=='prediction':
            n_predictor_bins=autoencoder_params['n_predictor_bins']
            n_predicted_bins=autoencoder_params['n_predicted_bins']
        
        # Verify that betas sum to <= 1:
        #if beta0+beta1+beta_xor > 1:
        #    raise ValueError('beta0 + beta1 greater than 1; please ensure beta0 + beta1 <= 1.')
        
        # Unpack some batching parameters:
        batch_size=int(autoencoder_params['batch_size'])
        n_epochs=int(autoencoder_params['n_epochs'])
    else:
        ae_df = None
    
    # If also running MLP:
    if mlp_params!=None:
        mlp_df = pd.DataFrame()
        mlp_hidden_layer_sizes=mlp_params['hidden_layer_sizes']
        mlp_activation=mlp_params['activation']        
        mlp_alpha=mlp_params['alpha']        
        mlp_solver=mlp_params['solver']        
        mlp_lr=mlp_params['learning_rate']        
        mlp_lr_init=mlp_params['learning_rate_init']
    else:
        mlp_df=None
        
    # Load previously-simulated whisker data if requested:
    if sessions_in!=None:
        save_sessions=False # no need to re-save whisker simulation if loading from disk in the first place
        sessions=load_simulation(sessions_in)
        n_files=len(np.unique(sessions.file_idx))
    # If not loading previously-run whisker simulation and save_sessions is True: 
    elif save_sessions:
        train_sessions=[]
        test_sessions=[]
    else:
        train_sessions=None
        test_sessions=None

    for k in range(n_files):
        print('Running file {} out of {}...'.format(k+1,n_files))
        
        # Initialize dataframe of results for current repeat:
        curr_perf_df = pd.DataFrame()
        curr_geo_df = pd.DataFrame()
        curr_perf_orig_df = pd.DataFrame()
        
        # Simulate session (if not loading previously-simulated session):
        if sessions_in==None:
            
            # Generate session for training autoencoder:
            print('Simulating whisker contact data...')
            start_sim = time.time()
            train_session=simulate_session(sim_params, sum_bins=True)
            stop_sim = time.time()
            print('simulate_session duration={}'.format(stop_sim - start_sim))
            train_session['file_idx']=k
            
            # Generate separate session for testing autoencoder:
            test_session=simulate_session(sim_params, sum_bins=True)
            test_session['file_idx']=k
            
            if save_sessions:
                train_sessions.append(test_session)
                test_sessions.append(test_session)
        else:
            session=sessions[sessions.file_idx==k]
        
        # Prepare simulated trial data for *training* autoencoder:
        F_train, train_labels0=prep_data4ae(train_session, tasks[0])
        F_train, train_labels1=prep_data4ae(train_session, tasks[1])
        if zscore_data:
            F_train = zscore(F_train, 0)
            F_train[np.isnan(F_train)] = 0 # < Can get nans in F_test if certain features are all 0 (e.g., no contacts on short whisker before stim moves into place); just replace with 0 
        F_train_torch=Variable(torch.from_numpy(np.array(F_train,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor
        train_labels=np.array([train_labels0,train_labels1])
        train_labels=np.transpose(train_labels)
        train_labels_torch=Variable(torch.from_numpy(np.array(train_labels,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
    
        # Prepare stimulated trial data for *testing* autoencoder:
        F_test, test_labels0=prep_data4ae(test_session, tasks[0])
        F_test, test_labels1=prep_data4ae(test_session, tasks[1])
        if zscore_data:
            F_test = zscore(F_test, 0)
            F_test[np.isnan(F_test)] = 0 # < Can get nans in F_test if certain features are all 0 (e.g., no contacts on short whisker before stim moves into place); just replace with 0 
        F_test_torch=Variable(torch.from_numpy(np.array(F_test,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor
        test_labels=np.array([test_labels0,test_labels1])
        test_labels=np.transpose(test_labels)
        test_labels_torch=Variable(torch.from_numpy(np.array(test_labels,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
            
        # Test logistic regression performance on original data:
        task0_perf_orig=classifier(F_test,test_labels[:,0],1, 'logistic')
        task1_perf_orig=classifier(F_test,test_labels[:,1],1, 'logistic')
        
        curr_perf_orig_df['train'] = [task0_perf_orig[0], task1_perf_orig[0]]
        curr_perf_orig_df['test'] = [task0_perf_orig[1], task1_perf_orig[1]]
        curr_perf_orig_df['task'] = [0,1]
        curr_perf_orig_df['repeat'] = [k]*curr_perf_orig_df.shape[0]
        
        perf_orig_df = pd.concat([perf_orig_df, curr_perf_orig_df], axis=0)
    
        
        # Test MLP if requested:
        if mlp_params!=None:
            perf_orig_mlp=classifier(F_test,test_labels,model='mlp', hidden_layer_sizes=mlp_hidden_layer_sizes, activation=mlp_activation, solver=mlp_solver, reg=mlp_alpha, lr=mlp_lr, lr_init=mlp_lr_init)    
            mlp_df.loc[k,'train'] = perf_orig_mlp[0]
            mlp_df.loc[k,'test'] = perf_orig_mlp[1]
            mlp_df.loc[k,'repeat'] = k
        
        # Train and test autoencoders:
        if autoencoder_params is not None:
            print('Fitting autoencoder...')
            n_inp=F_train.shape[1]
            n_labels_task0=len(np.unique(train_labels[:,0]))
            n_labels_task1=len(np.unique(train_labels[:,1]))
        
            # Initialize task-optimized autoencoder:
            if rec_network_type=='autoencoder':
                model=ae_dispatch(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init,k=[n_labels_task0,n_labels_task1],xor=xor) 
                F_train_tgt_torch = F_train_torch
                F_test_tgt_torch = F_test_torch
                
                train_labels_ae = train_labels_torch
                test_labels_ae = test_labels_torch
                
            elif rec_network_type=='prediction':
                model=prediction_network(n_inp=n_predictor_bins*n_feat, n_hidden=n_hidden, n_out=n_predicted_bins*n_feat, sigma_init=sig_init, xor=xor)
                
                n_offsets = ( F_train_torch.shape[1] - n_feat*(n_predictor_bins + n_predicted_bins) ) / n_feat
                n_offsets = int(n_offsets)
                
                # Slide window across training data:
                f_tr = lambda x : boxcar(x, n_feat*n_predictor_bins, n_offsets, n_feat)
                F_train_torch_expanded = np.concatenate(list(map(f_tr, np.array(F_train_torch))), axis=0)
                
                # Slide window across target data to predict during training:
                f_tgt = lambda x : boxcar(x, n_feat*n_predicted_bins, n_offsets, n_feat)
                F_train_tgt_torch = F_train_torch[:,n_feat*n_predictor_bins:]
                F_train_tgt_torch_expanded = np.concatenate(list(map(f_tgt, np.array(F_train_tgt_torch))), axis=0)
                
                # Convert training data back to torch:
                F_train_torch = Variable(torch.from_numpy(F_train_torch_expanded), requires_grad=False)
                F_train_tgt_torch = Variable(torch.from_numpy(F_train_tgt_torch_expanded), requires_grad=False)
                
                # Reshape training labels:
                train_labels_ae = np.concatenate([np.matlib.repmat(row,n_offsets,1) for row in train_labels],axis=0)
                train_labels_torch=Variable(torch.from_numpy(np.array(train_labels_ae,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
                
                # Slide window across test data:
                F_test_torch_expanded = np.concatenate(list(map(f_tr, np.array(F_test_torch))), axis=0)

                # Slide window across target data to predict during test:                
                F_test_tgt_torch = F_test_torch[:,n_feat*n_predictor_bins:]
                F_test_tgt_torch_expanded = np.concatenate(list(map(f_tgt, np.array(F_test_tgt_torch))), axis=0)
                
                # Convert test data back to torch:
                F_test_torch = Variable(torch.from_numpy(F_test_torch_expanded), requires_grad=False)
                F_test_tgt_torch = Variable(torch.from_numpy(F_test_tgt_torch_expanded), requires_grad=False)
            
                # Reshape test labels:
                test_labels_ae = np.concatenate([np.matlib.repmat(row,n_offsets,1) for row in test_labels],axis=0)
                test_labels_torch=Variable(torch.from_numpy(np.array(test_labels_ae,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor                
            
            # Move variables to graphics card if requested:
            if gpu and torch.cuda.is_available():
                model = model.to('cuda')
                F_train_torch = F_train_torch.to('cuda')
                F_test_torch = F_test_torch.to('cuda')
                F_test_tgt_torch = F_test_tgt_torch.to('cuda')
                F_train_tgt_torch = F_train_tgt_torch.to('cuda')
                train_labels_torch = train_labels_torch.to('cuda')
                test_labels_torch = test_labels_torch.to('cuda')
                sig_neu = torch.tensor(sig_neu).to('cuda')
                
            # Get hidden representations before any learning:
            outp_init=model(F_test_torch,sig_neu,gpu=gpu)
            hidden_init=torch.Tensor(outp_init[1].detach()).to('cpu').numpy()
            
            # Fit autoencoder:
            start_fit_ae = time.time()
            outp_init=model(F_test_torch,sig_neu,gpu=gpu)
            curr_ae_df=fit_autoencoder(model=model,inpt_train=F_train_torch,tgt_train=F_train_tgt_torch, 
               clase_train=train_labels_torch, inpt_test=F_test_torch, 
               clase_test=test_labels_torch, n_epochs=n_epochs,batch_size=batch_size,
               lr=lr,sigma_noise=sig_neu, beta0=beta0, beta1=beta1, beta_sp=beta_sp, 
               p_norm=p_norm,xor=xor,beta_rec=beta_rec,beta_xor=beta_xor,
               chunked_rec=chunked_rec, chunk_size=n_feat, save_learning=save_learning, 
               gpu=gpu,verbose=verbose)
            stop_fit_ae = time.time()
            print('fit_autoencoder duration={}'.format(stop_fit_ae - start_fit_ae))
            
            # Get hidden and reconstructed representations:
            if save_learning:
                   
                # Test logistic regression performance on reconstructed data:            
                print('Testing classifier performance on reconstructed data...')
                
                # Iterate over tasks:
                for j in np.arange(test_labels.shape[1]):                    
                    curr_ae_df['perf_task{}_hidden'.format(j)] = [curr_ae_df.apply(lambda x : classifier(x.hidden_test[0], test_labels[:,j],1)[1], axis=1)]
                    curr_ae_df['perf_task{}_out'.format(j)] = [curr_ae_df.apply(lambda x : classifier(x.rec_test[0], test_labels[:,j],1)[1], axis=1)]
            else:
                rep_cols = ['inpt_train', 'inpt_test', 'hidden_train', 'hidden_test', 'rec_train', 'rec_test']
                for col in rep_cols:
                    curr_ae_df.loc[curr_ae_df.index[1:-1], col] = None
            
            # Rename some columns:
            if chunked_rec and rec_network_type=='prediction':
                src_cols = [x for x in curr_ae_df.columns if 'loss_rec_chunk' in x]
                for col in src_cols:
                    curr_ae_df = curr_ae_df.rename(columns={col:col.replace('chunk', 'bin')})
                    
            # Add class labels, repeat number:
            curr_ae_df['labels'] = [test_labels]*curr_ae_df.shape[0]
            curr_ae_df['repeat'] = [k]*curr_ae_df.shape[0]
            ae_df = pd.concat([ae_df, curr_ae_df], axis=0)
        
        # Split dataframe into separate rows for separate model layers:
        representation_cols = [x for x in curr_ae_df.columns if re.search('\w+_(test|train)',x) is not None]
        representation_df = curr_ae_df[representation_cols]
        layers = [x[:-6] for x in representation_df.columns if re.search('_train', x) is not None]
        representation_list = []
        for layer in layers:
            curr_cols = [x for x in representation_cols if layer in x]
            curr_representations = representation_df[curr_cols]
            curr_representations = curr_representations.rename(columns={layer+'_train':'train', layer+'_test':'test'})
            curr_representations['layer'] = layer
            curr_representations['epoch'] = curr_representations.index
            representation_list.append(curr_representations)
        representation_df = pd.concat(representation_list, axis=0)    
        representation_df.index = np.arange(representation_df.shape[0])        
        
        # Test geometry:
        if test_geometry:
            print('Testing geometry...')
            
            # Extract matrix of contacts:
            F=session2feature_array(test_session, field='features')
            F_summed=session2feature_array(test_session, field='features_bins_summed')
            
            # Only need contacts, not angles, so exclude odd columns:
            keep_columns=np.arange(0,F_summed.shape[1],2)
            F_summed=F_summed[:,keep_columns]
            
            # Binarize contacts:
            Fb=binarize_contacts(F_summed)
            
            # Decide whether to use summed or raw inputs to test geometry of input space:
            if sum_inpt:
                inpt_geo_feat=F_summed
            else:
                inpt_geo_feat=F
            
            # Test geometry iterating over subsamples to deal with any imbalances in trials per condition:
            start_measure_geo = time.time()
            tparallel_inpt_m, curr_perf_inpt, curr_geo_inpt = test_autoencoder_geometry(inpt_geo_feat, test_labels, n_geo_subsamples, geo_reg)
            stop_measure_geo = time.time()
            print('test_autoencoder_geometry duration={}'.format(stop_measure_geo - start_measure_geo))
            
            # Assign layers:
            curr_perf_inpt['layer'] = ['input']*curr_perf_inpt.shape[0]
            curr_geo_inpt['layer'] = ['input']*curr_geo_inpt.shape[0]
            
            # Aggreagate:
            curr_perf_df = pd.concat([curr_perf_df, curr_perf_inpt], axis=0)            
            curr_geo_df = pd.concat([curr_geo_df, curr_geo_inpt], axis=0)
            
            if autoencoder_params is not None:
                parallel_hidden_pre_m, curr_perf_hidden_pre, curr_geo_hidden_pre = test_autoencoder_geometry(hidden_init, np.array(test_labels_ae), n_geo_subsamples, geo_reg)
                parallel_hidden_m, curr_perf_hidden, curr_geo_hidden = test_autoencoder_geometry(hidden_rep, np.array(test_labels_ae), n_geo_subsamples, geo_reg)
                parallel_rec_m, curr_perf_rec, curr_geo_rec = test_autoencoder_geometry(rec_rep, np.array(test_labels_ae), n_geo_subsamples, geo_reg)

                # Pad dataframes of geometry results with layer:
                curr_perf_hidden_pre['layer'] = ['hidden_pre']*curr_perf_hidden_pre.shape[0]
                curr_perf_hidden['layer'] = ['hidden']*curr_perf_hidden.shape[0]
                curr_perf_rec['layer'] = ['reconstruction']*curr_perf_rec.shape[0]

                curr_perf_hidden_pre['model_type'] = [rec_network_type]*curr_perf_hidden_pre.shape[0]
                curr_perf_hidden['model_type'] = [rec_network_type]*curr_perf_hidden.shape[0]
                curr_perf_rec['model_type'] = [rec_network_type]*curr_perf_rec.shape[0]
            
                # Pad dataframes of geometry results with layer:
                curr_geo_hidden_pre['layer'] = ['hidden_pre']*curr_geo_hidden_pre.shape[0]
                curr_geo_hidden['layer'] = ['hidden']*curr_geo_hidden.shape[0]
                curr_geo_rec['layer'] = ['reconstruction']*curr_geo_rec.shape[0]

                curr_geo_hidden_pre['model_type'] = [rec_network_type]*curr_geo_hidden_pre.shape[0]
                curr_geo_hidden['model_type'] = [rec_network_type]*curr_geo_hidden.shape[0]
                curr_geo_rec['model_type'] = [rec_network_type]*curr_geo_rec.shape[0]
            
                # Aggregate:
                curr_perf_df = pd.concat([curr_perf_df, curr_perf_hidden_pre, curr_perf_hidden, curr_perf_rec], axis=0)
                curr_geo_df = pd.concat([curr_geo_df, curr_geo_hidden_pre, curr_geo_hidden, curr_geo_rec], axis=0)
                
                curr_perf_df['penalty'] = [penalty]*curr_perf_df.shape[0]
                curr_geo_df['penalty'] = [penalty]*curr_geo_df.shape[0]
                
            """
            # Plot mean data by XOR condition:
            if plot_xor and k==n_files-1:
                # Average XOR data across subsamples:
                xor_dats_inpt=np.mean(xor_dats_inpt,axis=0)
                xor_means_files.append(xor_dats_inpt)
                
                # Plot:
                xor_fig=plt.figure(figsize=(4,4))
                xor_ax=xor_fig.add_subplot(111)
                xor_ax.violinplot(xor_means_files[-1],showmeans=True)
            """

            curr_geo_df['repeat'] = [k]*curr_geo_df.shape[0]
            geo_df = pd.concat([geo_df, curr_geo_df], axis=0)
            
        curr_perf_df['repeat'] = [k]*curr_perf_df.shape[0]
        perf_df = pd.concat([perf_df, curr_perf_df],axis=0)
        
    # Add some general hyperparameters:
    if autoencoder_params is not None:
        perf_orig_df['model_type'] = [rec_network_type]*perf_orig_df.shape[0]
        ae_df['model_type'] = [rec_network_type]*ae_df.shape[0]    
        perf_df['model_type'] = [rec_network_type]*perf_df.shape[0]    
        geo_df['model_type'] = [rec_network_type]*geo_df.shape[0]
    
        if mlp_df is not None:    
            mlp_df['model_type'] = [rec_network_type]*mlp_df.shape[0]    
            
        if rec_network_type=='prediction':
    
            perf_orig_df['n_predictor_bins'] = [n_predictor_bins]*perf_orig_df.shape[0]
            ae_df['n_predictor_bins'] = [n_predictor_bins]*ae_df.shape[0]    
            perf_df['n_predictor_bins'] = [n_predictor_bins]*perf_df.shape[0]    
            geo_df['n_predictor_bins'] = [n_predictor_bins]*geo_df.shape[0]    
    
            perf_orig_df['n_predicted_bins'] = [n_predicted_bins]*perf_orig_df.shape[0]
            ae_df['n_predicted_bins'] = [n_predicted_bins]*ae_df.shape[0]    
            perf_df['n_predicted_bins'] = [n_predicted_bins]*perf_df.shape[0]    
            geo_df['n_predicted_bins'] = [n_predicted_bins]*geo_df.shape[0]    
    
            if mlp_df is not None:    
                mlp_df['n_predictor_bins'] = [n_predictor_bins]*mlp_df.shape[0]    
                mlp_df['n_predicted_bins'] = [n_predicted_bins]*mlp_df.shape[0]            
            
    # Rename tasks for performance results:
    perf_task_names = perf_df.apply(lambda x : task_strs[x.task] if x.task!='xor' else 'xor', axis=1)
    perf_df['task'] = perf_task_names
        
    # Rename dichotomies for geometry results:
    geo_df['dichotomy_idx'] = geo_df['dichotomy']    
    dich_names = geo_df.apply(lambda x : task_strs[x.dichotomy], axis=1)
    geo_df['dichotomy'] = dich_names    
    
    # Rename training partition for geometry results:
    geo_df['train_partition_inds'] = geo_df['train_partition']
    train_partition_names = geo_df.apply(lambda x : tasks[not bool(x.dichotomy_idx)][x.train_partition], axis=1)
    geo_df['train_partition'] = train_partition_names
    
    # Reindex: 
    perf_df.index = np.arange(perf_df.shape[0])
    geo_df.index = np.arange(geo_df.shape[0])
    perf_orig_df.index = np.arange(perf_orig_df.shape[0])    
    
    results = dict()
    results['perf_orig_df'] = perf_orig_df
    results['ae_df'] = ae_df
    results['perf_df'] = perf_df
    results['geo_df'] = geo_df
    results['mlp_df'] = mlp_df
    results['train_sessions'] = train_sessions
    results['test_sessions'] = test_sessions
    
    time.sleep(2)
    end_time=datetime.now()
    duration = end_time - start_time
    
    if save_perf:
        
        # Make current folder default:
        if output_directory==None:
            output_directory=os.getcwd()
            
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
            
        # Save results:
        #h5path = os.path.join(output_directory, 'iterate_autoencoder_results.h5')
        #save_ae_results(h5path,perf_orig,perf_out,perf_hidden,loss_epochs, perf_orig_mlp,task_rec,ccgp_rec,parallelism_rec,task_hidden_pre,ccgp_hidden_pre,parallelism_hidden_pre,task_hidden,ccgp_hidden,parallelism_hidden)

        results_path = os.path.join(output_directory, 'iterate_fit_autoencoder_resultss.pickle')
        pickle.dump(results, open(results_path, 'wb'))         
        
        if save_sessions and sessions==None:
            sessions_df=pd.concat(sessions, ignore_index=True)
            sessions_path=os.path.join(output_directory, 'simulated_sessions.pickle')
            pickle.dump(sessions_df, open(sessions_path, 'wb'))
        
        """
        # Save plot of means of XOR data:
        if test_geometry and plot_xor:
            xor_fig_path=os.path.join(output_directory,'xor_means.png')
            xor_fig.savefig(xor_fig_path,dpi=500)
        """
        
        # Save metadata if analysis_metadata successfully imported:
        if 'analysis_metadata' in sys.modules:
            
            # Initialize metadata object:
            M=fmt_ae_metadata(sim_params,autoencoder_params,mlp_params=mlp_params)
            
            # If loading previously-simulated session and it was passed as path,
            # add file path to metadata:
            if sessions_in!=None and type(sessions_in)==str:
                M.add_input(sessions_in)
            
            # Save autoencoder parameters if applicable:
            if autoencoder_params is not None: 
                M.add_param('model_type', rec_network_type)
                M.add_param('train_on_xor', xor)
                if xor:
                    M.add_param('beta_xor', beta_xor)                
                if rec_network_type=='prediction':
                    M.add_param('n_predictor_bins', n_predictor_bins)
                    M.add_param('n_predicted_bins', n_predicted_bins)
            
            if test_geometry:
                M.add_param('geometry_reg', geo_reg)
                M.add_param('n_geometry_subsamples', n_geo_subsamples)
            
            M.add_param('tasks', tasks)
            M.add_param('n_files', n_files)
            M.add_param('sum_inpt', sum_inpt)
            M.add_output(results_path)
            M.date=end_time.strftime('%Y-%m-%d')
            M.time=end_time.strftime('%H:%M:%S')
            M.duration=seconds_2_full_time_str(duration.seconds)
            """
            if test_geometry and plot_xor:
                M.add_output(xor_fig_path)
            """
            if save_sessions and sessions==None:
                M.add_output(sessions_path)
            metadata_path=os.path.join(output_directory, 'iterate_autoencoder_metdata.json')
            write_metadata(M, metadata_path)
    
    return results




def prep_data4ae(session, task):
    
    F=session2feature_array(session) # extract t-by-g matrix of feature data, where t is number of trials, g is total number of features (across all time bins)
    labels=session2labels(session, task) # generate vector of labels    
    return F, labels
    


def save_ae_results(fpath, perf_orig, perf_out, perf_hidden, loss_epochs, 
    perf_orig_mlp=None, task_rec=None, ccgp_rec=None, parallelism_rec=None, 
    task_hidden_pre=None, ccgp_hidden_pre=None, parallelism_pre=None, 
    task_hidden=None, ccgp_hidden=None, parallelism_hidden=None):
    """
    Save results from iterate_fit_autoencoder() to disk. 

    """
    
    with h5py.File(fpath, 'w') as hfile:
        if perf_orig is not None:
            hfile.create_dataset('perf_orig', data=perf_orig)
        if perf_out is not None:
            hfile.create_dataset('perf_out', data=perf_out)
        if perf_hidden is not None:
            hfile.create_dataset('perf_hidden', data=perf_hidden)
        if loss_epochs is not None:
            hfile.create_dataset('loss_epochs', data=loss_epochs)
        if perf_orig_mlp!=None:
            hfile.create_dataset('perf_orig_mlp', data=perf_orig_mlp)    
        if task_rec is not None:
            hfile.create_dataset('task_rec', data=task_rec)
        if ccgp_rec is not None: 
            hfile.create_dataset('ccgp_rec', data=ccgp_rec)
        if parallelism_rec is not None: 
            hfile.create_dataset('parallelism_rec', data=parallelism_rec)
        if task_hidden_pre is not None: 
            hfile.create_dataset('task_hidden_pre', data=task_hidden_pre)
        if ccgp_hidden_pre is not None:
            hfile.create_dataset('ccgp_hidden_pre', data=ccgp_hidden_pre)
        if parallelism_pre is not None:
            hfile.create_dataset('parallelism_pre', data=parallelism_pre)
        if task_hidden is not None: 
            hfile.create_dataset('task_hidden', data=task_hidden)
        if ccgp_hidden is not None:
            hfile.create_dataset('ccgp_hidden', data=ccgp_hidden)
        if parallelism_hidden is not None:
            hfile.create_dataset('parallelism_hidden', data=parallelism_hidden)

    

def fmt_ae_metadata(sim_params, autoencoder_params, mlp_params=None):
    """
    Format some metadata for iterate_fit_autoencoder.

    Parameters
    ----------
    sim_params : dict
        Same as input to iterate_fit_autoencdoer().
        
    autoencoder_params : dict
        Same as input to iterate_fit_autoencdoer().
        
    mlp_params : dict, optional
        Same as input to iterate_fit_autoencdoer().

    Returns
    -------
    M : analysis_metadata.analysis_metadata.Metadata
        Metadata object.

    """
    M=Metadata()
    
    # Write simulation parameters to metadata:
    sim_params_out=dict()
    sim_params_out['concavity']=sim_params['concavity']
    sim_params_out['n_whisk']=int(sim_params['n_whisk'])
    sim_params_out['prob_poiss']=float(sim_params['prob_poiss'])
    sim_params_out['noise_w']=float(sim_params['noise_w'])
    sim_params_out['spread']=sim_params['spread']  
    sim_params_out['speed']=float(sim_params['speed'])  
    sim_params_out['ini_phase_m']=float(sim_params['ini_phase_m'])
    sim_params_out['ini_phase_spr']=float(sim_params['ini_phase_spr'])
    sim_params_out['delay_time']=float(sim_params['delay_time'])
    sim_params_out['freq_m']=float(sim_params['freq_m'])
    sim_params_out['freq_std']=float(sim_params['freq_std'])            
    sim_params_out['t_total']=float(sim_params['t_total'])
    sim_params_out['dt']=float(sim_params['dt'])            
    sim_params_out['dx']=float(sim_params['dx'])            
    sim_params_out['n_trials_pre']=int(sim_params['n_trials_pre'])
    sim_params_out['amp']=float(sim_params['amp'])            
    sim_params_out['freq_sh']=sim_params['freq_sh']
    sim_params_out['z1']=sim_params['z1']
    sim_params_out['disp']=sim_params['disp']
    sim_params_out['theta']=sim_params['theta']
    sim_params_out['steps_mov']=sim_params['steps_mov']
    sim_params_out['rad_vec']=sim_params['rad_vec']
    sim_params_out['init_position']=sim_params['init_position']
    M.add_param('sim_params', sim_params_out)

    # Write autoencoder hyperparameters to metadata:
    if autoencoder_params is not None:
        autoencoder_params_out=dict()
        if type(autoencoder_params['n_hidden'])!=list and type(autoencoder_params['n_hidden'])!=np.ndarray:
            autoencoder_params_out['n_hidden']=int(autoencoder_params['n_hidden'])
        else:
            autoencoder_params_out['n_hidden']=autoencoder_params['n_hidden']
        autoencoder_params_out['model_type']=autoencoder_params['type']            
        autoencoder_params_out['sig_init']=float(autoencoder_params['sig_init'])            
        autoencoder_params_out['sig_neu']=float(autoencoder_params['sig_neu'])                        
        autoencoder_params_out['lr']=float(autoencoder_params['lr'])                        
        autoencoder_params_out['beta0']=float(autoencoder_params['beta0'])
        autoencoder_params_out['beta1']=float(autoencoder_params['beta1'])
        autoencoder_params_out['beta_xor']=float(autoencoder_params['beta_xor'])
        autoencoder_params_out['n_epochs']=int(autoencoder_params['n_epochs'])                        
        autoencoder_params_out['batch_size']=int(autoencoder_params['batch_size'])                        
        autoencoder_params_out['beta_sp']=float(autoencoder_params['beta_sp'])                                    
        autoencoder_params_out['p_norm']=float(autoencoder_params['p_norm']) 
        if autoencoder_params['type']=='prediction':
            autoencoder_params_out['n_predictor_bins'] = int(autoencoder_params['n_predictor_bins'])
            autoencoder_params_out['n_predicted_bins'] = int(autoencoder_params['n_predicted_bins'])
    else:
        autoencoder_params_out = None
    M.add_param('autoencoder_params', autoencoder_params_out)
    
    # Write MLP hyperparameters to metadata if necessary:
    if mlp_params is not None:
        mlp_params_out=dict()
        mlp_params_out['hidden_layer_sizes']=mlp_params['hidden_layer_sizes']
        mlp_params_out['activation']=mlp_params['activation']
        mlp_params_out['alpha']=float(mlp_params['alpha'])
        mlp_params_out['solver']=mlp_params['solver']
        mlp_params_out['learning_rate']=float(mlp_params['learning_rate'])
        mlp_params_out['learning_rate_init']=float(mlp_params['learning_rate_init'])
        M.add_param('mlp_params', mlp_params_out)
    
    return M



def test_autoencoder_geometry(feat_decod, feat_binary, n_subsamples, reg):
    """
    Test geometry over multiple data subsamples; use to control for any 
    imbalances in trials per condition.

    Parameters
    ----------
    feat_decod : array-like
        t-by-f matrix to decode binary variables from, where t is the number of
        trials and f is the number of input features.
        
    feat_binary : array-like
        t-by-2 binary matrix, where t is the number of trials.
        
    n_subsamples : int
        Number of subsamples to iterate over.

    Returns
    -------
    task_m : numpy.ndarray
        3-by-2 array of task performance results averaged across subsamples; 
        same format as corresponding output of geometry_2D() function, but 
        averaged across subsamples.
        
    ccgp_m : numpy.ndarray
        2-by-2-by-2 array of CCGP results averaged across subsamples; same 
        format as corresponding output of geometry_2D() function, but averaged 
        across subsamples.

    """
    
    # Initialize arrays of results
    task_total=np.empty((n_subsamples,3,2)) #task performance
    ccgp_total=np.empty((n_subsamples,2,2,2)) #ccgp
    parallelism_total=np.empty((n_subsamples,2))
    
    # Find minimum number of trials per condition:
    bin_conditions=find_matching_2d_bin_trials(feat_binary)
    min_n=min([x['count'] for x in bin_conditions])
    
    xor_dats=[]
    
    # Iterate over subsamples
    perf_df_all = pd.DataFrame()
    geo_df_all = pd.DataFrame()
    for s in np.arange(n_subsamples):
        
        # Select current subsample:
        curr_subsample_indices=subsample_2d_bin(bin_conditions, min_n)
        feat_binary_subsample=feat_binary[curr_subsample_indices]
        feat_decod_subsample=feat_decod[curr_subsample_indices]
        
        # Test geometry:
        perf_tasks, perf_ccgp, parallel, xor_dat, perf_df, geo_df= geometry_2D(feat_decod_subsample,feat_binary_subsample,reg) # on reconstruction
        xor_dats.append(xor_dat)

        task_total[s,:,:]=perf_tasks
        ccgp_total[s,:,:,:]=perf_ccgp
        parallelism_total[s,:]=parallel            
        
        # Add some parameters:
        perf_df['subsample'] = [s]*perf_df.shape[0]
        perf_df['trial_indices'] = [curr_subsample_indices]*perf_df.shape[0]
            
        geo_df['subsample'] = [s]*geo_df.shape[0]
        geo_df['trial_indices'] = [curr_subsample_indices]*geo_df.shape[0]
        
        # Merge current geometry results with overall dataframe:
        perf_df_all = pd.concat([perf_df_all, perf_df],axis=0)
        geo_df_all = pd.concat([geo_df_all, geo_df],axis=0)
    
    # Average across subsamples:
    task_m=np.mean(task_total,axis=0)
    ccgp_m=np.mean(ccgp_total,axis=0)
    parallel_m=np.mean(parallelism_total,axis=0)
    xor_dats=np.array(xor_dats)
    
    return perf_df_all, geo_df_all
    


def ae_dispatch(n_inp,n_hidden,sigma_init,k=[2,2],xor=False):
    if type(n_hidden)!=list and type(n_hidden)!=np.ndarray:
        ae=sparse_autoencoder_1(n_inp,n_hidden,sigma_init,k=k,xor=xor)
    elif len(n_hidden)==2:
        ae=sparse_autoencoder_2(n_inp,n_hidden,sigma_init,k=k,xor=xor)
    elif len(n_hidden)==3:
        ae=sparse_autoencoder_3(n_inp,n_hidden,sigma_init,k=k,xor=xor)
    else:
        raise AssertionError('Invalid number of hidden layers; please select number of hidden layers from 1-3.') 
    return ae



def boxcar(X, width, n_shifts, stride=1):
    """
    Return the contents of a sliding window across a 1-dimensional array.

    Parameters
    ----------
    X : array-like
        m-element array.
        
    width : int
        Window size.
        
    n_shifts : int
        Number of different offsets by which to slide window across input matrix.
        
    stride : int
        Spacing between offsets.

    Returns
    -------
    Z : numpy.ndarray
        b-by-w output matrix, where b is the number of offsets and w is the 
        window size. Each row of Z represents the contents of a sliding window
        across input array X. If row i of Z is equivalent to elements k:k+w of X, 
        then row i+1 is equivalent to elements k+s:k+w+s, where s equals the stride. 
        In other words still, for any k and offset i between 0 and s, X[k*s+i:k*s+i+w] 
        is equivalent to Z[k+i,:]. 

    """
    # TODO: Add validation to ensure sliding window doesn't overrun input matrix dimensions.
    Y = [[X[i:i+width]] for i in np.arange(0, n_shifts*stride, stride)]
    Z = np.concatenate(Y, axis=0)
    return Z



def roll_mask(X, width, n_shifts, stride=1, mask_val=0):
    n = len(X)
    Y = [np.roll(np.concatenate([X[i:i+width], mask_val*np.ones(n-width)]),i) for i in np.arange(0, n_shifts*stride, stride)]
    Z = np.array(Y)
    return Z



# Autoencoder Architecture
class sparse_autoencoder(nn.Module):
    def __init__(self,n_inp,sigma_init,k=[2,2],xor=False):    
        super(sparse_autoencoder,self).__init__()
        self.n_inp=n_inp
        self.sigma_init=sigma_init       
        self.k=k
        self.xor=xor
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.sigma_init)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=self.sigma_init)



class sparse_autoencoder_1(sparse_autoencoder):
    def __init__(self,n_inp,n_hidden,sigma_init,k=[2,2],xor=False):
        super(sparse_autoencoder_1,self).__init__(n_inp,sigma_init,k=k,xor=xor)
        self.n_hidden=n_hidden
        self.enc=torch.nn.Linear(n_inp,n_hidden)
        self.dec=torch.nn.Linear(n_hidden,n_inp)
        self.dec2=torch.nn.Linear(n_hidden,self.k[0])
        self.dec3=torch.nn.Linear(n_hidden,self.k[1])
        if xor:
            self.dec4=torch.nn.Linear(n_hidden,2) # XOR
        
    def forward(self,x,sigma_noise,gpu=False):
        if not gpu:
            x_hidden = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden)
        else:
            x_hidden = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden).to('cuda')
        x = self.dec(x_hidden)
        x2 = self.dec2(x_hidden)
        x3 = self.dec3(x_hidden)
        if self.xor:    
            x4 = self.dec4(x_hidden)
            return x,x_hidden,x2,x3,x4
        else:
            return x,x_hidden,x2,x3



class sparse_autoencoder_2(sparse_autoencoder):
    def __init__(self,n_inp,n_hidden,sigma_init,k=[2,2],xor=False):
        super(sparse_autoencoder_2,self).__init__(n_inp,sigma_init,k=k,xor=xor)
        self.n_inp=n_inp
        self.n_hidden=n_hidden
        self.sigma_init=sigma_init
        self.enc=torch.nn.Linear(n_inp,n_hidden[0])
        self.h0=torch.nn.Linear(n_hidden[0],n_hidden[1])
        self.dec=torch.nn.Linear(n_hidden[1],n_inp)
        self.dec2=torch.nn.Linear(n_hidden[1],self.k[0])
        self.dec3=torch.nn.Linear(n_hidden[1],self.k[1])
        if xor:
            self.dec4=torch.nn.Linear(n_hidden[1],2) # XOR
        
    def forward(self,x,sigma_noise,gpu=False):
        x_hidden0 = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden[0])
        if gpu:
            x_hidden0 = x_hidden0.to('cuda')
        x_hidden1 = F.relu(self.h0(x_hidden0))+sigma_noise*torch.randn(x_hidden0.size(0),self.n_hidden[1])
        x = self.dec(x_hidden1)
        x2 = self.dec2(x_hidden1)
        x3 = self.dec3(x_hidden1)
        if self.xor:    
            x4 = self.dec4(x_hidden1)
            return x,x_hidden1,x2,x3,x4
        else:
            return x,x_hidden1,x2,x3



class sparse_autoencoder_3(sparse_autoencoder):
    def __init__(self,n_inp,n_hidden,sigma_init,k=[2,2],xor=False):
        super(sparse_autoencoder_3,self).__init__(n_inp,sigma_init,k=k,xor=xor)
        self.n_inp=n_inp
        self.n_hidden=n_hidden
        self.sigma_init=sigma_init
        self.enc=torch.nn.Linear(n_inp,n_hidden[0])
        self.h0=torch.nn.Linear(n_hidden[0],n_hidden[1])
        self.h1=torch.nn.Linear(n_hidden[1],n_hidden[2])
        self.dec=torch.nn.Linear(n_hidden[2],n_inp)
        self.dec2=torch.nn.Linear(n_hidden[2],self.k[0])
        self.dec3=torch.nn.Linear(n_hidden[2],self.k[1])
        if xor:
            self.dec4=torch.nn.Linear(n_hidden[2],2) # XOR
        
    def forward(self,x,sigma_noise,gpu=False):
        x_hidden0 = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden[0])
        if gpu:
            x_hidden0 = x_hidden0.to('cuda')
        x_hidden1 = F.relu(self.h0(x_hidden0))+sigma_noise*torch.randn(x_hidden0.size(0),self.n_hidden[1])
        x_hidden2 = F.relu(self.h1(x_hidden1))+sigma_noise*torch.randn(x_hidden1.size(0),self.n_hidden[2])
        x = self.dec(x_hidden2)
        x2 = self.dec2(x_hidden2)
        x3 = self.dec3(x_hidden2)
        if self.xor:    
            x4 = self.dec4(x_hidden2)
            return x,x_hidden2,x2,x3,x4
        else:
            return x,x_hidden2,x2,x3



class prediction_network(nn.Module):
    def __init__(self,n_inp,n_hidden,n_out,sigma_init,k=[2,2],xor=False):
        super(prediction_network,self).__init__()
        self.n_inp=n_inp
        self.n_hidden=n_hidden
        self.n_out=n_out
        self.sigma_init=sigma_init       
        self.k=k
        self.xor=xor
        self.enc=torch.nn.Linear(n_inp,n_hidden)
        self.dec=torch.nn.Linear(n_hidden,n_out)
        self.dec2=torch.nn.Linear(n_hidden,self.k[0])
        self.dec3=torch.nn.Linear(n_hidden,self.k[1])
        if xor:
            self.dec4=torch.nn.Linear(n_hidden,2) # XOR
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.sigma_init)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=self.sigma_init)

    def forward(self,x,sigma_noise,gpu=False):
        if not gpu:
            x_hidden = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden)
        else:
            x_hidden = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden).to('cuda')
        x = self.dec(x_hidden)
        x2 = self.dec2(x_hidden)
        x3 = self.dec3(x_hidden)
        if self.xor:    
            x4 = self.dec4(x_hidden)
            return x,x_hidden,x2,x3,x4
        else:
            return x,x_hidden,x2,x3



def sparsity_loss(data,p):
    #shap=data.size()
    #nt=shap[0]*shap[1]
    #loss=(1/nt)*torch.norm(data,p)
    #loss=torch.norm(data,p)
    #loss=torch.mean(torch.sigmoid(100*(data-0.1)),axis=(0,1))
    loss=torch.mean(torch.pow(abs(data),p),axis=(0,1))
    return loss


