import os
import sys
from datetime import datetime
import pathlib
import h5py
import pickle
import numpy as np
import pandas as pd
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
def fit_autoencoder(model,data_train,clase_train,data_test,clase_test,n_epochs,batch_size,lr,sigma_noise,beta0,beta1,beta_sp,p_norm,xor=False,beta_xor=0,save_learning=True,verbose=False):
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
    beta_rec=1-beta0-beta1-beta_xor
    
    train_trial_indices=torch.Tensor(np.arange(len(clase_train)))
    train_loader=DataLoader(torch.utils.data.TensorDataset(data_train,data_train,train_trial_indices),batch_size=batch_size,shuffle=True)

    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    loss_rec=torch.nn.MSELoss()
    loss_ce0=torch.nn.CrossEntropyLoss()
    loss_ce1=torch.nn.CrossEntropyLoss()
    if xor:
        loss_xor=torch.nn.CrossEntropyLoss()
    model.train()
    
    n_trials_train=len(clase_train)
    n_trials_test=len(clase_test)
    n_input_features=data_train.shape[1]
    n_hidden=model.enc.out_features
    
    results=dict()
    results['loss_rec_vec']=np.empty(n_epochs,dtype=np.float32); results['loss_rec_vec'][:]=np.nan
    results['loss_ce_vec']=np.empty(n_epochs,dtype=np.float32); results['loss_ce_vec'][:]=np.nan     
    results['loss_sp_vec']=np.empty(n_epochs,dtype=np.float32); results['loss_sp_vec'][:]=np.nan
    results['loss_vec']=np.empty(n_epochs,dtype=np.float32); results['loss_vec'][:]=np.nan
    
    if save_learning:
        results['data_epochs_train']=np.empty((n_epochs, n_trials_train, n_input_features),dtype=np.float32);
        results['data_hidden_train']=np.empty((n_epochs, n_trials_train, n_hidden),dtype=np.float32);
        results['data_epochs_test']=np.empty((n_epochs, n_trials_test, n_input_features),dtype=np.float32);
        results['data_hidden_test']=np.empty((n_epochs, n_trials_test, n_hidden),dtype=np.float32);
    else:
        results['data_epochs_train']=np.empty((n_trials_train, n_input_features),dtype=np.float32);
        results['data_hidden_train']=np.empty((n_trials_train, n_hidden),dtype=np.float32);
        results['data_epochs_test']=np.empty((n_trials_test, n_input_features),dtype=np.float32);
        results['data_hidden_test']=np.empty((n_trials_test, n_hidden),dtype=np.float32);        

    t=0
    while t<n_epochs: 
        #print (t)
        
        # Compute loss, generate hidden and output representations using training trials:
        outp_train=model(data_train,sigma_noise)
        
        # Save 
        if save_learning:
            results['data_epochs_train'][t]=outp_train[0].detach().numpy()
            results['data_hidden_train'][t]=outp_train[1].detach().numpy()
            
        # Compute non-CE terms of loss function:
        curr_loss_rec=loss_rec(outp_train[0],data_train).item()
        curr_loss_sp=sparsity_loss(outp_train[1],p_norm).item()
        
        # Compute CE terms of loss function:
        curr_loss_ce0=loss_ce0(outp_train[2],clase_train[:,0]).item()
        curr_loss_ce1=loss_ce1(outp_train[3],clase_train[:,1]).item()
        
        if xor:
            xor_labels=np.sum(np.array(clase_train),axis=1)%2 # Define the XOR function wrt to the two variables
            xor_labels=Variable(torch.from_numpy(np.array(xor_labels,dtype=np.int64)),requires_grad=False)
            curr_loss_xor=loss_xor(outp_train[4],xor_labels).item()
        else:
            curr_loss_xor=0
        
        curr_loss_ce_total=beta0*curr_loss_ce0+beta1*curr_loss_ce1
        curr_loss_total=(beta_rec*curr_loss_rec+curr_loss_ce_total+beta_xor*curr_loss_xor+beta_sp*curr_loss_sp)

        results['loss_rec_vec'][t]=curr_loss_rec
        results['loss_ce_vec'][t]=curr_loss_ce_total
        results['loss_sp_vec'][t]=curr_loss_sp
        results['loss_vec'][t]=curr_loss_total
        
        # Generate hidden and output layer representations of held-out trials: 
        outp_test=model(data_test,sigma_noise)
        if save_learning:
            results['data_epochs_test'][t]=outp_test[0].detach().numpy()
            results['data_hidden_test'][t]=outp_test[1].detach().numpy()

        #if verbose and t%10==0:
        #    print('Running autoencoder training epoch {} out of {}...'.format(t+1,n_epochs))
        
        if t==0 or t==(n_epochs-1):
            print (t,'rec ',curr_loss_rec,'ce ',curr_loss_ce_total,'sp ',curr_loss_sp,'total ',curr_loss_total)
        for batch_idx, (targ1, targ2, trial_indices) in enumerate(train_loader):
           
            optimizer.zero_grad()
            output=model(targ1,sigma_noise)

            loss_r=loss_rec(output[0],targ2) # reconstruction error
            
            trial_indices=trial_indices.type(torch.long) # need to do some annoying reformatting to get tensor to work as array of indices
            
            curr_task0_labels=clase_train[trial_indices,0]
            loss_cla0=loss_ce0(output[2],curr_task0_labels) # cross entropy error
            
            curr_task1_labels=clase_train[trial_indices,1]
            loss_cla1=loss_ce1(output[3],curr_task1_labels) # cross entropy error
            
            # compute xor cross-entropy if requested:
            if xor:
                curr_xor_labels=xor_labels[trial_indices]
                loss_x=loss_xor(output[4],curr_xor_labels)
            else:
                loss_x=0
            
            loss_s=sparsity_loss(output[1],p_norm)
            loss_t=(beta_rec*loss_r+beta0*loss_cla0+beta1*loss_cla1+beta_xor*loss_x+beta_sp*loss_s)

            loss_t.backward() # compute gradient
            optimizer.step() # weight update
            
        t=(t+1)
    model.eval()
    
    if not save_learning:
        results['data_epochs_train']=outp_train[0].detach().numpy()
        results['data_hidden_train']=outp_train[1].detach().numpy()
        results['data_epochs_test']=outp_test[0].detach().numpy()
        results['data_hidden_test']=outp_test[1].detach().numpy()                
    
    return results



def iterate_fit_autoencoder(sim_params, tasks, n_files, autoencoder_params=None, mlp_params=None, save_learning=True, test_geometry=True, n_geo_subsamples=10, geo_reg=1.0, xor=False, sum_inpt=True, sessions_in=None, save_perf=False, save_sessions=False, plot_xor=False, output_directory=None, verbose=False):
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
    
    # Unpack some autoencoder parameters:
    if autoencoder_params is not None:
        n_hidden=autoencoder_params['n_hidden']
        if type(n_hidden)!=list and type(n_hidden)!=np.ndarray:
            n_hidden=int(n_hidden)
        sig_init=float(autoencoder_params['sig_init'])
        sig_neu=float(autoencoder_params['sig_neu'])
        lr=float(autoencoder_params['lr'])
        beta0=float(autoencoder_params['beta0'])
        beta1=float(autoencoder_params['beta1'])
        if xor:
            beta_xor=float(autoencoder_params['beta_xor'])
        else:
            beta_xor=0
        beta_sp=float(autoencoder_params['beta_sp'])
        p_norm=float(autoencoder_params['p_norm'])
        
        # Verify that betas sum to <= 1:
        if beta0+beta1+beta_xor > 1:
            raise ValueError('beta0 + beta1 greater than 1; please ensure beta0 + beta1 <= 1.')
        
        # Unpack some batching parameters:
        batch_size=int(autoencoder_params['batch_size'])
        n_epochs=int(autoencoder_params['n_epochs'])
        
        # Initialize output arrays:
        perf_orig=np.zeros((n_files,2,2))
        if save_learning:
            perf_out=np.zeros((n_files,n_epochs,2))
            perf_hidden=np.zeros((n_files,n_epochs,2))
            loss_epochs=np.zeros((n_files,n_epochs))
        
    if autoencoder_params is None or not save_learning:
            perf_out=None
            perf_hidden=None
            loss_epochs=None
        
    task_inpt=np.zeros((n_files,3,2))    
    ccgp_inpt=np.zeros((n_files,2,2,2))
    parallelism_inpt=np.zeros((n_files,2))        

    if autoencoder_params is not None:
        task_hidden_pre=np.zeros((n_files,3,2))    
        ccgp_hidden_pre=np.zeros((n_files,2,2,2))
        parallelism_hidden_pre=np.zeros((n_files,2))
        
        task_hidden=np.zeros((n_files,3,2))    
        ccgp_hidden=np.zeros((n_files,2,2,2))
        parallelism_hidden=np.zeros((n_files,2))    
        
        task_rec=np.zeros((n_files,3,2))    
        ccgp_rec=np.zeros((n_files,2,2,2))
        parallelism_rec=np.zeros((n_files,2))
        xor_means_files=[] # will be used for plotting means of XOR task
    else:
        task_hidden_pre=None
        ccgp_hidden_pre=None
        parallelism_hidden_pre=None

        task_hidden=None
        ccgp_hidden=None
        parallelism_hidden=None
        
        task_rec=None
        ccgp_rec=None
        ccgp_rec=None
        parallelism_rec=None
    
    # If also running MLP:
    if mlp_params!=None:
        perf_orig_mlp=np.zeros((n_files,2))
        mlp_hidden_layer_sizes=mlp_params['hidden_layer_sizes']
        mlp_activation=mlp_params['activation']        
        mlp_alpha=mlp_params['alpha']        
        mlp_solver=mlp_params['solver']        
        mlp_lr=mlp_params['learning_rate']        
        mlp_lr_init=mlp_params['learning_rate_init']
    else:
        perf_orig_mlp=None
        
    # Load previously-simulated whisker data if requested:
    if sessions_in!=None:
        save_sessions=False # no need to re-save whisker simulation if loading from disk in the first place
        sessions=load_simulation(sessions_in)
        n_files=len(np.unique(sessions.file_idx))
    # If not loading previously-run whisker simulation and save_sessions is True: 
    elif save_sessions:
        train_sessions=[]
        test_sessions=[]

    for k in range(n_files):
        print('Running file {} out of {}...'.format(k+1,n_files))
        # Simulate session (if not loading previously-simulated session):
        if sessions_in==None:
            
            # Generate session for training autoencoder:
            print('Simulating whisker contact data...')
            train_session=simulate_session(sim_params, sum_bins=True)
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
        F_train_torch=Variable(torch.from_numpy(np.array(F_train,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor
        train_labels=np.array([train_labels0,train_labels1])
        train_labels=np.transpose(train_labels)
        train_labels_torch=Variable(torch.from_numpy(np.array(train_labels,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
    
        # Prepare stimulated trial data for *testing* autoencoder:
        F_test, test_labels0=prep_data4ae(test_session, tasks[0])
        F_test, test_labels1=prep_data4ae(test_session, tasks[1])
        F_test_torch=Variable(torch.from_numpy(np.array(F_test,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor
        test_labels=np.array([test_labels0,test_labels1])
        test_labels=np.transpose(test_labels)
        test_labels_torch=Variable(torch.from_numpy(np.array(test_labels,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
            
        # Test logistic regression performance on original data:
        perf_orig[k,0]=classifier(F_test,test_labels[:,0],1, 'logistic')
        perf_orig[k,1]=classifier(F_test,test_labels[:,1],1, 'logistic')
        
        # Test MLP if requested:
        if mlp_params!=None:
            perf_orig_mlp[k]=classifier(F_test,test_labels,model='mlp', hidden_layer_sizes=mlp_hidden_layer_sizes, activation=mlp_activation, solver=mlp_solver, reg=mlp_alpha, lr=mlp_lr, lr_init=mlp_lr_init)    
        
        # Initialize task-optimized autoencoder:
        if autoencoder_params is not None:
            print('Fitting autoencoder...')
            n_inp=F_train.shape[1]
            n_labels_task0=len(np.unique(train_labels[:,0]))
            n_labels_task1=len(np.unique(train_labels[:,1]))
            model=ae_dispatch(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init,k=[n_labels_task0,n_labels_task1],xor=xor) 
            
            # Get control hidden representations before any learning:
            outp_init=model(F_test_torch,sig_neu)
            hidden_init=outp_init[1].detach().numpy()
            
            # Fit autoencoder:
            ae=fit_autoencoder(model=model,data_train=F_train_torch, clase_train=train_labels_torch, data_test=F_test_torch, clase_test=test_labels_torch, n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu, beta0=beta0, beta1=beta1, beta_sp=beta_sp, p_norm=p_norm,xor=xor,beta_xor=beta_xor,save_learning=save_learning, verbose=verbose)
                
            # Get hidden and reconstructed representations:
            if save_learning:
                loss_epochs[k]=ae['loss_vec']
                hidden_rep=ae['data_hidden_test'][-1]
                rec_rep=ae['data_epochs_test'][-1]
                   
                # Test logistic regression performance on reconstructed data:            
                print('Testing classifier performance on reconstructed data...')
                for i in range(n_epochs):
                    perf_out[k,i]=classifier(ae['data_epochs_test'][i],test_labels,1)
                    perf_hidden[k,i]=classifier(ae['data_hidden_test'][i],test_labels,1)
            else:
                hidden_rep=ae['data_hidden_test']
                rec_rep=ae['data_epochs_test']
        
        # Test geometry if requested:
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
            task_inpt_m, ccgp_inpt_m, parallel_inpt_m  = test_autoencoder_geometry(inpt_geo_feat, test_labels, n_geo_subsamples, geo_reg)
            if autoencoder_params is not None:
                task_hidden_pre_m, ccgp_hidden_pre_m, parallel_hidden_pre_m  = test_autoencoder_geometry(hidden_init, test_labels, n_geo_subsamples, geo_reg)
                task_hidden_m, ccgp_hidden_m, parallel_hidden_m  = test_autoencoder_geometry(hidden_rep, test_labels, n_geo_subsamples, geo_reg)
                task_rec_m, ccgp_rec_m, parallel_rec_m  = test_autoencoder_geometry(rec_rep, test_labels, n_geo_subsamples, geo_reg)
            
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
                
            # Write results to output array:
            task_inpt[k]=task_inpt_m
            ccgp_inpt[k]=ccgp_inpt_m
            parallelism_inpt[k]=parallel_inpt_m

            if autoencoder_params is not None:            
                task_hidden_pre[k]=task_hidden_pre_m
                ccgp_hidden_pre[k]=ccgp_hidden_pre_m
                parallelism_hidden_pre[k]=parallel_hidden_pre_m
                
                task_hidden[k]=task_hidden_m
                ccgp_hidden[k]=ccgp_hidden_m
                parallelism_hidden[k]=parallel_hidden_m
                
                task_rec[k]=task_rec_m
                ccgp_rec[k]=ccgp_rec_m
                parallelism_rec[k]=parallel_rec_m
            
        else:
            task_rec_m=None
            ccgp_rec_m=None
            parallel_rec_m=None
            task_hidden_m=None
            ccgp_hidden_m=None
            parallel_hidden_m=None
            
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
            
        # Save HDF5 of results:
        h5path = os.path.join(output_directory, 'iterate_autoencoder_results.h5')
        save_ae_results(h5path,perf_orig,perf_out,perf_hidden,loss_epochs, perf_orig_mlp,task_rec,ccgp_rec,parallelism_rec,task_hidden_pre,ccgp_hidden_pre,parallelism_hidden_pre,task_hidden,ccgp_hidden,parallelism_hidden)
        
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
            M.add_param('geometry_reg', geo_reg)
            M.add_param('n_geometry_subsamples', n_geo_subsamples)
            
            # If loading previously-simulated session and it was passed as path,
            # add file path to metadata:
            if sessions_in!=None and type(sessions_in)==str:
                M.add_input(sessions_in)
            
            # Add misc.:
            M.add_param('train_on_xor', xor)
            if xor:
                M.add_param('beta_xor', beta_xor)                
            M.add_param('tasks', tasks)
            M.add_param('n_files', n_files)
            M.add_param('sum_inpt', sum_inpt)
            M.date=end_time.strftime('%Y-%m-%d')
            M.time=end_time.strftime('%H:%M:%S')
            M.duration=seconds_2_full_time_str(duration.seconds)
            M.add_output(h5path)
            """
            if test_geometry and plot_xor:
                M.add_output(xor_fig_path)
            """
            if save_sessions and sessions==None:
                M.add_output(sessions_path)
            metadata_path=os.path.join(output_directory, 'iterate_autoencoder_metdata.json')
            write_metadata(M, metadata_path)
    
    results=dict()
    results['perf_orig']=perf_orig
    if save_learning:
        results['perf_out']=perf_out
        results['perf_hidden']=perf_hidden
        results['loss_epochs']=loss_epochs
    if mlp_params!=None:
        results['perf_orig_mlp']=perf_orig_mlp
    if test_geometry:
        
        results['task_inpt']=task_inpt
        results['ccgp_inpt']=ccgp_inpt
        results['parallelism_inpt']=parallelism_inpt
        
        results['task_hidden_pre']=task_hidden_pre
        results['ccgp_hidden_pre']=ccgp_hidden_pre
        results['parallelism_hidden_pre']=parallelism_hidden_pre
        
        results['task_hidden']=task_hidden
        results['ccgp_hidden']=ccgp_hidden
        results['parallelism_hidden']=parallelism_hidden
        
        results['task_rec']=task_rec
        results['ccgp_rec']=ccgp_rec
        results['parallelism_rec']=parallelism_rec
    
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
        autoencoder_params_out['sig_init']=float(autoencoder_params['sig_init'])            
        autoencoder_params_out['sig_neu']=float(autoencoder_params['sig_neu'])                        
        autoencoder_params_out['lr']=float(autoencoder_params['lr'])                        
        autoencoder_params_out['beta0']=float(autoencoder_params['beta0'])
        autoencoder_params_out['beta1']=float(autoencoder_params['beta1'])
        autoencoder_params_out['n_epochs']=int(autoencoder_params['n_epochs'])                        
        autoencoder_params_out['batch_size']=int(autoencoder_params['batch_size'])                        
        autoencoder_params_out['beta_sp']=float(autoencoder_params['beta_sp'])                                    
        autoencoder_params_out['p_norm']=float(autoencoder_params['p_norm']) 
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
    for s in np.arange(n_subsamples):
        
        # Select current subsample:
        curr_subsample_indices=subsample_2d_bin(bin_conditions, min_n)
        feat_binary_subsample=feat_binary[curr_subsample_indices]
        feat_decod_subsample=feat_decod[curr_subsample_indices]
        
        # Test geometry:
        perf_tasks, perf_ccgp, parallel, xor_dat = geometry_2D(feat_decod_subsample,feat_binary_subsample,reg) # on reconstruction
        xor_dats.append(xor_dat)

        task_total[s,:,:]=perf_tasks
        ccgp_total[s,:,:,:]=perf_ccgp
        parallelism_total[s,:]=parallel            
    
    # Average across subsamples:
    task_m=np.mean(task_total,axis=0)
    ccgp_m=np.mean(ccgp_total,axis=0)
    parallel_m=np.mean(parallelism_total,axis=0)
    xor_dats=np.array(xor_dats)
    
    return task_m, ccgp_m, parallel_m
    


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
        
    def forward(self,x,sigma_noise):
        x_hidden = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden)
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
        
    def forward(self,x,sigma_noise):
        x_hidden0 = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden[0])
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
        
    def forward(self,x,sigma_noise):
        x_hidden0 = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden[0])
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



def sparsity_loss(data,p):
    #shap=data.size()
    #nt=shap[0]*shap[1]
    #loss=(1/nt)*torch.norm(data,p)
    #loss=torch.norm(data,p)
    #loss=torch.mean(torch.sigmoid(100*(data-0.1)),axis=(0,1))
    loss=torch.mean(torch.pow(abs(data),p),axis=(0,1))
    return loss


