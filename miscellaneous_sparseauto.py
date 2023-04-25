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
def fit_autoencoder(model,data_train,clase_train,data_test,clase_test,n_epochs,batch_size,lr,sigma_noise,beta,beta_sp,p_norm,save_learning=True,verbose=False):
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
    
    train_loader=DataLoader(torch.utils.data.TensorDataset(data_train,data_train,clase_train),batch_size=batch_size,shuffle=True)

    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    loss1=torch.nn.MSELoss()
    loss2=torch.nn.CrossEntropyLoss()
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
        if verbose and t%10==0:
            print('Running autoencoder training epoch {} out of {}...'.format(t+1,n_epochs))
        
        # Compute loss, generate hidden and output representations using training trials:
        outp_train=model(data_train,sigma_noise)
        if save_learning:
            results['data_epochs_train'][t]=outp_train[0].detach().numpy()
            results['data_hidden_train'][t]=outp_train[1].detach().numpy()
        loss_rec=loss1(outp_train[0],data_train).item()
        loss_ce=loss2(outp_train[2],clase_train).item()
        loss_sp=sparsity_loss(outp_train[2],p_norm).item()
        loss_total=((1-beta)*loss_rec+beta*loss_ce+beta_sp*loss_sp)
        results['loss_rec_vec'][t]=loss_rec
        results['loss_ce_vec'][t]=loss_ce
        results['loss_sp_vec'][t]=loss_sp
        results['loss_vec'][t]=loss_total
        
        # Generate hidden and output layer representations of held-out trials: 
        outp_test=model(data_test,sigma_noise)
        if save_learning:
            results['data_epochs_test'][t]=outp_test[0].detach().numpy()
            results['data_hidden_test'][t]=outp_test[1].detach().numpy()
        
        if t==0 or t==(n_epochs-1):
            print (t,'rec ',loss_rec,'ce ',loss_ce,'sp ',loss_sp,'total ',loss_total)
        for batch_idx, (targ1, targ2, cla) in enumerate(train_loader):
            optimizer.zero_grad()
            output=model(targ1,sigma_noise)
            loss_r=loss1(output[0],targ2) # reconstruction error
            loss_cla=loss2(output[2],cla) # cross entropy error
            loss_s=sparsity_loss(output[2],p_norm)
            loss_t=((1-beta)*loss_r+beta*loss_cla+beta_sp*loss_s)
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



def iterate_fit_autoencoder(sim_params, autoencoder_params, task, n_files, mlp_params=None, save_learning=True, test_geometry=True, n_geo_subsamples=10, geo_reg=1.0, sessions_in=None, save_perf=False, save_sessions=False, plot_xor=False, output_directory=None, verbose=False):
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
    n_hidden=int(autoencoder_params['n_hidden'])
    sig_init=float(autoencoder_params['sig_init'])
    sig_neu=float(autoencoder_params['sig_neu'])
    lr=float(autoencoder_params['lr'])
    beta=float(autoencoder_params['beta'])
    beta_sp=float(autoencoder_params['beta_sp'])
    p_norm=float(autoencoder_params['p_norm'])
    
    # Unpack some batching parameters:
    batch_size=int(autoencoder_params['batch_size'])
    n_epochs=int(autoencoder_params['n_epochs'])
    
    # Initialize output arrays:
    perf_orig=np.zeros((n_files,2))
    if save_learning:
        perf_out=np.zeros((n_files,n_epochs,2))
        perf_hidden=np.zeros((n_files,n_epochs,2))
        loss_epochs=np.zeros((n_files,n_epochs))
    else:
        perf_out=None
        perf_hidden=None
        loss_epochs=None
        
    task_inpt=np.zeros((n_files,3,2))    
    ccgp_inpt=np.zeros((n_files,2,2,2))    
    task_hidden=np.zeros((n_files,3,2))    
    ccgp_hidden=np.zeros((n_files,2,2,2))
    task_rec=np.zeros((n_files,3,2))    
    ccgp_rec=np.zeros((n_files,2,2,2))
    xor_means_files=[] # will be used for plotting means of XOR task
    
    
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
        F_train, train_labels=prep_data4ae(train_session, task)
        F_train_torch=Variable(torch.from_numpy(np.array(F_train,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor
        train_labels_torch=Variable(torch.from_numpy(np.array(train_labels,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
    
        # Prepare stimulated trial data for *testing* autoencoder:
        F_test, test_labels=prep_data4ae(test_session, task)
        F_test_torch=Variable(torch.from_numpy(np.array(F_test,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor
        test_labels_torch=Variable(torch.from_numpy(np.array(test_labels,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
            
        # Test logistic regression performance on original data:
        perf_orig[k]=classifier(F_test,test_labels,1, 'logistic')
        
        # Test MLP if requested:
        if mlp_params!=None:
            perf_orig_mlp[k]=classifier(F_test,test_labels,model='mlp', hidden_layer_sizes=mlp_hidden_layer_sizes, activation=mlp_activation, solver=mlp_solver, reg=mlp_alpha, lr=mlp_lr, lr_init=mlp_lr_init)    
        
        # Create and fit task-optimized autoencoder:
        print('Fitting autoencoder...')
        n_inp=F_train.shape[1]
        model=sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init,k=len(np.unique(train_labels))) 
        ae=fit_autoencoder(model=model,data_train=F_train_torch, clase_train=train_labels_torch, data_test=F_test_torch, clase_test=test_labels_torch, n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu, beta=beta, beta_sp=beta_sp, p_norm=p_norm, save_learning=save_learning, verbose=verbose)
            
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
            
            # Test geometry iterating over subsamples to deal with any imbalances in trials per condition:
            task_inpt_m, ccgp_inpt_m, xor_dats_inpt = test_autoencoder_geometry(F_summed, Fb, n_geo_subsamples, geo_reg)
            task_hidden_m, ccgp_hidden_m, xor_dats_hidden = test_autoencoder_geometry(hidden_rep, Fb, n_geo_subsamples, geo_reg)
            task_rec_m, ccgp_rec_m, xor_dats_rec = test_autoencoder_geometry(rec_rep, Fb, n_geo_subsamples, geo_reg)
            
            # Average XOR data across subsamples:
            xor_dats_inpt=np.mean(xor_dats_inpt,axis=0)
            xor_means_files.append(xor_dats_inpt)
            
            # Plot mean data by XOR condition:
            if plot_xor and k==n_files-1:
                xor_fig=plt.figure(figsize=(4,4))
                xor_ax=xor_fig.add_subplot(111)
                xor_ax.violinplot(xor_means_files[-1],showmeans=True)
                
            # Write results to output array:
            task_inpt[k]=task_inpt_m
            ccgp_inpt[k]=ccgp_inpt_m
            task_hidden[k]=task_hidden_m
            ccgp_hidden[k]=ccgp_hidden_m
            task_rec[k]=task_rec_m
            ccgp_rec[k]=ccgp_rec_m
            
        else:
            task_rec_m=None
            ccgp_rec_m=None
            task_hidden_m=None
            ccgp_hidden_m=None
            
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
        save_ae_results(h5path,perf_orig,perf_out,perf_hidden,loss_epochs, perf_orig_mlp,task_rec,ccgp_rec,task_hidden,ccgp_hidden)
        
        if save_sessions and sessions==None:
            sessions_df=pd.concat(sessions, ignore_index=True)
            sessions_path=os.path.join(output_directory, 'simulated_sessions.pickle')
            pickle.dump(sessions_df, open(sessions_path, 'wb'))
        
        # Save plot of means of XOR data:
        if test_geometry and plot_xor:
            xor_fig_path=os.path.join(output_directory,'xor_means.png')
            xor_fig.savefig(xor_fig_path,dpi=500)
        
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
            M.add_param('task', task)
            M.add_param('n_files', n_files)
            M.date=end_time.strftime('%Y-%m-%d')
            M.time=end_time.strftime('%H:%M:%S')
            M.duration=seconds_2_full_time_str(duration.seconds)
            M.add_output(h5path)
            if test_geometry and plot_xor:
                M.add_output(xor_fig_path)
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
        results['task_hidden']=task_hidden
        results['ccgp_hidden']=ccgp_hidden
        results['task_rec']=task_rec
        results['ccgp_rec']=ccgp_rec
    
    return results



def prep_data4ae(session, task):
    
    F=session2feature_array(session) # extract t-by-g matrix of feature data, where t is number of trials, g is total number of features (across all time bins)
    labels=session2labels(session, task) # generate vector of labels    
    return F, labels
    


def save_ae_results(fpath, perf_orig, perf_out, perf_hidden, loss_epochs, 
    perf_orig_mlp=None, task_rec=None, ccgp_rec=None, task_hidden=None, ccgp_hidden=None):
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
        if task_hidden is not None: 
            hfile.create_dataset('task_hidden', data=task_hidden)
        if ccgp_hidden is not None:
            hfile.create_dataset('ccgp_hidden', data=ccgp_hidden)

    

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
    M.add_param('sim_params', sim_params_out)

    # Write autoencoder hyperparameters to metadata:
    autoencoder_params_out=dict()
    autoencoder_params_out['n_hidden']=int(autoencoder_params['n_hidden'])
    autoencoder_params_out['sig_init']=float(autoencoder_params['sig_init'])            
    autoencoder_params_out['sig_neu']=float(autoencoder_params['sig_neu'])                        
    autoencoder_params_out['lr']=float(autoencoder_params['lr'])                        
    autoencoder_params_out['beta']=float(autoencoder_params['beta'])
    autoencoder_params_out['n_epochs']=int(autoencoder_params['n_epochs'])                        
    autoencoder_params_out['batch_size']=int(autoencoder_params['batch_size'])                        
    autoencoder_params_out['beta_sp']=float(autoencoder_params['beta_sp'])                                    
    autoencoder_params_out['p_norm']=float(autoencoder_params['p_norm'])                                                
    M.add_param('autoencoder_params', autoencoder_params_out)
    
    # Write MLP hyperparameters to metadata if necessary:
    if mlp_params!=None:
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
        perf_tasks, perf_ccgp, xor_dat = geometry_2D(feat_decod_subsample,feat_binary_subsample,reg) # on reconstruction
        xor_dats.append(xor_dat)

        task_total[s,:,:]=perf_tasks
        ccgp_total[s,:,:,:]=perf_ccgp            
    
    # Average across subsamples:
    task_m=np.mean(task_total,axis=0)
    ccgp_m=np.mean(ccgp_total,axis=0)
    xor_dats=np.array(xor_dats)
    
    return task_m, ccgp_m
    


# Autoencoder Architecture
class sparse_autoencoder_1(nn.Module):
    def __init__(self,n_inp,n_hidden,sigma_init,k=2):
        super(sparse_autoencoder_1,self).__init__()
        self.n_inp=n_inp
        self.n_hidden=n_hidden
        self.sigma_init=sigma_init
        self.enc=torch.nn.Linear(n_inp,n_hidden)
        self.dec=torch.nn.Linear(n_hidden,n_inp)
        self.dec2=torch.nn.Linear(n_hidden,k)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.sigma_init)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=self.sigma_init)
        
    def forward(self,x,sigma_noise):
        x_hidden = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden)
        x = self.dec(x_hidden)
        x2 = self.dec2(x_hidden)
        return x,x_hidden,x2

def sparsity_loss(data,p):
    #shap=data.size()
    #nt=shap[0]*shap[1]
    #loss=(1/nt)*torch.norm(data,p)
    #loss=torch.norm(data,p)
    #loss=torch.mean(torch.sigmoid(100*(data-0.1)),axis=(0,1))
    loss=torch.mean(torch.pow(abs(data),p),axis=(0,1))
    return loss


