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
from simulation_whiskers.simulate_task import simulate_session, session2feature_array, session2labels, load_simulation
from simulation_whiskers.functions_geometry import geometry_2D, find_matching_2d_bin_trials, subsample_2d_bins
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
def fit_autoencoder(model,data_train,clase_train,data_test,clase_test,n_epochs,batch_size,lr,sigma_noise,beta,beta_sp,p_norm):
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
    
    loss_rec_vec=np.empty(n_epochs); loss_rec_vec[:]=np.nan
    loss_ce_vec=np.empty(n_epochs); loss_ce_vec[:]=np.nan     
    loss_sp_vec=np.empty(n_epochs); loss_sp_vec[:]=np.nan
    loss_vec=np.empty(n_epochs); loss_vec[:]=np.nan
    data_epochs_train=np.empty((n_epochs, n_trials_train, n_input_features));
    data_hidden_train=np.empty((n_epochs, n_trials_train, n_hidden));
    
    data_epochs_test=np.empty((n_epochs, n_trials_test, n_input_features));
    data_hidden_test=np.empty((n_epochs, n_trials_test, n_hidden));

    t=0
    while t<n_epochs: 
        #print (t)
        
        # Compute loss, generate hidden and output representations using training trials:
        outp_train=model(data_train,sigma_noise)
        data_epochs_train[t]=outp_train[0].detach().numpy()
        data_hidden_train[t]=outp_train[1].detach().numpy()
        loss_rec=loss1(outp_train[0],data_train).item()
        loss_ce=loss2(outp_train[2],clase_train).item()
        loss_sp=sparsity_loss(outp_train[2],p_norm).item()
        loss_total=((1-beta)*loss_rec+beta*loss_ce+beta_sp*loss_sp)
        loss_rec_vec[t]=loss_rec
        loss_ce_vec[t]=loss_ce
        loss_sp_vec[t]=loss_sp
        loss_vec[t]=loss_total
        
        # Generate hidden and output layer representations of held-out trials: 
        outp_test=model(data_test,sigma_noise)
        data_epochs_test[t]=outp_test[0].detach().numpy()
        data_hidden_test[t]=outp_test[1].detach().numpy()
        
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
    return loss_rec_vec,loss_ce_vec,loss_sp_vec,loss_vec,np.array(data_epochs_test),np.array(data_hidden_test), np.array(data_epochs_train),np.array(data_hidden_train)



def iterate_fit_autoencoder(sim_params, autoencoder_params, task, n_files, n_geo_subsamples=10, mlp_params=None, test_geometry=True, sessions_in=None, save_perf=False, save_sessions=False, output_directory=None):
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
    n_hidden=autoencoder_params['n_hidden']
    sig_init=autoencoder_params['sig_init']
    sig_neu=autoencoder_params['sig_neu']
    lr=autoencoder_params['lr']
    beta=autoencoder_params['beta']
    beta_sp=autoencoder_params['beta_sp']
    p_norm=autoencoder_params['p_norm']
    
    # Unpack some batching parameters:
    batch_size=autoencoder_params['batch_size']
    n_epochs=autoencoder_params['n_epochs']
    n_splits=autoencoder_params['n_splits']
    
    # Initialize output arrays:
    perf_orig=np.zeros((n_files,2))
    perf_out=np.zeros((n_files,n_splits,n_epochs,2))
    perf_hidden=np.zeros((n_files,n_splits,n_epochs,2))
    loss_epochs=np.zeros((n_files,n_splits,n_epochs))
    
    # If also running MLP:
    if mlp_params!=None:
        perf_orig_mlp=np.zeros((n_files,2))
        mlp_hidden_layer_sizes=mlp_params['hidden_layer_sizes']
        mlp_activation=mlp_params['activation']        
        mlp_alpha=mlp_params['alpha']        
        mlp_solver=mlp_params['solver']        
        mlp_lr=mlp_params['learning_rate']        
        mlp_lr_init=mlp_params['learning_rate_init']

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
        
        # Simulate session (if not loading previously-simulated session):
        if sessions_in==None:
            
            # Generate session for training autoencoder:
            train_session=simulate_session(sim_params, sum_bins=True)
            train_session['file_idx']=k
            
            # Generate separate session for testing autoencoder:
            test_session=simulate_session(sim_params, sum_bins=True)
            test_session['file_idx']=k
            
            if save_sessions:
                train_sessions.append(session)
                test_sessions.append(session)
        else:
            session=sessions[sessions.file_idx==k]
        
        # Prepare simulated trial data for *training* autoencoder:
        F_train=session2feature_array(train_session) # extract t-by-g matrix of feature data, where t is number of trials, g is total number of features (across all time bins)
        n_inp=F.shape[1]
        x_torch_train=Variable(torch.from_numpy(np.array(F_train,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor
        train_labels=session2labels(train_session, task) # generate vector of labels    
        train_labels_torch=Variable(torch.from_numpy(np.array(train_labels,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
    
        # Prepare stimulated trial data for *testing* autoencoder:
        F_test=session2feature_array(test_session)
        x_torch_test=Variable(torch.from_numpy(np.array(F_test,dtype=np.float32)),requires_grad=False) 
        test_labels=session2labels(test_session, task) # generate vector of labels    
        test_labels_torch=Variable(torch.from_numpy(np.array(test_labels,dtype=np.int64)),requires_grad=False) 
            
        # Test logistic regression performance on original data:
        perf_orig[k]=classifier(F_test,test_labels,1, 'logistic')
        
        # Test MLP if requested:
        if mlp_params!=None:
            perf_orig_mlp[k]=classifier(F_test,labels_test,model='mlp', hidden_layer_sizes=mlp_hidden_layer_sizes, activation=mlp_activation, solver=mlp_solver, reg=mlp_alpha, lr=mlp_lr, lr_init=mlp_lr_init)    
        
        # Create and fit task-optimized autoencoder:
        model=sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init,k=len(np.unique(labels))) 
        loss_rec_vec, loss_ce_vec, loss_sp_vec, loss_vec, data_epochs_test, data_hidden_test, data_epochs_train, data_hidden_train=fit_autoencoder(model=model,data_train=x_torch_train, clase_train=train_labels_torch, data_test=x_torch_test, clase_test=test_labels_torch, n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu, beta=beta, beta_sp=beta_sp, p_norm=p_norm)
        loss_epochs[k,cv_idx,:]=loss_vec
            
        # Test logistic regression performance on reconstructed data:
        for i in range(n_epochs):
            perf_out[k,cv_idx,i]=classifier(data_epochs_test[i],test_labels,1)
            perf_hidden[k,cv_idx,i]=classifier(data_hidden_test[i],test_labels,1)
        
        # Test geometry if requested:
        if test_geometry:
            
            # Initialize arrays of results
            task_rec_total=np.empty(n_geo_subsamples,3,2) #reconstruction performance
            ccgp_rec_total=np.empty(n_geo_subsamples,2,2,2) #reconstruction ccgp
            
            task_hidden_total=np.empty(n_geo_subsamples,3,2) #hidden performance
            ccgp_hidden_total=np.empty(n_geo_subsamples,2,2,2) #hidden ccgp
            
            # Extract matrix of summed contacts:
            F_summed=session2feature_array(session.iloc[train_index], field='features_bins_summed')
            
            # Only need contacts, not angles, so exclude odd columns:
            keep_columns=np.arange(0,F_summed.shape[1],2)
            F_summed=F_summed[:,keep_columns]
            
            # Binarize contacts:
            Fb=binarize_contacts(F_summed)
            
            # Find minimum number of trials per condition:
            bin_conditions=find_matching_2d_binar_trials(Fb)
            min_n=min([x['count'] for x in bin_conditions])
            
            # Iterate over subsamples
            for s in np.arange(n_geo_subsamples):
                
                # Select current subsample:
                curr_subsample_indices=subsample_2d_bin(bin_conditions, min_n)
                Fb_subsample=Fb[curr_subsample]
                rec_subsample=data_epochs_test[-1][curr_subsample]
                hidden_subsample=data_hidden_test[-1][curr_subsample]
                
                # Test geometry:
                perf_tasks_rec, perf_ccgp_rec = geometry_2D(rec_subsample,Fb_subsample,geo_reg) # on reconstruction
                perf_tasks_hidden, perf_ccgp_hidden = geometry_2D(hidden_subsample,Fb_subsample,geo_reg) # on hidden layer
    
                task_rec_total[s,:,:]=perf_tasks_rec            
                ccgp_rec_total[s,:,:,:]=perf_ccgp_rec            
    
                task_hidden_total[s,:,:]=perf_tasks_hidden            
                ccgp_hidden_total[s,:,:,:]=perf_ccgp_hidden            
    
    
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
            
        # Save HDF5:
        h5path = os.path.join(output_directory, 'iterate_autoencoder_results.h5')
        with h5py.File(h5path, 'w') as hfile:
            hfile.create_dataset('perf_orig', data=perf_orig)
            hfile.create_dataset('perf_out', data=perf_out)
            hfile.create_dataset('perf_hidden', data=perf_hidden)
            hfile.create_dataset('loss_epochs', data=loss_epochs)
            if mlp_params!=None:
                hfile.create_dataset('perf_orig_mlp', data=perf_orig_mlp)    
        
        if save_sessions and sessions==None:
            sessions_df=pd.concat(sessions, ignore_index=True)
            sessions_path=os.path.join(output_directory, 'simulated_sessions.pickle')
            pickle.dump(sessions_df, open(sessions_path, 'wb'))
        
        # Save metadata if analysis_metadata successfully imported:
        if 'analysis_metadata' in sys.modules:
            M=Metadata()
            
            # If loading previously-simulated session and it was passed as path,
            # add file path to metadata:
            if sessions_in!=None and type(sessions_in)==str:
                M.add_input(sessions_in)
            
            # Write simulation parameters to metadata:
            sim_params_out=dict()
            sim_params_out['n_whisk']=sim_params['n_whisk']
            sim_params_out['prob_poiss']=sim_params['prob_poiss']
            sim_params_out['noise_w']=sim_params['noise_w']
            sim_params_out['spread']=sim_params['spread']  
            sim_params_out['speed']=sim_params['speed']  
            sim_params_out['ini_phase_m']=sim_params['ini_phase_m']
            sim_params_out['ini_phase_spr']=sim_params['ini_phase_spr']
            sim_params_out['delay_time']=sim_params['delay_time']
            sim_params_out['freq_m']=sim_params['freq_m']
            sim_params_out['freq_std']=sim_params['freq_std']            
            sim_params_out['t_total']=sim_params['t_total']
            sim_params_out['dt']=sim_params['dt']            
            sim_params_out['dx']=sim_params['dx']            
            sim_params_out['n_trials_pre']=sim_params['n_trials_pre']
            sim_params_out['amp']=sim_params['amp']            
            sim_params_out['freq_sh']=sim_params['freq_sh']
            sim_params_out['z1']=sim_params['z1']
            sim_params_out['disp']=sim_params['disp']
            sim_params_out['theta']=sim_params['theta']
            sim_params_out['steps_mov']=sim_params['steps_mov']
            sim_params_out['rad_vec']=sim_params['rad_vec']
            M.add_param('sim_params', sim_params_out)

            # Write autoencoder hyperparameters to metadata:
            autoencoder_params_out=dict()
            autoencoder_params_out['n_hidden']=autoencoder_params['n_hidden']
            autoencoder_params_out['sig_init']=autoencoder_params['sig_init']            
            autoencoder_params_out['sig_neu']=autoencoder_params['sig_neu']                        
            autoencoder_params_out['lr']=autoencoder_params['lr']                        
            autoencoder_params_out['beta']=autoencoder_params['beta']                        
            autoencoder_params_out['n_epochs']=autoencoder_params['n_epochs']                        
            autoencoder_params_out['batch_size']=autoencoder_params['batch_size']                        
            autoencoder_params_out['beta_sp']=autoencoder_params['beta_sp']                                    
            autoencoder_params_out['p_norm']=autoencoder_params['p_norm']                                                
            M.add_param('autoencoder_params', autoencoder_params_out)
            
            # Write MLP hyperparameters to metadata if necessary:
            if mlp_params!=None:
                mlp_params_out=dict()
                mlp_params_out['hidden_layer_sizes']=mlp_params['hidden_layer_sizes']
                mlp_params_out['activation']=mlp_params['activation']
                mlp_params_out['alpha']=mlp_params['alpha']
                mlp_params_out['solver']=mlp_params['solver']
                mlp_params_out['learning_rate']=mlp_params['learning_rate']
                mlp_params_out['learning_rate_init']=mlp_params['learning_rate_init']
                M.add_param('mlp_params', mlp_params_out)
            
            M.add_param('task', task)
            M.add_param('n_files', n_files)
            M.date=end_time.strftime('%Y-%m-%d')
            M.time=end_time.strftime('%H:%M:%S')
            M.duration=seconds_2_full_time_str(duration.seconds)
            M.add_output(h5path)
            if save_sessions and sessions==None:
                M.add_output(sessions_path)
            metadata_path=os.path.join(output_directory, 'iterate_autoencoder_metdata.json')
            write_metadata(M, metadata_path)
    
    results=dict()
    results['perf_orig']=perf_orig
    results['perf_out']=perf_out
    results['perf_hidden']=perf_hidden
    results['loss_epochs']=loss_epochs
    if mlp_params!=None:
        results['perf_orig_mlp']=perf_orig_mlp
    
    return results
    

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


