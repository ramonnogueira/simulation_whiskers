import os
import sys
from datetime import datetime
import pathlib
import h5py
import numpy as np
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
from sklearn.model_selection import StratifiedKFold
def warn(*args, **kwargs):
    pass
import warnings
from simulation_whiskers.simulate_task import simulate_session, session2feature_array, session2labels
warnings.warn = warn
nan=float('nan')
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata, seconds_2_full_time_str
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False
import time

# Standard classifier
def classifier(data,clase,reg):
    n_splits=5
    perf=nan*np.zeros((n_splits,2))
    cv=StratifiedKFold(n_splits=n_splits)
    g=-1
    for train_index, test_index in cv.split(data,clase):
        g=(g+1)
        clf = LogisticRegression(C=reg,class_weight='balanced')
        clf.fit(data[train_index],clase[train_index])
        perf[g,0]=clf.score(data[train_index],clase[train_index])
        perf[g,1]=clf.score(data[test_index],clase[test_index])
    return np.mean(perf,axis=0)

# Fit the autoencoder. The data needs to be in torch format
def fit_autoencoder(model,data,clase,n_epochs,batch_size,lr,sigma_noise,beta,beta_sp,p_norm):
    train_loader=DataLoader(torch.utils.data.TensorDataset(data,data,clase),batch_size=batch_size,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    loss1=torch.nn.MSELoss()
    loss2=torch.nn.CrossEntropyLoss()
    model.train()
    
    n_trials=len(clase)
    n_input_features=data.shape[1]
    n_hidden=model.enc.out_features
    
    loss_rec_vec=np.empty(n_epochs); loss_rec_vec[:]=np.nan
    loss_ce_vec=np.empty(n_epochs); loss_ce_vec[:]=np.nan     
    loss_sp_vec=np.empty(n_epochs); loss_sp_vec[:]=np.nan
    loss_vec=np.empty(n_epochs); loss_vec[:]=np.nan
    data_epochs=np.empty((n_epochs, n_trials, n_input_features));
    data_hidden=np.empty((n_epochs, n_trials, n_hidden));

    t=0
    while t<n_epochs: 
        #print (t)
        outp=model(data,sigma_noise)
        data_epochs[t]=outp[0].detach().numpy()
        data_hidden[t]=outp[1].detach().numpy()
        loss_rec=loss1(outp[0],data).item()
        loss_ce=loss2(outp[2],clase).item()
        loss_sp=sparsity_loss(outp[2],p_norm).item()
        loss_total=((1-beta)*loss_rec+beta*loss_ce+beta_sp*loss_sp)
        loss_rec_vec[t]=loss_rec
        loss_ce_vec[t]=loss_ce
        loss_sp_vec[t]=loss_sp
        loss_vec[t]=loss_total
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
    return loss_rec_vec,loss_ce_vec,loss_sp_vec,loss_vec,np.array(data_epochs),np.array(data_hidden)


def iterate_fit_autoencoder(sim_params, autoencoder_params, task, n_files, save_output=False, output_directory=None):
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
    
    # Initialize output arrays:
    perf_orig=np.zeros((n_files,2))
    perf_out=np.zeros((n_files,n_epochs,2))
    perf_hidden=np.zeros((n_files,n_epochs,2))
    loss_epochs=np.zeros((n_files,n_epochs))


    for k in range(n_files):
        
        # Simulate session:
        session=simulate_session(sim_params)
        
        # Prepare simulated trial data for autoencoder:
        F=session2feature_array(session) # extract t-by-g matrix of feature data, where t is number of trials, g is total number of features (across all time bins)
        n_inp=F.shape[1]
        x_torch=Variable(torch.from_numpy(np.array(F,dtype=np.float32)),requires_grad=False) # convert features from numpy array to pytorch tensor
        labels=session2labels(session, task) # generate vector of labels    
        labels_torch=Variable(torch.from_numpy(np.array(labels,dtype=np.int64)),requires_grad=False) # convert labels from numpy array to pytorch tensor
    
        # Test logistic regression performance on original data:
        perf_orig[k]=classifier(F,labels,1)
        
        # Create and fit task-optimized autoencoder:
        model=sparse_autoencoder_1(n_inp=n_inp,n_hidden=n_hidden,sigma_init=sig_init,k=len(np.unique(labels))) 
        loss_rec_vec, loss_ce_vec, loss_sp_vec, loss_vec, data_epochs, data_hidden=fit_autoencoder(model=model,data=x_torch, clase=labels_torch, n_epochs=n_epochs,batch_size=batch_size,lr=lr,sigma_noise=sig_neu, beta=beta, beta_sp=beta_sp, p_norm=p_norm)
        loss_epochs[k]=loss_vec
        
        # Test logistic regression performance on reconstructed data:
        for i in range(n_epochs):
            perf_out[k,i]=classifier(data_epochs[i],labels,1)
            perf_hidden[k,i]=classifier(data_hidden[i],labels,1)
    
    time.sleep(2)
    end_time=datetime.now()
    duration = end_time - start_time
    
    if save_output:
        
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
        
        # Save metadata if analysis_metadata successfully imported:
        if 'analysis_metadata' in sys.modules:
            M=Metadata()
            
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
            
            M.add_param('task', task)
            M.add_param('n_files', n_files)
            M.date=end_time.strftime('%Y-%m-%d')
            M.time=end_time.strftime('%H:%M:%S')
            M.duration=seconds_2_full_time_str(duration.seconds)
            M.add_output(h5path)
            metadata_path=os.path.join(output_directory, 'iterate_autoencoder_metdata.json')
            write_metadata(M, metadata_path)
                
    return perf_orig, perf_out, perf_hidden, loss_epochs
    

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
