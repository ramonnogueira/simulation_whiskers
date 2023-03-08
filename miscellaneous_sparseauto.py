import os
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
from simulate_task import simulate_session, session2feature_array, session2labels
warnings.warn = warn
nan=float('nan')


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
    
    loss_rec_vec=[]
    loss_ce_vec=[]
    loss_sp_vec=[]
    loss_vec=[]
    data_epochs=[]
    data_hidden=[]
    t=0
    while t<n_epochs: 
        #print (t)
        outp=model(data,sigma_noise)
        data_epochs.append(outp[0].detach().numpy())
        data_hidden.append(outp[1].detach().numpy())
        loss_rec=loss1(outp[0],data).item()
        loss_ce=loss2(outp[2],clase).item()
        loss_sp=sparsity_loss(outp[2],p_norm).item()
        loss_total=((1-beta)*loss_rec+beta*loss_ce+beta_sp*loss_sp)
        loss_rec_vec.append(loss_rec)
        loss_ce_vec.append(loss_ce)
        loss_sp_vec.append(loss_sp)
        loss_vec.append(loss_total)
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
    return np.array(loss_rec_vec),np.array(loss_ce_vec),np.array(loss_sp_vec),np.array(loss_vec),np.array(data_epochs),np.array(data_hidden)


def iterate_fit_autoencoder(sim_params, autoencoder_params, task, n_files, save_output=False, output_directory=None):
    
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
    
    if save_output:
        
        # Make current folder default:
        if output_directory==None:
            output_directory=os.getcwd()
            
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
            
            h5path = os.path.join(output_directory, 'iterate_autoencoder_results.h5')
            with h5py.File(h5path, 'w') as hfile:
                hfile.create_dataset('perf_orig', data=perf_orig)
                hfile.create_dataset('perf_out', data=perf_out)
                hfile.create_dataset('perf_hidden', data=perf_hidden)
                hfile.create_dataset('loss_epochs', data=loss_epochs)
                
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
