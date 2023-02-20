import os
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
def fit_autoencoder(model,data,n_epochs,batch_size,lr,sigma_noise):
    train_loader=DataLoader(torch.utils.data.TensorDataset(data,data),batch_size=batch_size,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    loss=torch.nn.MSELoss()
    model.train()

    loss_vec=[]
    data_epochs=[]
    t=0
    while t<n_epochs: 
        #print (t)
        outp=model(data,sigma_noise)
        data_epochs.append(outp[0].detach().numpy())
        loss_rec=loss(outp[0],data).item()
        loss_vec.append(loss_rec)
        if t==0 or t==(n_epochs-1):
            print (t,loss_rec)
        for batch_idx, (targ1, targ2) in enumerate(train_loader):
            optimizer.zero_grad()
            output=model(targ1,sigma_noise)
            loss_r=loss(output[0],targ2) # reconstruction error
            loss_r.backward() # compute gradient
            optimizer.step() # weigth update
        t=(t+1)
    model.eval()
    return np.array(loss_vec),np.array(data_epochs)

# Autoencoder Architecture
class sparse_autoencoder_1(nn.Module):
    def __init__(self,n_inp,n_hidden,sigma_init):
        super(sparse_autoencoder_1,self).__init__()
        self.n_inp=n_inp
        self.n_hidden=n_hidden
        self.sigma_init=sigma_init
        self.enc=torch.nn.Linear(n_inp,n_hidden)
        self.dec=torch.nn.Linear(n_hidden,n_inp)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.sigma_init)
            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=self.sigma_init)
        
    def forward(self,x,sigma_noise):
        x_hidden = F.relu(self.enc(x))+sigma_noise*torch.randn(x.size(0),self.n_hidden)
        x = self.dec(x_hidden)
        return x,x_hidden

