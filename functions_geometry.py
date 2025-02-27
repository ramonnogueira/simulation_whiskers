import os
import matplotlib.pylab as plt
import numpy as np
import scipy
import pandas as pd
import pickle as pkl
from numpy.random import permutation
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import ortho_group 
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import torch
nan=float('nan')

# Evaluate Geometry
# Feat decoding is the features to be decoded (e.g. neural activity). Matrix number of trials x number of features
# Feat binary is the variables to decode. Matrix number of trials x 2. Each trial is a 2D binary word ie [0,1] (two variables to values each variable)
# reg is regularization
def geometry_2D(feat_decod,feat_binary,reg):
    """
    Analyze geometry of representation of decoded variable in input feature 
    space.

    Parameters
    ----------
    feat_decod : array-like
        t-by-f, where t is number of trials and f is number of features per
        trial.
   
    feat_binary : array-like
        t-by-2, where 2 is number of trials.
    
    reg : float
        Regularization parameter used in logistic regression.


    Returns
    -------
    perf_tasks : numpy.ndarray
        3-by-2 array of logistic regression performance on 3 tasks: 1) binary 
        classification of output variable 1, 2) binary classification of output 
        variable 2, and 3) XOR task defined over 2 output variables. Each row
        corresponds to one task; column 0 is performance on training data,
        column 1 is performance on test data.
    
    perf_ccgp : numpy.ndarray
        2-by-2-by-2 array of CCGP performance. (TODO: double-check this!)
        
            Axis 0 ('slices') : each slice corresponds to a feature dimension 
                to decode.
            
            Axis 1 ('rows') : given slice corresponding to decoded feature 
                dimension, each row corresponds to one value of the *other*,
                *non*-decoded feature dimension; training data for decoders 
                will be drawn only from trials with corresponding value of 
                non-decoded feature dimension.
            
            Axis 2 ('columns') : column 0: performance on training data; 
                column 1: performance on test data. 
        

    """
    
    # Assigns to each binary word a number from 0 to 3: [0,0] -> 0, [0,1] -> 1, [1,0] -> 2, [1,1] -> 3.
    exp_uq=np.unique(feat_binary,axis=0)
    feat_binary_exp=np.zeros(len(feat_binary))
    for t in range(len(feat_binary)):
        for tt in range((len(exp_uq))):
            gg=(np.sum(feat_binary[t]==exp_uq[tt])==len(feat_binary[0]))
            if gg:
                feat_binary_exp[t]=tt
    
    # Define the dichotomies for the 2D case            
    dichotomies=np.array([[0,0,1,1],[0,1,0,1]])
    train_dich=np.array([[[0,2],[1,3]],[[0,1],[2,3]]])
    test_dich=np.array([[[1,3],[0,2]],[[2,3],[0,1]]])

    # Initialize output dataframes:
    geo_df = pd.DataFrame(columns=['dichotomy', 'train_partition', 'train_accuracy', 'test_accuracy', 'parallelism'])

    # Evaluates CCGP/parallelism (abstraction)
    all_dichotomies = []
    all_train_partitions = []
    all_test_acc = []
    all_train_acc = []
    all_par = []
    perf_ccgp=nan*np.zeros((len(dichotomies),len(train_dich[0]),2))
    parallel=nan*np.zeros(len(dichotomies))
    
    #Loop on "dichotomies"
    for k in range(len(dichotomies)): 
      para=nan*np.zeros((len(train_dich[0]),len(feat_decod[0])))
      
      #Loop on ways to train this particular "dichotomy"
      for kk in range(len(train_dich[0])): 
         ind_train=np.where((feat_binary_exp==train_dich[k][kk][0])|(feat_binary_exp==train_dich[k][kk][1]))[0]
         ind_test=np.where((feat_binary_exp==test_dich[k][kk][0])|(feat_binary_exp==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(feat_binary_exp))
         for i in range(4):
             ind_task=(feat_binary_exp==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=reg,class_weight='balanced',solver='lbfgs')
         #supp=LinearSVC(C=reg,class_weight='balanced')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         para[kk]=supp.coef_[0]
         perf_ccgp[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         perf_ccgp[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
         
         all_dichotomies.append(k)
         all_train_partitions.append(kk)
         all_train_acc.append(supp.score(feat_decod[ind_train],task[ind_train]))
         all_test_acc.append(supp.score(feat_decod[ind_test],task[ind_test]))
         
      parallel[k]=np.dot(para[0],para[1])/(np.linalg.norm(para[0])*np.linalg.norm(para[1]))
      all_par += 2*[np.dot(para[0],para[1])/(np.linalg.norm(para[0])*np.linalg.norm(para[1]))]
  
    geo_df['dichotomy'] = all_dichotomies
    geo_df['train_partition'] = all_train_partitions
    geo_df['train_accuracy'] = all_train_acc
    geo_df['test_accuracy'] = all_test_acc
    geo_df['parallelism'] = all_par
    
    return geo_df



def perf_2D(feat_decod,feat_binary,clf_type='logistic',
    lr_params={'C':1,'class_weight':'balanced', 'solver':'lbfgs'}, 
    mlp_params=None):
    
    if clf_type != 'logistic' and clf_type != 'mlp':
        raise AssertionError('Please specify either ''logistic'' or ''mlp'' for `clf_type` param.')

    # Initialize logistic regression if requested:
    if clf_type == 'logistic': 
        if lr_params is not None:
            clf=LogisticRegression(C=lr_params['C'],class_weight=lr_params['class_weight'],solver=lr_params['solver'])
        else:
            raise AssertionError('`clf_type` set to ''logistic'' but no `lr_params` specified.')
            
    # Initialize MLP if requested:
    elif clf_type == 'mlp' :
        if mlp_params is None:
            clf=MLPClassifier(hidden_layer_sizes=mlp_params['hidden_layer_sizes'],
                              activation=mlp_params['activation'],
                              solver=mlp_params['solver'],
                              alpha=mlp_params['reg'],
                              learning_rate=mlp_params['lr'], 
                              learning_rate_init=mlp_params['lr_init'])
        else:
            raise AssertionError('`clf_type` set to ''mlp'' but no `mlp_params` specified.')
    
    # Evaluate decoding perf on variable 1, variable 2 and xor tasks.
    xor=np.sum(feat_binary,axis=1)%2 # Define the XOR function wrt to the two variables
    n_cv=5
    perf_tasks_pre=np.zeros((n_cv,3,2))

    # Initialize output dataframes:
    perf_df = pd.DataFrame(columns=['task', 'train', 'test'])
    all_tasks = []
    all_train_acc = []
    all_test_acc = []

    # Variable 1
    skf=StratifiedKFold(n_splits=n_cv,shuffle=True)
    g=-1
    for train, test in skf.split(feat_decod,feat_binary[:,0]):
        g=(g+1)
        mod=clf.fit(feat_decod[train],feat_binary[:,0][train])
        perf_tasks_pre[g,0,0]=clf.score(feat_decod[train],feat_binary[:,0][train])
        perf_tasks_pre[g,0,1]=clf.score(feat_decod[test],feat_binary[:,0][test])

    # Variable 2
    skf=StratifiedKFold(n_splits=n_cv,shuffle=True)
    g=-1
    for train, test in skf.split(feat_decod,feat_binary[:,1]):
        g=(g+1)
        mod=clf.fit(feat_decod[train],feat_binary[:,1][train])
        perf_tasks_pre[g,1,0]=clf.score(feat_decod[train],feat_binary[:,1][train])
        perf_tasks_pre[g,1,1]=clf.score(feat_decod[test],feat_binary[:,1][test])

    # XOR
    skf=StratifiedKFold(n_splits=n_cv,shuffle=True)
    g=-1
    for train, test in skf.split(feat_decod,xor):
        g=(g+1)
        
        # Initialize array that will be used for storing XOR distributions:
        if g==0:
            xor_dat=np.empty((n_cv,int(np.sum(xor)),2))
        
        mod=clf.fit(feat_decod[train],xor[train])
        perf_tasks_pre[g,2,0]=clf.score(feat_decod[train],xor[train])
        perf_tasks_pre[g,2,1]=clf.score(feat_decod[test],xor[test])
        
        # Save data split by XOR label:
        xor0=feat_decod[xor==0]
        xor0_m=np.mean(xor0,axis=1)
        xor1=feat_decod[xor==1]
        xor1_m=np.mean(xor1,axis=1)
        xor_dat[g,:,0]=xor0_m
        xor_dat[g,:,1]=xor1_m
        
    xor_dat=np.mean(xor_dat,axis=0)
    perf_tasks=np.mean(perf_tasks_pre,axis=0)
    
    all_tasks = [0,1,'xor']
    all_train_acc = perf_tasks[:,0]
    all_test_acc = perf_tasks[:,1]
    
    perf_df['clf_type'] = clf_type
    perf_df['task'] = all_tasks
    perf_df['train'] = all_train_acc
    perf_df['test'] = all_test_acc
    
    return perf_df
    


def find_matching_2d_bin_trials(feat_binary):
    """
    Find indices of trials matching each of 4 possible conditions defined over
    2 binary variables. 

    Parameters
    ----------
    feat_binary : array-like
        t-by-2 matrix, where t is the number of trials.

    Returns
    -------
    conditions : list
        List of 4 dictionaries, each corresponding to a possible permutation 
        of 2 binary variables ([0,0], [1,0], [0,1], and [1,1]). Each 
        dictionary defines the following keys:
            
            condition : list
                Stimulus condition, defined over 2 binary variables. Either 
                [0,0], [1,0], [0,1], or [1,1].
                
            trial_nums: numpy.ndarray
                Indices of trials of corresponding condition.
                
            count: int
                Number of trials of corresponding condition.

    """
    
    dim1_vals=[0,1]
    dim2_vals=[0,1]
    conditions=[]
    for x in dim1_vals:
        b1=feat_binary[:,0]==x
        for y in dim2_vals:
            b2=feat_binary[:,1]==y
            b=b1&b2
            matching_indices=np.argwhere(b)
            matching_indices=np.squeeze(matching_indices)
            
            # Define dict:
            d=dict()
            d['condition']=[x,y]
            d['trial_nums']=matching_indices
            d['count']=len(matching_indices)
            conditions.append(d)
    return conditions



def subsample_2d_bin(dicts, k):
    """
    Generate indices for a balanced subsample of trials with conditions
    defined over 2 binary output variables. 

    Parameters
    ----------
    dicts : list
        List of 4 dicts, each corresponding to one possible combination of
        values of 2 binary variables. Should be same format as output of 
        find_matching_2d_bin_trials().
        
    k : int
        Number of trials from each condiition to include.

    Returns
    -------
    all_indices : list
        List of trial indices. Should include k trials of each condition.

    """
    # Make sure that k is less than or equal to number of trials for all 
    # conditions:
    if np.any([k>len(x['trial_nums']) for x in dicts]):
        raise IndexError('k greater than number of trials of at least one condition.')
    
    all_indices=[]
    for d in dicts:
        curr_trials=permutation(d['trial_nums'])[0:k]
        all_indices+=list(curr_trials)
    return all_indices



def participation_ratio(X):
    
    # Assume X to be a samples-by-features tensor:
    
    # Center data:    
    mu = torch.mean(X, axis=0)
    Mu = mu.repeat(X.shape[0],1)
    X_ctr = X - Mu
    
    # Compute covariance matrix:
    Cov = torch.matmul(X_ctr.T, X_ctr)
    
    # Compute eigenvalues of covariance matrix:
    eig = torch.linalg.eig(Cov).eigenvalues
    
    # Compute participation ratio:
    pr = (torch.sum(eig)**2)/torch.sum(eig**2)
    
    return pr