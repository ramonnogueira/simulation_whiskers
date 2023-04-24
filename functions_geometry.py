import os
import matplotlib.pylab as plt
import numpy as np
import scipy
import pandas
import pickle as pkl
from numpy.random import permutation
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import ortho_group 
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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

    ###################################
    # Evaluate decoding perf on variable 1, variable 2 and xor tasks.
    xor=np.sum(feat_binary,axis=1)%2 # Define the XOR function wrt to the two variables
    n_cv=5
    perf_tasks_pre=np.zeros((n_cv,3,2))

    # Variable 1
    skf=StratifiedKFold(n_splits=n_cv,shuffle=True)
    g=-1
    for train, test in skf.split(feat_decod,feat_binary[:,0]):
        g=(g+1)
        supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
        mod=supp.fit(feat_decod[train],feat_binary[:,0][train])
        perf_tasks_pre[g,0,0]=supp.score(feat_decod[train],feat_binary[:,0][train])
        perf_tasks_pre[g,0,1]=supp.score(feat_decod[test],feat_binary[:,0][test])

    # Variable 2
    skf=StratifiedKFold(n_splits=n_cv,shuffle=True)
    g=-1
    for train, test in skf.split(feat_decod,feat_binary[:,1]):
        g=(g+1)
        supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
        mod=supp.fit(feat_decod[train],feat_binary[:,1][train])
        perf_tasks_pre[g,1,0]=supp.score(feat_decod[train],feat_binary[:,1][train])
        perf_tasks_pre[g,1,1]=supp.score(feat_decod[test],feat_binary[:,1][test])

    # XOR
    skf=StratifiedKFold(n_splits=n_cv,shuffle=True)
    g=-1
    for train, test in skf.split(feat_decod,xor):
        g=(g+1)
        supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
        mod=supp.fit(feat_decod[train],xor[train])
        perf_tasks_pre[g,2,0]=supp.score(feat_decod[train],xor[train])
        perf_tasks_pre[g,2,1]=supp.score(feat_decod[test],xor[test])

    perf_tasks=np.mean(perf_tasks_pre,axis=0)

    ###############################################
    # Calculate Abstraction (CCGP)
    
    # Define the dichotomies for the 2D case            
    dichotomies=np.array([[0,0,1,1],[0,1,0,1]])
    train_dich=np.array([[[0,2],[1,3]],[[0,1],[2,3]]])
    test_dich=np.array([[[1,3],[0,2]],[[2,3],[0,1]]])

    # Evaluates CCGP (abstraction)
    perf_ccgp=nan*np.zeros((len(dichotomies),len(train_dich[0]),2))
    for k in range(len(dichotomies)): #Loop on "dichotomies"
      for kk in range(len(train_dich[0])): #Loop on ways to train this particular "dichotomy"
         ind_train=np.where((feat_binary_exp==train_dich[k][kk][0])|(feat_binary_exp==train_dich[k][kk][1]))[0]
         ind_test=np.where((feat_binary_exp==test_dich[k][kk][0])|(feat_binary_exp==test_dich[k][kk][1]))[0]

         task=nan*np.zeros(len(feat_binary_exp))
         for i in range(4):
             ind_task=(feat_binary_exp==i)
             task[ind_task]=dichotomies[k][i]

         supp=LogisticRegression(C=reg,class_weight='balanced',solver='lbfgs')
         #supp=LinearSVC(C=reg,class_weight='balanced')
         mod=supp.fit(feat_decod[ind_train],task[ind_train])
         perf_ccgp[k,kk,0]=supp.score(feat_decod[ind_train],task[ind_train])
         perf_ccgp[k,kk,1]=supp.score(feat_decod[ind_test],task[ind_test])
         
    return perf_tasks,perf_ccgp



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