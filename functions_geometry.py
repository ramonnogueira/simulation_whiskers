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
    skf=StratifiedKFold(n_splits=n_cv)
    g=-1
    for train, test in skf.split(feat_decod,feat_binary[:,0]):
        g=(g+1)
        supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
        mod=supp.fit(feat_decod[train],feat_binary[:,0][train])
        perf_tasks_pre[g,0,0]=supp.score(feat_decod[train],feat_binary[:,0][train])
        perf_tasks_pre[g,0,1]=supp.score(feat_decod[test],feat_binary[:,0][test])

    # Variable 2
    skf=StratifiedKFold(n_splits=n_cv)
    g=-1
    for train, test in skf.split(feat_decod,feat_binary[:,1]):
        g=(g+1)
        supp=LogisticRegression(C=1,class_weight='balanced',solver='lbfgs')
        mod=supp.fit(feat_decod[train],feat_binary[:,1][train])
        perf_tasks_pre[g,1,0]=supp.score(feat_decod[train],feat_binary[:,1][train])
        perf_tasks_pre[g,1,1]=supp.score(feat_decod[test],feat_binary[:,1][test])

    # XOR
    skf=StratifiedKFold(n_splits=n_cv)
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



