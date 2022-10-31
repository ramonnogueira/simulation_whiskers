import os
import matplotlib.pylab as plt
import numpy as np
import scipy
import math
import sys
import tables
import pandas
import pickle as pkl
from scipy.stats import sem
from scipy.stats import pearsonr
from numpy.random import permutation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#import functions_miscellaneous
from sklearn.svm import LinearSVC
from scipy.stats import ortho_group 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import datetime
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
   pass
import warnings
warnings.warn = warn

#####################################
# Functions

def y_circ(x,r,pos0,amp,freq_sh):
    x0=pos0[0]
    y0=pos0[1]
    y_cnc=np.sqrt(r**2-(x-x0)**2)+y0+amp*np.cos(freq_sh*x)
    y_cnx=-np.sqrt(r**2-(x-x0)**2)+y0+amp*np.cos(freq_sh*x)
    return (y_cnc,y_cnx)

# Center of the circle when it is placed on the top right corner
def center0_func(r,z1):
    x_cnc=0.5*(z1+10-np.sqrt(-z1**2+20*z1-100+2*(r**2)))
    x_cnx=0.5*(z1+10+np.sqrt(-z1**2+20*z1-100+2*(r**2)))
    return ([x_cnc,x_cnc],[x_cnx,x_cnx])

def func_in_out_new(shape,wt,center,rad,stim,prob_poiss,amp,freq_sh):
    # Obtain the "shadow" region
    m_left=shape[1,1]/shape[1,0]
    m_right=shape[-2,1]/shape[-2,0]
    c_left=m_left*np.linspace(0,12,10)
    c_right=m_right*np.linspace(0,12,10)
    #If angle of whisker is bigger than left line or smaller than right line prob = 0 (no contact)  
    if wt[1]/wt[0]>m_left or wt[1]/wt[0]<m_right:
        prob=0    
    else: # If angle is between left and right lines
        dist=((wt[0]-center[0])**2+(wt[1]-center[1]-amp*np.cos(freq_sh*wt[0]))**2)
        if stim==0:
            if dist>=rad**2:
                prob=prob_poiss
            if dist<rad**2:
                prob=0
        if stim==1:
            if dist>rad**2:
                prob=0
            if dist<rad**2:
                prob=prob_poiss
    return prob,c_left,c_right

def rotation_center(center,theta):
    mat_rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.dot(mat_rot,center)
            
################################################################

# Path to save figures
#path_save='/home/ramon/Dropbox/chris_randy/plots/reviews/'
path_save='C:\\Users\\danie\\Documents\\simulation_whiskers\\results\\'
save_figs = True


##########################
# Simulation Parameters
# Whiskers
n_whisk=3
l_vec=np.linspace(10,7,n_whisk)
print (l_vec)
prob_poiss=1.01
noise_w=0.3
spread=1/n_whisk

# Time and movement
speed=2
ini_phase_m=0
ini_phase_spr=1e1
delay_time=0
freq_m=3
freq_std=0.1
std_reset=0

t_total=2
dt=0.1
t_vec=np.linspace(0,t_total,int(t_total/dt))
dx=0.01

n_trials_pre=2000
n_files=5

# Shape
amp=0
freq_sh=10
z1=4
disp=4.5 #(z1 4, disp 5.5 or 4.5),(z1 5, disp 3.5),(z1 6, disp 2)
rad_vec=np.logspace(np.log10(10-z1),np.log10(50),4)
#rad_vec=np.array([11])
n_trials=n_trials_pre*len(rad_vec)
theta=0 # not bigger than 0.3
steps_mov=np.array([9,10,11],dtype=np.int16) #10 and 11 is good for counts profile
concavity=np.array([0,1],dtype=np.int16)

models_vec=[(),(100),(100,100),(100,100,100)]
lr=1e-3
activation='relu'
reg=1e-3
n_cv=10
test_size=0.2
col_vec=['green','orange']
c_corr=[-1,1]
lab_vec=['Lin','NonLin1','NonLin2','NonLin3']

verbose=True

now=datetime.datetime.now()
datestr=now.strftime('%Y-%m-%d')
timestr=now.strftime('%H:%M:%S')

ini_phase=np.random.vonmises(ini_phase_m,ini_phase_spr,n_trials)
freq_whisk=np.random.normal(freq_m,freq_std,n_trials)
# plt.hist(ini_phase)
# plt.xlim(-np.pi,np.pi)
# plt.title('Ini phase')
# plt.show()
# plt.hist(freq_whisk)
# plt.title('Freq whisk')
# plt.xlim(0,6)
# plt.show()

# Create first figure of initial conditions
fig=plt.figure(figsize=(2,2))
ax=fig.add_subplot(111)
#functions_miscellaneous.adjust_spines(ax,['left','bottom'])

for i in range(15): # Loop across trials
    ind_stim=np.random.choice(concavity,replace=False)
    stim=ind_stim
    curv=np.random.choice(rad_vec,replace=False)
    timem=np.random.choice(steps_mov,replace=False)
    center0=center0_func(curv,z1)[ind_stim] # Center 0
    center1=(center0+c_corr[ind_stim]*disp/curv) # Center displaced
    center2=rotation_center(center1,c_corr[ind_stim]*theta) # Center rotated

    l=np.sqrt((z1-10)**2+(z1-10)**2)
    x_len=abs(l*np.cos(-np.pi/4+c_corr[ind_stim]*theta))
    x_shape_pre=np.linspace(5+0.5*z1-0.5*x_len,5+0.5*z1+0.5*x_len,int((10-z1)/0.01))
    x_shape=(x_shape_pre+c_corr[ind_stim]*disp/curv) 
    y_shape=y_circ(x_shape,curv,center2,amp,freq_sh)[ind_stim]
    shape=np.stack((x_shape,y_shape),axis=1)
    ax.scatter(shape[:,0],shape[:,1],color=col_vec[ind_stim],s=0.5,alpha=0.5)

    center_t=(center1-speed*timem*dt)
    x_shape2=(x_shape-speed*timem*dt)
    y_shape2=y_circ(x_shape2,curv,center_t,amp,freq_sh)[ind_stim]
    shape2=np.stack((x_shape2,y_shape2),axis=1)
    ax.scatter(shape2[:,0],shape2[:,1],color=col_vec[ind_stim],s=0.5)

#plt.axvline(0,color='black',linestyle='--')
#plt.plot(np.arange(60)-30,np.zeros(60),color='black',linestyle='--')
#plt.plot(np.arange(60)-30,np.arange(60)-30,color='black',linestyle='--')
#plt.plot(np.arange(60)-30,-np.arange(60)+30,color='black',linestyle='--')
angle_t=np.sin(freq_whisk[i]*t_vec+ini_phase[i])
for iii in range(n_whisk):
    nw=np.random.normal(0,noise_w,2)
    ang_inst=(-0.2+iii*spread)
    wt_pre=np.array([l_vec[iii]*np.cos(ang_inst),l_vec[iii]*np.sin(ang_inst)])
    wt=(wt_pre+nw)
    ax.plot([0,wt_pre[0]],[0,wt_pre[1]],color='black',alpha=(iii+1)/n_whisk)
    ax.scatter(wt[0],wt[1],color='black',alpha=(iii+1)/n_whisk)
if save_figs:
    frame_wiggles_fig_path = path_save+'model_reproduce_frame_wiggles.png'
    fig.savefig(frame_wiggles_fig_path,dpi=500,bbox_inches='tight')

perf_pre=nan*np.zeros((n_files,len(rad_vec),len(models_vec),n_cv,2))
lr_pre=nan*np.zeros((n_files,len(rad_vec),n_cv,2))
counts=nan*np.zeros((n_files,len(rad_vec),n_whisk))
for f in range(n_files):
    if verbose:
        print ('Running file {} out of {}...'.format(f, n_files))
    ini_phase=np.random.vonmises(ini_phase_m,ini_phase_spr,n_trials)
    freq_whisk=np.random.normal(freq_m,freq_std,n_trials)
    curvature=nan*np.zeros(n_trials)
    time_mov=nan*np.zeros(n_trials)
    stimulus=nan*np.zeros(n_trials)
    
    features=np.zeros((n_trials,len(t_vec),2*n_whisk))
    #features=np.zeros((n_trials,len(t_vec),n_whisk))
    for i in range(n_trials): # Loop across trials
        if verbose and np.remainder(i,100)==0:    
            print ('    Simulating trial {} out of {}...'.format(i, n_trials))
        ind_stim=np.random.choice(concavity,replace=False)
        stimulus[i]=ind_stim
        curvature[i]=np.random.choice(rad_vec,replace=False)
        time_mov[i]=np.random.choice(steps_mov,replace=False)
        #print (stimulus[i],curvature[i],time_mov[i])
        #print (ini_phase[i],freq_whisk[i])
        # Create shape t=0
        center0=center0_func(curvature[i],z1)[ind_stim]
        center1=(center0+c_corr[ind_stim]*disp/curvature[i])
        center2=rotation_center(center1,c_corr[ind_stim]*theta)
        
        l=np.sqrt((z1-10)**2+(z1-10)**2)
        x_len=abs(l*np.cos(-np.pi/4+c_corr[ind_stim]*theta))
        x_shape_pre=np.linspace(5+0.5*z1-0.5*x_len,5+0.5*z1+0.5*x_len,int((10-z1)/0.01))
        x_shape=(x_shape_pre+c_corr[ind_stim]*disp/curvature[i]) 
        y_shape=y_circ(x_shape,curvature[i],center2,amp,freq_sh)[ind_stim]
        shape=np.stack((x_shape,y_shape),axis=1)

        for ii in range(len(t_vec)): # Loop across time steps
            #print ('Step ',t_vec[ii])
            #plt.scatter(shape[:,0],shape[:,1],color='black',s=1)
            # Shape
            # if ii==0:
            #     angle_t=np.sin(ini_phase[i])
            # else:
            #     if np.sum(features[i,ii-1])!=0:
            #         angle_t=np.sin(ini_phase_m+np.random.normal(0,std_reset))
            #     else:
            #         angle_t=np.sin(freq_whisk[i]*dt+angle_t)
            angle_t=np.sin(freq_whisk[i]*t_vec[ii]+ini_phase[i])
            
            if  (ii>=delay_time) and ii<(time_mov[i]+delay_time):
                center2=(center2-speed*dt)
                x_shape=(x_shape-speed*dt)
                y_shape=y_circ(x_shape,curvature[i],center2,amp,freq_sh)[ind_stim]
                shape=np.stack((x_shape,y_shape),axis=1)
                
            # Whisker
            for iii in range(n_whisk):
                nw=np.random.normal(0,noise_w,2)
                ang_inst=(angle_t+iii*spread)
                wt_pre=np.array([l_vec[iii]*np.cos(ang_inst),l_vec[iii]*np.sin(ang_inst)])
                wt=(wt_pre+nw)
                prob,c1,c2=func_in_out_new(shape,wt,center2,curvature[i],ind_stim,prob_poiss,amp,freq_sh)
                ct_bin=int(np.random.uniform(0,1)<prob)
                features[i,ii,2*iii]=ct_bin
                #features[i,ii,iii]=ct_bin
                if ct_bin==1:
                    features[i,ii,2*iii+1]=ang_inst
                #plt.plot([0,wt_pre[0]],[0,wt_pre[1]],color='green',alpha=(iii+1)/n_whisk)
            #     #plt.plot(np.linspace(0,12,10),c1,color='black',linestyle='--')
            #     #plt.plot(np.linspace(0,12,10),c2,color='black',linestyle='--')
            #     plt.scatter(wt[0],wt[1],color='green',alpha=(iii+1)/n_whisk)
            # print (iii,features[i,ii])
            # plt.xlim([0,12])
            # plt.ylim([0,12])
            # plt.show()

    # Classifier
    if verbose:
        print('    Training classifiers...')
    feat_class=np.reshape(features,(len(features),-1))
    #feat_class=np.sum(features,axis=1)
    # MLP
    for i in range(len(rad_vec)):
        #print (i)
        ind_rad=np.where((curvature==rad_vec[i]))[0]
        for j in range(len(models_vec)):
            if verbose:
                print('        Training NonLin-{} classifier for curvature={}....'.format(j, rad_vec[i]))
            skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
            g=0
            for train,test in skf.split(feat_class[ind_rad],stimulus[ind_rad]):
                mod=MLPClassifier(models_vec[j],learning_rate_init=lr,alpha=reg,activation=activation)
                mod.fit(feat_class[ind_rad][train],stimulus[ind_rad][train])
                perf_pre[f,i,j,g,0]=mod.score(feat_class[ind_rad][train],stimulus[ind_rad][train])
                perf_pre[f,i,j,g,1]=mod.score(feat_class[ind_rad][test],stimulus[ind_rad][test])
                g=(g+1)
    # Log regress
    for i in range(len(rad_vec)):
        #print (i)
        ind_rad=np.where((curvature==rad_vec[i]))[0]
        skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
        g=0
        if verbose:
            print('        Training linear classifier for curvature={}....'.format(j, rad_vec[i]))
        for train,test in skf.split(feat_class[ind_rad],stimulus[ind_rad]):
            mod=LogisticRegression(C=1/reg)
            #mod=LinearSVC()
            mod.fit(feat_class[ind_rad][train],stimulus[ind_rad][train])
            lr_pre[f,i,g,0]=mod.score(feat_class[ind_rad][train],stimulus[ind_rad][train])
            lr_pre[f,i,g,1]=mod.score(feat_class[ind_rad][test],stimulus[ind_rad][test])
            g=(g+1)

    #print (np.mean(perf_pre,axis=(0,3)))
    #print (np.mean(lr_pre,axis=(0,2)))

    # Counts XOR
    col_vec=['green','orange']
    # pair_vec=[[0,1],[0,2],[1,2]]
    # for g in range(len(pair_vec)):
    #     fig=plt.figure(figsize=(2,2))
    #     ax=fig.add_subplot(111)
    #     functions_miscellaneous.adjust_spines(ax,['left','bottom'])
    #     pair=pair_vec[g]
    #     for i in range(2):
    #         index=np.where(stimulus==i)[0]
    #         ax.scatter(np.sum(features,axis=1)[index,pair[0]]+np.random.normal(0,0.1,len(index)),np.sum(features,axis=1)[index,pair[1]]+np.random.normal(0,0.1,len(index)),color=col_vec[i],alpha=0.6,s=0.1)
    #     m0=np.mean(np.sum(features,axis=1)[stimulus==0],axis=0)
    #     m1=np.mean(np.sum(features,axis=1)[stimulus==1],axis=0)
    #     print (m0)
    #     print (m1)
    #     print (np.linalg.norm(m0-m1))
    #     ax.scatter(m0[pair[0]],m0[pair[1]],color='green',s=10)
    #     ax.scatter(m1[pair[0]],m1[pair[1]],color='orange',s=10)
    #     ax.set_xlabel('Contacts C%i'%(pair[0]))
    #     ax.set_ylabel('Contacts C%i'%(pair[1]))
    #     #fig.savefig('/home/ramon/Dropbox/chris_randy/plots/reviews/contacts_C%i_C%i_prueba.png'%(int(pair[0]/2)+1,int(pair[1]/2)+1),dpi=500,bbox_inches='tight')
    #     fig.savefig('/home/ramon/Dropbox/chris_randy/plots/reviews/contacts_C%i_C%i_prueba.png'%(pair[0],pair[1]),dpi=500,bbox_inches='tight')
                
    # # Counts
    # for i in range(len(rad_vec)):
    #     #print (rad_vec[i])
    #     ind_rad0=np.where((curvature==rad_vec[i])&(stimulus==0))[0]
    #     ind_rad1=np.where((curvature==rad_vec[i])&(stimulus==1))[0]
    #     for ii in range(n_whisk):
    #         counts[f,i,ii]=np.mean(np.sum(features[ind_rad1,:,2*ii],axis=1))-np.mean(np.sum(features[ind_rad0,:,2*ii],axis=1))
        

perf=np.mean(perf_pre,axis=3)
perf_m=np.mean(perf,axis=0)
perf_sem=sem(perf,axis=0)
print (perf_m)

perf_lr=np.mean(lr_pre,axis=2)
lr_m=np.mean(perf_lr,axis=0)
lr_sem=sem(perf_lr,axis=0)
print (lr_m)

# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')
# for jj in range(2):
#     index=np.where(stimulus==jj)[0]
#     ax.scatter(np.sum(features,axis=1)[index][:,0]+np.random.normal(0,0.1,len(index)),np.sum(features,axis=1)[index][:,1]+np.random.normal(0,0.1,len(index)),np.sum(features,axis=1)[index][:,2]+np.random.normal(0,0.1,len(index)),color=col_vec[jj],s=1,alpha=0.5)
# ax.set_xlabel('C1')
# ax.set_ylabel('C2')
# ax.set_zlabel('C3')
# plt.show()

# Cuidado!
perf_m[:,0]=lr_m
perf_sem[:,0]=lr_sem

# Perf Curvature
for j in range(len(models_vec)):
    if j==0:
        plt.errorbar(rad_vec,perf_m[:,j,1],yerr=perf_sem[:,j,1],color='orange',label=lab_vec[j])
    else:
        plt.errorbar(rad_vec,perf_m[:,j,1],yerr=perf_sem[:,j,1],color='green',alpha=(j+1)/len(models_vec),label=lab_vec[j])
plt.plot(rad_vec,0.5*np.ones(len(rad_vec)),color='black',linestyle='--')
plt.xscale('log')
plt.xlabel('Curvature (Rad)')
plt.ylabel('Performance')
plt.legend(loc='best')
plt.ylim([0.4,1])
plt.show()
if save_figs:
    perf_v_curv_fig_path = path_save+'performance_v_curvature.pdf'
    fig.savefig(perf_v_curv_fig_path,dpi=500,bbox_inches='tight')

###################################
# Fig 2
model_labels=['Linear','NonLin-1','NonLin-2','NonLin-3']
alpha_vec=[0.4,0.6,0.8,1.0]
width=0.15

fig=plt.figure(figsize=(2,2))
ax=fig.add_subplot(111)
#functions_miscellaneous.adjust_spines(ax,['left','bottom'])
ax.plot([-3.5*width,3.5*width],0.5*np.ones(2),color='black',linestyle='--')
#plt.xticks(width*np.arange(len(models_vec))-1.5*width,model_labels,rotation='vertical')
for j in range(1,len(models_vec)):
    ax.bar(j*width-1.5*width,perf_m[0,j,1],yerr=perf_sem[0,j,1],color='green',width=width,alpha=alpha_vec[j])
    #ax.scatter(j*width-1.5*width+p+np.random.normal(0,std_n,3),perf[:,p,j,1],color='black',alpha=alpha_vec[j],s=4)
ax.bar(-1.5*width,lr_m[0,1],yerr=lr_sem[0,1],color='green',width=width,alpha=alpha_vec[0])
ax.set_ylim([0.4,1.0])
#ax.set_xlim([-3.5*width,3.5*width])
ax.set_ylabel('Decoding Performance')
if save_figs:
    model_rep_beh_path = path_save+'model_reproduce_behavior_wiggles.pdf'
    fig.savefig(model_rep_beh_path,dpi=500,bbox_inches='tight')
    
    # Save metadata:
    metadata = dict()
    metadata['params']['n_whisk']=n_whisk
    metadata['params']['prob_poiss']=prob_poiss
    metadata['params']['noise_w']=noise_w
    metadata['params']['spread']=spread
    
    metadata['params']['speed']=speed
    metadata['params']['ini_phase_m']=ini_phase_m
    metadata['params']['ini_phase_spr']=ini_phase_spr
    metadata['params']['delay_time']=delay_time
    metadata['params']['freq_m']=freq_m
    metadata['params']['freq_std']=freq_std
    metadata['params']['std_reset']=std_reset
    
    metadata['params']['t_total']=t_total
    metadata['params']['dt']=dt
    metadata['params']['dx']=dx
    
    metadata['params']['n_trials_pre']=n_trials_pre
    metadata['params']['n_files']=n_files
    
    metadata['params']['amp']=amp
    metadata['params']['freq_sh']=freq_sh
    metadata['params']['z1']=z1
    metadata['params']['disp']=disp #(z1 4, disp 5.5 or 4.5),(z1 5, disp 3.5),(z1 6, disp 2)
    metadata['params']['rad_vec']=rad_vec
    metadata['params']['theta']=theta # not bigger than 0.3
    metadata['params']['steps_mov']=steps_mov
    
    metadata['params']['models_vec']=models_vec
    metadata['params']['lr']=lr
    metadata['params']['activation']=activation
    metadata['params']['reg']=reg
    metadata['params']['n_cv']=n_cv
    metadata['params']['test_size']=test_size

    metadata['outputs'][0]['path']= frame_wiggles_fig_path
    metadata['outputs'][1]['path']=perf_v_curv_fig_path
    metadata['outputs'][2]['path']=model_rep_beh_path
    
    metadata['date']=datestr
    metadata['time']=timestr

# #######################################
# # counts
# counts_m=np.mean(counts,axis=0)
# counts_sem=sem(counts,axis=0)
# fig=plt.figure(figsize=(4,4))
# width=0.5
# ax=fig.add_subplot(2,2,1)
# functions_miscellaneous.adjust_spines(ax,['left','bottom'])
# ax.set_ylabel('Contact Difference\n Convex - Concave')
# ax.set_xlim([-0.5,2.5])
# ax.set_yticks([-1,0,1])
# plt.xticks([0,1,2],['C1','C2','C3'])
# ax.bar(np.arange(n_whisk),counts_m[i],yerr=counts_sem[i],color=['blue','green','red'],width=width)
# ax.plot(np.arange(5)-1,np.zeros(5),color='black',linestyle='--')
# fig.savefig(path_save+contacts_reproduce_behavior.pdf',dpi=500,bbox_inches='tight')

# # Counts
# counts_m=np.mean(counts,axis=0)
# counts_sem=sem(counts,axis=0)
# width=0.2
# #plt.bar(l_vec,counts_m[i],yerr=counts_sem[i],color=['blue','green','red'],width=width)
# plt.bar(np.arange(n_whisk),counts_m[i],yerr=counts_sem[i],color=['blue','green','red'],width=width)
# #plt.plot(np.arange(5)-1,np.zeros(5),color='black',linestyle='--')
# #plt.xlim([-0.5,2.5])
# #plt.xticks([0,1,2],['C1','C2','C3'])
# #plt.ylim([-1.5,1.5])
# #plt.yticks([-1,0,1])
# plt.ylabel('Contact Difference\n Convex - Concave')
# plt.show()

# #Perf vs time
# # print ('Perf time')
# # n_cv=10
# # for i in range(len(rad_vec)):
# #     print (i)
# #     perf_time=nan*np.zeros((len(features[0]),n_cv,2))
# #     ind_rad=np.where((curvature==rad_vec[i]))[0]
# #     for ii in range(1,len(features[0])):
# #         feat_class=np.reshape(features[:,0:ii],(len(features),-1))
# #         skf=StratifiedShuffleSplit(n_cv,0.2)
# #         g=0
# #         for train,test in skf.split(feat_class[ind_rad],stimulus[ind_rad]):
# #             mod=MLPClassifier(models_vec[0],learning_rate_init=lr,alpha=reg,activation=activation)
# #             mod.fit(feat_class[ind_rad][train],stimulus[ind_rad][train])
# #             perf_time[ii,g,0]=mod.score(feat_class[ind_rad][train],stimulus[ind_rad][train])
# #             perf_time[ii,g,1]=mod.score(feat_class[ind_rad][test],stimulus[ind_rad][test])
# #             g=(g+1)
# #     perf_time_m=np.mean(perf_time,axis=1)
# #     plt.plot(np.arange(20),perf_time_m[:,1],color='green',alpha=(i+1)/len(rad_vec))
# # plt.plot(np.arange(20),0.5*np.ones(20),color='black',linestyle='--')
# # plt.ylim([0.4,1.0])
# # plt.ylabel('Prob. Correct Lick')
# # plt.xlabel('Time')
# # plt.show()


