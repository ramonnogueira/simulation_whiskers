import os
from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np
import scipy
import math
import sys
import tables
import pandas as pd
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
import json
import pathlib
import argparse
nan=float('nan')
minf=float('-inf')
pinf=float('inf')
def warn(*args, **kwargs):
   pass
import warnings
warnings.warn = warn
try:
    from analysis_metadata.analysis_metadata import Metadata, write_metadata
except ImportError or ModuleNotFoundError:
    analysis_metdata_imported=False

#####################################
# Functions

def y_circ(x,r,pos0,amp,freq_sh):
    """
    Compute y-coordinates for stimulus.

    Parameters
    ----------
    x : numpy.ndarray
        s-element array, where s in the number of points defining the stimulus.
    
    r : float
        Radius of circle inscribing stimulus arc.
    
    pos0 : numpy.ndarray
        2-element array defining center of circle inscribing stimulus arc.
    
    amp : float
        Amplitude of stimulus texture.
    
    freq_sh : float
        Frequency of stimulus texture.

    Returns
    -------
    y_cnc : numpy.ndarray
        s-element array of y-coordinates for concave stimulus.
        
    y_cnx : TYPE
        s-element array of y-coordinates for convex stimulus.

    """
    x0=pos0[0]
    y0=pos0[1]
    y_cnc=np.sqrt(r**2-(x-x0)**2)+y0+amp*np.cos(freq_sh*x)
    y_cnx=-np.sqrt(r**2-(x-x0)**2)+y0+amp*np.cos(freq_sh*x)
    return (y_cnc,y_cnx)



# Center of the circle when it is placed on the top right corner
def center0_func(r,z1):
    """
    Compute center of circle inscribing stimulus arc.  

    Parameters
    ----------
    r : float
        Radius.
        
    z1 : float
        Controls angle subtended by stimulus arc 
        [TODO: check this is accurate; better way of describing this?].

    Returns
    -------
    Pair of lists; first element defines center of circle for concave stimulus.
    second element defines center of circle for convex stimulus.

    """
    x_cnc=0.5*(z1+10-np.sqrt(-z1**2+20*z1-100+2*(r**2))) #concave
    x_cnx=0.5*(z1+10+np.sqrt(-z1**2+20*z1-100+2*(r**2))) #convex
    return ([x_cnc,x_cnc],[x_cnx,x_cnx])



def func_in_out_new(shape,wt,center,rad,stim,prob_poiss,amp,freq_sh):
    """
    Compute probability of contact for a given whisker at a given time step.

    Parameters
    ----------
    shape : numpy.ndarray
        s-by-2 array describing stimulus shape, where s is the number of points
        defining the stimulus. First row are x-coordinates, second row are y-
        coordinates [TODO: double-check this order is correct?].
    
    wt : numpy.ndarray
        2-element array defining x- and y-coordinate of whisker tip.
    
    center : numpy.ndarray
        2-element array defining center of circle of which stimulus is an arc.
        
    rad : float
        Radius of circle defining stimulus.
        
    stim : bool 
        Whether stimulus is concave or convex.
    
    prob_poiss : float
        Probability of contact given the whisker intersects the shape.
        
    amp : float
        Amplitude of stimulus texture (sinusoid added to arc).
        
    freq_sh : float
        Frequency of stimulus texture (sinusoid added to arc).

    Returns
    -------
    prob : float
        Probability of whisker contact.
        
    c_left : numpy.ndarray
        Line from center of whisker pad (origin) to left edge of stimulus.
        
    c_right : numpy.ndarray
        Line from center of whisker pad (origin) to right edge of stimulus.

    """
    
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
    """
    Define transformation matrix for rotating point around origin.

    Parameters
    ----------
    center : numpy.ndarray
        2-element array of coordinates to rotate around origin.
        
    theta : float
        Angle to rotate point by.

    Returns
    -------
    mat_rot : numpy.ndarray
        Rotation matrix.
        
    center : numpy.ndarray
        Copy of input coorindates.

    """
    mat_rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    return np.dot(mat_rot,center)



def offset_shape(shape,scale):
    # Find max distance:
    distances=[x[0]**2+x[1]**2 for x in shape]
    max_distance=np.max(distances)
    
    # Compute and apply offset:
    offset=max_distance*(1-scale)
    shape_out=shape+offset
    
    return shape_out
            


def generate_default_sim_params():
    """
    Define default simulation parameters and decoder hyperparameters. Call from
    compare_stim_decoders() when parameters are not specified by user. 

    Paramters
    ---------
    None. 

    Returns
    -------
    default_params : dict
        Dict of default simulation parameters and decoder hyperparameters.

    """
    default_params = {
        
        # Simulation parameters:
        'n_whisk':3,
        'prob_poiss':1.01,
        'noise_w': 0.3,
        'spread': 'auto',
        
        # Time and movement:
        'speed': 2,
        'ini_phase_m': 0,
        'ini_phase_spr': 1e1,
        'delay_time': 0,
        'freq_m': 3,
        'freq_std': 0.1,
        'std_reset': 0,
        't_total': 2,
        'dt': 0.1,
        'dx': 0.01,
        'n_trials_pre': 2000,
        'n_files': 5,
        
        # Shape:
        'amp': 0,
        'freq_sh': 10,
        'z1': 4, 
        'disp': 4.5,
        'theta': 0,
        'steps_mov': [9,10,11],
        'max_rad': 50,
        'n_rad': 4,
        
        # Classifier parameters:
        'models_vec': [(),(100),(100,100),(100,100,100)],
        'lr': 1e-3,
        'activation': 'relu',
        'reg': 1e-3,
        'n_cv': 10,
        'test_size': 0.2
        }
    return default_params



def generate_default_mlp_hparams():
    """
    Define default MLP hyperparameters. Call from compare_stim_decoders() when 
    hyperparameters are not specified by user. 

    Paramters
    ---------
    None. 

    Returns
    -------
    default_params : dict
        Dict of default MLP hyperparameters.

    """
    default_params = {
        # Classifier parameters:
        'models_vec': [(),(100),(100,100),(100,100,100)],
        'lr': 1e-3,
        'activation': 'relu',
        'reg': 1e-3,
        'n_cv': 10,
        'test_size': 0.2
        }
    return default_params



def illustrate_stimuli(hparams=None, rows=None, labels=None, stim=None, n_stim=15, save_figs=False, output_directory=None, fig_name=None):
    """
    Plot illustration of whiskers and random example stimuli. 

    Parameters
    ----------
    hparams : str | dict
        Same as for compare_stim_decoders().
    
    save_figs : bool, optional
        Whether to save figures. 
    
    output_directory : str, optional
        Directory to save figures in. 

    Returns
    -------
    fig : matplotlib.figure.Figure
        Illustration of whiskers and random example stimuli. Stimuli vary by
        convexity and curvature. Each stimulus is plotted twice, once at its 
        initial position then again at its final position. 

    """
    
    # Load hyperparameters:
    h = load_sim_params(hparams)
    
    # Assign loaded hyperparameters to local variables:
    
    # Simulation parameters:
    n_whisk=h['n_whisk']    
    noise_w=h['noise_w']
    spread=h['spread']
    
    # Time and movement:    
    speed=h['speed']
    ini_phase_m=h['ini_phase_m']
    ini_phase_spr=h['ini_phase_spr']
    freq_m=h['freq_m']
    freq_std=h['freq_std']
    t_total=h['t_total']
    dt=h['dt']
    dx=h['dx']
    n_trials_pre=h['n_trials_pre']
    steps_mov=h['steps_mov']
    steps_mov=np.array(h['steps_mov'],dtype=np.int16)
    init_position=h['init_position']
    
    # Shape:
    amp=h['amp']
    freq_sh=h['freq_sh']
    z1=h['z1']
    disp=h['disp']
    theta=h['theta']
    max_rad=h['max_rad']
    n_rad=h['n_rad']
    rad_vec=h['rad_vec']
    concavity=h['concavity']
    
    # Define misc. necessary constants:
    l_vec=np.linspace(10,7,h['n_whisk'])
    if spread=='auto':
        spread=1/n_whisk
    t_vec=np.linspace(0,t_total,int(t_total/dt)) 
    #concavity=np.array([0,1],dtype=np.int16)    
    #rad_vec=np.logspace(np.log10(10-h['z1']),np.log10(max_rad),n_rad)
    col_vec=['green','orange']
    c_corr=[-1,1]
    n_trials=n_trials_pre*len(rad_vec)
    
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
    
    # Define rows if not supplied with input:
    if rows is None:
        rows=[]
        for i in range(n_stim):
            curr_dict=dict()
            if stim==None:
                curr_dict['stimulus']=np.random.choice(concavity,replace=False)
            else:
                curr_dict['stimulus']=stim
            curr_dict['curvature']=np.random.choice(rad_vec,replace=False)
            curr_dict['time_mov']=np.random.choice(steps_mov,replace=False)
            curr_dict['z1']=np.random.choice(z1,replace=False)
            curr_dict['theta']=np.random.choice(theta,replace=False)
            curr_dict['freq_sh']=np.random.choice(freq_sh,replace=False)            
            rows.append(curr_dict)
        
        
    # Iterate over trials to illustrate:
    
    for r in np.arange(len(rows)): # Loop across trials #TODO: make this a parameter
        
        # Select color for current trial if applicable:
        if labels is not None:
            curr_color=col_vec[int(labels[r])]
        
        if type(rows)==list:
            r2=rows[r]
        elif type(rows)==pd.core.frame.DataFrame:
            r2=rows.iloc[r]
        
        # Illustrate stimulus for current trial:
        illustrate_stimulus(ax, r2['stimulus'], r2['curvature'], r2['z1'], init_position, r2['time_mov'], speed, dt, r2['theta'], disp, amp, r2['freq_sh'], color=curr_color)

        """
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
        """
    
    #plt.axvline(0,color='black',linestyle='--')
    #plt.plot(np.arange(60)-30,np.zeros(60),color='black',linestyle='--')
    #plt.plot(np.arange(60)-30,np.arange(60)-30,color='black',linestyle='--')
    #plt.plot(np.arange(60)-30,-np.arange(60)+30,color='black',linestyle='--')
    for iii in range(n_whisk):
        nw=np.random.normal(0,noise_w,2)
        ang_inst=(-0.2+iii*spread)
        wt_pre=np.array([l_vec[iii]*np.cos(ang_inst),l_vec[iii]*np.sin(ang_inst)])
        wt=(wt_pre+nw)
        ax.plot([0,wt_pre[0]],[0,wt_pre[1]],color='black',alpha=(iii+1)/n_whisk)
        ax.scatter(wt[0],wt[1],color='black',alpha=(iii+1)/n_whisk)
    if save_figs:
        if fig_name==None:
            fig_name='model_reproduce_frame_wiggles.png'
        # If requested output directory does not exist, create it:
        if output_directory == None:
            output_directory = os.getcwd()
        elif not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        frame_wiggles_fig_path = os.path.join(output_directory,fig_name)
        fig.savefig(frame_wiggles_fig_path,dpi=500,bbox_inches='tight')
    
    return fig



def illustrate_stimulus(ax, ind_stim, curv, z1, init_position, timem, speed, dt, theta, 
                        disp, amp, freq_sh, color=None):
    """
    Generate plot of a single stimulus at the beginning and end of its 
    movement.

    Parameters
    ----------
    ax : matplotlib.axes_subplots.AxesSubplot
        Axes to plot stimulus in.
    
    ind_stim : 0 | 1
        Stimulus index; 0: concave, 1: convex.
    
    curv : numpy.float64
        Stimulus curvature.
    
    z1 : float
        Inverse stimulus arc length (higher z->shorter arc length).
    
    timem : int
        Number of time steps stimulus moves. 
    
    speed : float 
        Stimulus speed.
    
    dt : float
        Time step size.
    
    theta : float
        DESCRIPTION.
    
    disp : TYPE
        DESCRIPTION.
    
    amp : float
        Stimulus texture amplitude.
    
    freq_sh : float
        Stimulus texture frequency.


    Returns
    -------
    None.

    """   
    
    if color==None:
        col_vec=['green','orange']
        color=col_vec[ind_stim]
    c_corr=[-1,1]

    stim=ind_stim
    corr=c_corr[ind_stim]
    center0=center0_func(curv,z1)[ind_stim] # Center 0
    center1=(center0+corr*disp/curv) # Center displaced
    center1=(center1-2*init_position*dt) # Center displaced
    center2=rotation_center(center1,corr*theta) # Center rotated

    l=np.sqrt((z1-10)**2+(z1-10)**2)
    x_len=abs(l*np.cos(-np.pi/4+corr*theta))
    x_shape_pre=np.linspace(5+0.5*z1-0.5*x_len,5+0.5*z1+0.5*x_len,int((10-z1)/0.01))
    x_shape=(x_shape_pre+corr*disp/curv) 
    x_shape=(x_shape-2*init_position*dt) # Center displaced
    y_shape=y_circ(x_shape,curv,center2,amp,freq_sh)[ind_stim]
    shape=np.stack((x_shape,y_shape),axis=1)
    ax.scatter(shape[:,0],shape[:,1],color=color,s=0.5,alpha=0.5)

    center_t=(center1-speed*timem*dt)
    x_shape2=(x_shape-speed*timem*dt)
    y_shape2=y_circ(x_shape2,curv,center_t,amp,freq_sh)[ind_stim]
    shape2=np.stack((x_shape2,y_shape2),axis=1)
    ax.scatter(shape2[:,0],shape2[:,1],color=color,s=0.5)
    


def compare_stim_decoders(sim_params, mlp_hparams, task, sum_bins=False, plot_train=False, save_figs=False, output_directory=None, verbose=False):
    """
    Train and test one or more decoders (logistic regression or MLP) on a 
    simulated shape discrimination task. 

    Parameters
    ----------
    sim_params : str | dict, optional
        Simulation parameters. If str, should be path to a JSON file encoding 
        relevant variables; if dict, should define one key for each 
        parameter/hyperparameter. See example_sim_params.json file in this repo 
        for example. Should define the following keys:
            
            n_whisk : int
                Number of whiskers. 
                
            prob_poiss : float
                Probability with which a whisker intersecting the stimulus will 
                count as a contact.
                
            noise_w : float
                Standard deviation of normal distribution describing whisker
                position.
                
            spread : float | 'auto'
                Number describing how far apart the whiskers should be spaced.
                If 'auto', spread will just be set to 1/n_whisk.
                
            speed : float
                Distance moved by stimulus per time step.
                
            ini_phase_m : float
                Mean whisker phase at beginning of trial.
                
            ini_phase_spr: float
                Dispersion of whisker phase at beginning of trial.
                
            delay_time : float
                Amount of time to wait at beginning of each trial before moving
                stimulus.
                
            freq_m : float
                Mean whisking frequency. TODO: confirm units; Hz?
                
            freq_std : float
                Standard deviation of normal distribution describing whisking
                frequency across trials. 
                
            std_reset: float
                TODO: looks like this isn't actually used anywhere; consider 
                eliminating?
                
            t_total : int
                Trial duration.
                
            dt : float
                Time step size (same units as t_total).
                
            dx : float
                TODO: looks like this isn't actually used anywhere; consider 
                eliminating?
                
            n_trials_pre : int
                Trials per curvature condition. 
                
            n_files : int
                Number of separate behavioral sessions to simulate. 
                
            amp : float
                Texture amplitude (sinusoid added to curved concave/convex
                shape stimulus).
                
            freq_sh : float 
                Texture frequency (sinusoid added to curved concave/convex
                shape stimulus).
                
            z1 : float
                Controls overall stimulus size. TODO: confirm this, get better
                understanding of how this works?
                
            disp : float
                TODO: add better explanation for this. 
                
            theta : float
                TODO: add better explanation for this. 

            steps_mov : list
                List of int. Each element is a possible number of time steps
                the stimulus can move on a given trial. Selected from at random
                on each trial.
                
    mlp_hparams : str | dict, optional
        MLP hyperparameters. If str, should be path to a JSON file encoding 
        relevant variables; if dict, should define one key for each 
        parameter/hyperparameter. See example_mlp_params.json file in this repo 
        for example. Should define the following keys:
            
            models_vec : list
                List of lists and int. Each element corresponds to a different
                model to train and test on the stimulated whisker task data. An
                empty list corresponds to a linear model (logistic regression);
                a scalar corresponds to a multilayer perceptron (MLP) with one 
                intermediate layer; a list with n>1 elements corresponds to an
                MLP with n intermediate layers.
                
            lr : float
                Learning rate for MLPs.
                
            activation : str
                Activation function for MLP units.
                
            reg: float 
                Regularization parameter for MLPs.
                
            n_cv : int
                Number of cross-validations for linear and nonlinear models.
                
            test_size : float
                Fraction of trials to hold out as test.

    save_figs : bool, optional
        Whether to save figures to disk. 
    
    output_directory : str, optional
        Where to save figures if `save_figs` is True. Will save to current
        working directory by default. 
    
    verbose : bool, optional
        Whether to display status messages while running simulation and 
        decoders. 

    Returns
    -------
    None, but generates the following figures and saves to disk if requested:
        
        Fig 1: Illustration of random examples of simulated stimuli and 
        whiskers.
        
        Fig 2: Decoder performance vs. curvature for all decoder types tested.
        
        Fig 3: Bar graph of overall decoding performance for different decoder
        types tested (averaged across curvatures).
        
    Also, if `save_figs` is True, saves JSON file of analysis metadata to disk
    in same directory as figures. 

    """
    
    plt.ion()
    now=datetime.datetime.now()
    
    # Load parameters/hyperparameters:
    h = load_sim_params(sim_params)
    decoder_hparams = load_mlp_hparams(mlp_hparams)
    task = load_task_def(task)
    
    # Decide whether to separately train and test decoders for different 
    # curvatures; obviously can't do this if decoding curvature itself:
    split_by_curvature = not np.any(['curvature' in x for x in task])

    # Define/create output directory if necessary:
    if save_figs:
        # If no output directory defined, just use current directory:
        if output_directory==None:
            output_directory = os.getcwd()
        # If requested output directory does not exist, create it:
        elif not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Simulation parameters:
    n_whisk=h['n_whisk']
    spread=h['spread']
    
    # Time and movement:
    n_files=h['n_files']
    
    # Shape:
    z1=h['z1']
    steps_mov=h['steps_mov']
    max_rad=h['max_rad']
    n_rad=h['n_rad']
    rad_vec=h['rad_vec']
    init_position=h['init_position']
    
    # Classifier parameters:
    models_vec=decoder_hparams['models_vec']
    lr=decoder_hparams['lr']
    activation=decoder_hparams['activation']
    mlp_reg=decoder_hparams['mlp_reg']
    lr_reg=decoder_hparams['lr_reg']
    n_cv=decoder_hparams['n_cv']
    test_size=decoder_hparams['test_size']
    if [] in models_vec:
        models_vec.remove([]) # remove redundant empty list for linear model; will run this anyway

    # Generate various necessary arrays, variables from loaded hyperparameters:
    #rad_vec=np.logspace(np.log10(10-z1),np.log10(max_rad),n_rad)
    h['rad_vec']=rad_vec
    col_vec=['green','orange']
    lab_vec=define_model_labels(models_vec)
    lab_vec=['Lin']+lab_vec
    steps_mov=np.array(h['steps_mov'],dtype=np.int16)
    
    # Initialize results arrays:    
    if split_by_curvature:
        perf_pre=nan*np.zeros((n_files,len(rad_vec),len(models_vec),n_cv,2))
        lr_pre=nan*np.zeros((n_files,len(rad_vec),n_cv,2))
        # If summing over bins as well:
    else:
        perf_pre=nan*np.zeros((n_files,len(models_vec),n_cv,2))
        lr_pre=nan*np.zeros((n_files,n_cv,2))

    # Initialized additional results arrays if also summing over time bins:
    if sum_bins:
        perf_pre_summed=deepcopy(perf_pre)
        lr_pre_summed=deepcopy(lr_pre)
    else:
        perf_pre_summed=None
        lr_pre_summed=None
    
    # Iterate over files:
    for f in range(n_files):
        if verbose:
            print ('Running file {} out of {}...'.format(f, n_files))

        #features, curvature, stimulus=simulate_session(h, rad_vec, verbose=verbose)
        
        # Simulate session:
        session = simulate_session(h, sum_bins=sum_bins, verbose=verbose)
        
        # Extract labels:
        labels = session2labels(session, task, label_all_trials=False)
        
        # Exclude trials that don't match conditions in task:
        keep_indices = ~np.isnan(labels)
        session = session[keep_indices]
        labels = labels[keep_indices]
        
        # Illustrate some example stimuli just for first file as a sanity check:
        if f==0:
            n_example_trials=15
            all_indices=np.arange(len(session))
            np.random.shuffle(all_indices)
            example_indices=all_indices[0:n_example_trials]
            example_trials=session.iloc[example_indices]
            example_labels=labels[example_indices]
            stimfig = illustrate_stimuli(hparams=h, rows=example_trials, labels=example_labels, save_figs=False)
            if save_figs:
                frame_wiggles_fig_path = os.path.join(output_directory, 'model_reproduce_frame_wiggles.png')
                stimfig.savefig(frame_wiggles_fig_path,dpi=500,bbox_inches='tight')
        
        # Reshape data:
        features = np.array(list(session['features']))
        feat_class=np.reshape(features,(len(features),-1))
        if sum_bins:
            features_summed = np.array(list(session['features_bins_summed']))
            feat_summed_class=np.reshape(features_summed,(len(features_summed),-1))    

        # Might delete later:            
        stimulus = np.array(session['stimulus'])
        curvature = np.array(session['curvature'])
    
        # Classifier
        if verbose:
            print('    Training classifiers...')
        #feat_class=np.sum(features,axis=1)
        # MLP
        if split_by_curvature: # If splitting MLPs by curvature:
            perf_axis=3
            for i in range(len(rad_vec)):
                #print (i)
                ind_rad=np.where((curvature==rad_vec[i]))[0]
                for j in range(len(models_vec)):
                    if verbose:
                        print('        Training NonLin-{} classifier for curvature={}....'.format(j+1, rad_vec[i]))
                    skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
                    g=0
                    for train,test in skf.split(feat_class[ind_rad],labels[ind_rad]):
                        mod=MLPClassifier(models_vec[j],learning_rate_init=lr,alpha=mlp_reg,activation=activation)
                        mod.fit(feat_class[ind_rad][train],labels[ind_rad][train])
                        perf_pre[f,i,j,g,0]=mod.score(feat_class[ind_rad][train],labels[ind_rad][train])
                        perf_pre[f,i,j,g,1]=mod.score(feat_class[ind_rad][test],labels[ind_rad][test])
                        g=(g+1)
                        
                    # If also computing performance summed across time bins:
                    if sum_bins:
                        g=0
                        skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
                        for train,test in skf.split(feat_summed_class[ind_rad],labels[ind_rad]):
                            mod_summed=MLPClassifier(models_vec[j],learning_rate_init=lr,alpha=mlp_reg,activation=activation)
                            mod_summed.fit(feat_summed_class[ind_rad][train],labels[ind_rad][train])
                            perf_pre_summed[f,i,j,g,0]=mod_summed.score(feat_summed_class[ind_rad][train],labels[ind_rad][train])
                            perf_pre_summed[f,i,j,g,1]=mod_summed.score(feat_summed_class[ind_rad][test],labels[ind_rad][test])    
                            g=(g+1)
                            
        else: # If not splitting MLPs by curvature:
            perf_axis=2            
            for j in range(len(models_vec)):
                if verbose:
                    print('        Training NonLin-{} classifier....'.format(j+1))
                skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
                g=0
                for train,test in skf.split(feat_class,labels):
                    mod=MLPClassifier(models_vec[j],learning_rate_init=lr,alpha=mlp_reg,activation=activation)
                    mod.fit(feat_class[train],labels[train])
                    perf_pre[f,j,g,0]=mod.score(feat_class[train],labels[train])
                    perf_pre[f,j,g,1]=mod.score(feat_class[test],labels[test])
                    g=(g+1)
                    
                # If also computing performance summed across time bins:
                if sum_bins:
                    g=0
                    skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
                    for train,test in skf.split(feat_summed_class,labels):
                        mod_summed=MLPClassifier(models_vec[j],learning_rate_init=lr,alpha=mlp_reg,activation=activation)
                        mod_summed.fit(feat_summed_class[train],labels[train])
                        perf_pre_summed[f,j,g,0]=mod_summed.score(feat_summed_class[train],labels[train])
                        perf_pre_summed[f,j,g,1]=mod_summed.score(feat_summed_class[test],labels[test])
                        g=(g+1)
                    
                    
        # Log regress
        if split_by_curvature: # If splitting logistic regressions by curvature
            perf_lr_axis=2
            for i in range(len(rad_vec)):
                #print (i)
                ind_rad=np.where((curvature==rad_vec[i]))[0]
                skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
                g=0
                if verbose:
                    print('        Training linear classifier for curvature={}....'.format(rad_vec[i]))
                for train,test in skf.split(feat_class[ind_rad],labels[ind_rad]):
                    mod=LogisticRegression(C=1/lr_reg)
                    #mod=LinearSVC()
                    mod.fit(feat_class[ind_rad][train],labels[ind_rad][train])
                    lr_pre[f,i,g,0]=mod.score(feat_class[ind_rad][train],labels[ind_rad][train])
                    lr_pre[f,i,g,1]=mod.score(feat_class[ind_rad][test],labels[ind_rad][test])
                    g=(g+1)
                    
                # If also computing performance summed across time bins:
                if sum_bins:
                    g=0
                    skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
                    for train,test in skf.split(feat_summed_class[ind_rad],labels[ind_rad]):
                        mod_summed=LogisticRegression(C=1/lr_reg)
                        mod_summed.fit(feat_summed_class[ind_rad][train],labels[ind_rad][train])
                        lr_pre_summed[f,i,g,0]=mod_summed.score(feat_summed_class[ind_rad][train],labels[ind_rad][train])
                        lr_pre_summed[f,i,g,1]=mod_summed.score(feat_summed_class[ind_rad][test],labels[ind_rad][test])
                        g=(g+1)    
                    
                    
        else: # If not splitting logistic regressions by curvature:
            perf_lr_axis=1
            skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
            g=0 
            for train,test in skf.split(feat_class,labels):
                mod=LogisticRegression(C=1/lr_reg)
                #mod=LinearSVC()
                mod.fit(feat_class[train],labels[train])
                lr_pre[f,g,0]=mod.score(feat_class[train],labels[train])
                lr_pre[f,g,1]=mod.score(feat_class[test],labels[test])
                g=(g+1)
                
            # If also computing performance summed across time bins:
            if sum_bins:
                g=0 
                skf=StratifiedShuffleSplit(n_splits=n_cv, test_size=test_size)
                for train,test in skf.split(feat_summed_class,labels):
                    mod_summed=LogisticRegression(C=1/lr_reg)
                    mod_summed.fit(feat_summed_class[train],labels[train])
                    lr_pre_summed[f,g,0]=mod_summed.score(feat_summed_class[train],labels[train])
                    lr_pre_summed[f,g,1]=mod_summed.score(feat_summed_class[test],labels[test])
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
            
    
    perf=np.mean(perf_pre,axis=perf_axis)
    perf_m=np.mean(perf,axis=0)
    perf_sem=sem(perf,axis=0)
    print (perf_m)
    
    perf_lr=np.mean(lr_pre,axis=perf_lr_axis)
    lr_m=np.mean(perf_lr,axis=0)
    lr_sem=sem(perf_lr,axis=0)
    print (lr_m)
    
    if sum_bins:
        perf_summed=np.mean(perf_pre_summed,axis=perf_axis)
        perf_summed_m=np.mean(perf_summed,axis=0)
        perf_summed_sem=sem(perf_summed,axis=0)
        print (perf_m)
        
        perf_lr_summed=np.mean(lr_pre_summed,axis=perf_lr_axis)
        lr_summed_m=np.mean(perf_lr_summed,axis=0)
        lr_summed_sem=sem(perf_lr_summed,axis=0)
    else:
        perf_summed_m=None
        perf_summed_sem=None
    
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
    if split_by_curvature:
        #perf_m[:,0]=lr_m
        #perf_sem[:,0]=lr_sem
        lr_m = np.expand_dims(lr_m, axis=1)
        perf_m=np.concatenate((lr_m,perf_m), axis=1)
        
        lr_sem = np.expand_dims(lr_sem, axis=1)
        perf_sem=np.concatenate((lr_sem,perf_sem), axis=1)
        
        if sum_bins:
            lr_summed_m = np.expand_dims(lr_summed_m, axis=1)
            perf_summed_m=np.concatenate((lr_summed_m,perf_summed_m), axis=1)
            
            lr_summed_sem = np.expand_dims(lr_summed_sem, axis=1)
            perf_summed_sem=np.concatenate((lr_summed_sem,perf_summed_sem), axis=1)
            
        
    else:
        perf_m=np.concatenate((np.array([lr_m]),perf_m), axis=0)
        perf_sem=np.concatenate((np.array([lr_sem]),perf_sem), axis=0)
        
        if sum_bins:
            perf_summed_m=np.concatenate((np.array([lr_summed_m]),perf_summed_m), axis=0)
            perf_summed_sem=np.concatenate((np.array([lr_summed_sem]),perf_summed_sem), axis=0)

    
    # Perf Curvature
    if split_by_curvature:
        fig1 = plot_perf_v_curv(perf_m, perf_sem, rad_vec, lab_vec=lab_vec)
        if save_figs:
            perf_v_curv_fig_path = os.path.join(output_directory, 'performance_v_curvature.png')
            fig1.savefig(perf_v_curv_fig_path,dpi=500,bbox_inches='tight')
    
    ###################################
    # Fig 2
    fig2 = plot_model_performances(perf_m, perf_sem, perf_summed_m=perf_summed_m, perf_summed_sem=perf_summed_sem, plot_train=plot_train, split_by_curvature=split_by_curvature)
    
    # Save figures and metadata:
    if save_figs:
        model_rep_beh_path = os.path.join(output_directory,'model_reproduce_behavior_wiggles.png')
        fig2.savefig(model_rep_beh_path,dpi=500,bbox_inches='tight')
     
        # Save metadata:
        metadata = dict()
        metadata['sim_params'] = h
        metadata['sim_params']['spread']=spread # this needs to be overwritten since the actual numeric value is computed locally
        metadata['sim_params']['rad_vec']=list(rad_vec)
        metadata['sim_params']['steps_mov']=[int(x) for x in steps_mov] # has to be converted to int to play nice with JSON
        metadata['decoder_hyperparams'] = decoder_hparams
        metadata['task']=task
    
        metadata['outputs'] = []
        metadata['outputs'].append({'path':frame_wiggles_fig_path})
        if split_by_curvature:
            metadata['outputs'].append({'path':perf_v_curv_fig_path})
        metadata['outputs'].append({'path':model_rep_beh_path})
        
        datestr=now.strftime('%Y-%m-%d')
        timestr=now.strftime('%H:%M:%S')
        metadata['date']=datestr
        metadata['time']=timestr
    
        metadata_path=os.path.join(output_directory, 'whisker_task_sim_metadata.json')
        json.dump(metadata,open(metadata_path,'w'), indent=4)
    
    results=dict()
    results['perf_m']=perf_m
    results['perf_sem']=perf_sem
    if sum_bins:
        results['perf_summed_m']=perf_summed_m
        results['perf_summed_sem']=perf_summed_sem
    return results
        

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



def simulate_session(params, save_output=False, sum_bins=False, output_directory=None, verbose=False):
    """
    Simulate whisker contact data for a single simulated session.     

    Parameters
    ----------
    params : dict
        Dict defining simulation parameters. Should define same keys as hparams
        parameter to compare_stim_decoders(). 
    
    rad_vec : array-like
        Array of different stimulus curvatures (radii) to randomly sample from.
    
    verbose : bool, optional
        Whether to display status messages.

    Returns
    -------
    session: pandas.core.frame.DataFrame
        Dataframe of results for simulated session. Each row corresponds to a 
        single trial. Defines following columns:
            
            stimulus : bool
                Whether stimulus was convex or concave on corresponding trial
                (0: concave, 1: convex).
            
            curvature : float
                Stimulus curvature. 
            
            freq_sh : float
                Scalar describing stimulus texture spatial frequency.
            
            amp : float
                Stimulus texture amplitude.
    
            theta : float
                Stimulus angle.
            
            z1 : float
                Stimulus size. 
        
            time_mov : int
                Number of time steps stimulus moves before arriving to final
                position.
            
            speed: float
                Stimulus speed. 
        
            curvature : numpy.ndarray
                t-element array of stimulus curvature on each trial. 
        
            stimulus : numpy.ndarray
                t-element binary array of stimulus condition (concave vs convex) on
                each trial
                
            features : numpy.ndarray
                t-by-2w matrix of  simulated whisker contact, where t is the number of 
                time steps per trial, and w is the number of whiskers. The i,2(j-1)-th 
                element is a boolean stating whether whisker j contacted the stimulus 
                at the i-th time bin; the i,2(j-1)+1-th element states the 
                angle of whisker k in the j-th time bin if there was a contact. 

    """
    
    # Define parameters locally:
    
    # Simulation parameters:
    n_whisk=params['n_whisk']
    prob_poiss=params['prob_poiss']
    noise_w=params['noise_w']
    spread=params['spread']
    
    # Time and movement:            
    speed=params['speed']
    ini_phase_m=params['ini_phase_m']
    ini_phase_spr=params['ini_phase_spr']
    delay_time=params['delay_time']    
    freq_m=params['freq_m']
    freq_std=params['freq_std']
    t_total=params['t_total']
    dt=params['dt']
    n_trials_pre=params['n_trials_pre']
    init_position=params['init_position']

    # Shape:
    amp=params['amp']
    freq_sh=params['freq_sh']
    z1=params['z1']
    disp=params['disp']
    theta=params['theta']    
    steps_mov=params['steps_mov']
    rad_vec=params['rad_vec']
    if 'concavity' in params:
        concavity=params['concavity']
    else:
        concavity=np.array([0,1],dtype=np.int16)
        params['concavity']=concavity

    iterable_params=[concavity, amp, freq_sh, z1, disp, theta, steps_mov, rad_vec]
    num_vals_per_param=[np.size(x) for x in iterable_params]
    n_conditions=np.product(num_vals_per_param)

    # Reformat some parameters into lists if necessary: TODO: find a more elegant way of doing this
    if type(rad_vec)!=list:
        rad_vec=[rad_vec]
    if type(freq_sh)!=list:
        freq_sh=[freq_sh]
    if type(z1)!=list:
        z1=[z1]
    if type(theta)!=list:
        theta=[theta]
    
    # Define misc. arrays, etc.:
    l_vec=np.linspace(10,7,n_whisk)
    if spread=='auto':
        spread=1/n_whisk
    n_trials=n_trials_pre*n_conditions
    steps_mov=np.array(params['steps_mov'],dtype=np.int16)
    c_corr=[-1,1]
    t_vec=np.linspace(0,t_total,int(t_total/dt)) 
    
    # Initialize arrays of trial parameters:
    curvature=nan*np.zeros(n_trials)
    time_mov=nan*np.zeros(n_trials)
    stimulus=nan*np.zeros(n_trials)    
    freq_sh_vec=nan*np.zeros(n_trials)    
    z1_vec=nan*np.zeros(n_trials)    
    theta_vec=nan*np.zeros(n_trials)    
    
    ini_phase=np.random.vonmises(ini_phase_m,ini_phase_spr,n_trials)
    freq_whisk=np.random.normal(freq_m,freq_std,n_trials)
    
    features=np.zeros((n_trials,len(t_vec),2*n_whisk))
    #features=np.zeros((n_trials,len(t_vec),n_whisk))
    
    session = pd.DataFrame()

    
    for i in range(n_trials): # Loop across trials
    
        if verbose and np.remainder(i,100)==0:    
            print ('    Simulating trial {} out of {}...'.format(i, n_trials))
        
        # Define some parameters for current trial:
        ind_stim=np.random.choice(concavity,replace=False); stimulus[i]=ind_stim
        curvature[i]=np.random.choice(rad_vec,replace=False)
        time_mov[i]=np.random.choice(steps_mov,replace=False)
        freq_sh_vec[i]=np.random.choice(freq_sh,replace=False)
        z1_vec[i]=np.random.choice(z1,replace=False)
        theta_vec[i]=np.random.choice(theta,replace=False)
        #print (stimulus[i],curvature[i],time_mov[i])
        #print (ini_phase[i],freq_whisk[i])
        
        # Create shape t=0
        center0=center0_func(curvature[i],z1_vec[i])[ind_stim]
        center1=(center0+c_corr[ind_stim]*disp/curvature[i])
        center1=(center1-2*init_position*dt)
        center2=rotation_center(center1,c_corr[ind_stim]*theta_vec[i])
        
        l=np.sqrt((z1_vec[i]-10)**2+(z1_vec[i]-10)**2)
        x_len=abs(l*np.cos(-np.pi/4+c_corr[ind_stim]*theta_vec[i]))
        x_shape_pre=np.linspace(5+0.5*z1_vec[i]-0.5*x_len,5+0.5*z1_vec[i]+0.5*x_len,int((10-z1_vec[i])/0.01))
        x_shape=(x_shape_pre+c_corr[ind_stim]*disp/curvature[i]) 
        x_shape=(x_shape-2*init_position*dt) 
        y_shape=y_circ(x_shape,curvature[i],center2,amp,freq_sh_vec[i])[ind_stim]
        shape=np.stack((x_shape,y_shape),axis=1)
        shape=offset_shape(shape,init_position)

        # Simulate contacts for current trial:
        curr_trial_features = simulate_trial(ind_stim, curvature[i], x_shape, freq_sh_vec[i], 
        center2, n_whisk, ini_phase[i], freq_whisk[i], noise_w, amp, spread,
        time_mov[i], speed, dt, delay_time, len(t_vec), prob_poiss)
        features[i,:,:] = curr_trial_features
        
        
        # Define full dict for current trial:
        
        # Stimulus shape parameters:
        trial_dict = dict()
        trial_dict['stimulus']=ind_stim
        trial_dict['curvature']=curvature[i]
        trial_dict['freq_sh']=freq_sh_vec[i]
        trial_dict['amp']=amp
        trial_dict['theta']=theta_vec[i]
        trial_dict['z1']=z1_vec[i]
        
        # Stimulus movement parameters:
        trial_dict['time_mov']=time_mov[i]
        trial_dict['speed']=speed        

        # Whisking parameters:            
        trial_dict['ini_phase']=ini_phase[i]
        trial_dict['freq_whisk']=freq_whisk[i]            

        # Contact/angle data: 
        trial_dict['features']=curr_trial_features
        if sum_bins:
            trial_dict['features_bins_summed'] = np.sum(curr_trial_features,0)
        
        session = session.append(trial_dict, ignore_index=True)
        
    # Save session if requested:
    if save_output: 
        
        # Make current folder default:
        if output_directory==None:
            output_directory=os.getcwd()
            
        # Create output directory if necessary:
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Save sessions dataframe as pickle:
        output_path=os.path.join(output_directory, 'simulated_session.pickle')
        with open(output_path, 'wb') as p:
            pkl.dump(session, p)
            
        # Write metadata if analysis_metadata module successfully imported:
        if 'analysis_metadata' in sys.modules:
            M=Metadata()
            M.parameters=params
            M.add_output(output_path)
            metadata_path = os.path.join(output_directory, 'simulation_metadata.json')
            write_metadata(M, metadata_path)
        
    #return features, curvature, stimulus
    return session



def simulate_trial(concavity, curvature, x_shape, freq_sh, center, n_whisk, 
ini_phase, freq_whisk, noise_w, amp, spread, mov_steps, speed, dt, delay_time, 
n_bins, prob_poiss):
    """
    Simulate whisker contacts for a single trial. 

    Parameters
    ----------
    concavity : bool
        Whether stimulus is convex or concave (0: concave, 1: convex).
        
    curvature : int | float
        Scalar describing stimulus curvature.
        
    x_shape : numpy.ndarray
        Array of stimulus x-coordinates.
        
    freq_sh : int | float
        Scalar describing stimulus texture spatial frequency.
        
    center : numpy.ndarray
        2-element array specifying center of circle defining stimulus.
        
    n_whisk : int
        Number fo whiskers.

    ini_phase : int | float
        Trial-initial whisker phase.

    freq_whisk : int | float
        Whisking frequency.

    noise_w : int | float
        Whisker contact noise level.

    amp : int | float
        Stimulus texture amplitude.

    spread : int | float
        Spacing between whiskers.

    mov_steps : int | float
        Number of time steps over which stimulus moves to final position.

    speed : int | float
        Stimulus speed.

    dt : float
        Time step size.

    delay_time : float
        Amount of time to wait at beginning of each trial before moving
        stimulus, in time steps.

    n_bins : int
        Number of time bins per trial.

    prob_poiss : float
        Parameter of Poisson distribution controlling whether contact given 
        whisker intersects shape.


    Returns
    -------
    features : numpy.ndarray
        t-by-2w matrix of  simulated whisker contact, where t is the number of 
        time steps per trial, and w is the number of whiskers. The i,2(j-1)-th 
        element is a boolean stating whether whisker j contacted the stimulus 
        at the i-th time bin; the i,2(j-1)+1-th element states the 
        angle of whisker k in the j-th time bin if there was a contact. 

    """
    
    
    l_vec=np.linspace(10,7,n_whisk)
    features=np.zeros((n_bins, 2*n_whisk))
    
    # Loop across time steps
    for t in np.arange(n_bins): 

        angle_t=np.sin(freq_whisk*t+ini_phase)
        
        if  (t>=delay_time) and t<(mov_steps+delay_time):
            center=(center-speed*dt)
            x_shape=(x_shape-speed*dt)
            y_shape=y_circ(x_shape,curvature,center,amp,freq_sh)[concavity]
            shape=np.stack((x_shape,y_shape),axis=1)
            
        # Loop across whiskers
        for w in range(n_whisk):
            nw=np.random.normal(0,noise_w,2)
            ang_inst=(angle_t+w*spread)
            wt_pre=np.array([l_vec[w]*np.cos(ang_inst),l_vec[w]*np.sin(ang_inst)])
            wt=(wt_pre+nw)
            prob,c1,c2=func_in_out_new(shape,wt,center,curvature, concavity,prob_poiss,amp,freq_sh)
            ct_bin=int(np.random.uniform(0,1)<prob)
            features[t,2*w]=ct_bin
            #features[i,ii,iii]=ct_bin
            if ct_bin==1:
                features[t,2*w+1]=ang_inst

    return features



def session2labels(session, task, label_all_trials=False):
    """
    Generate a vector of condition labels from a task definition and a table of 
    simulated trial data. 

    Parameters
    ----------
    session : pandas.core.frame.DataFrame
        Dataframe of simulated trials. Each row corresponds to a simulated 
        trial. Same format as output of simulate_session().
        
    task : array-like
        List of k dicts, where k is the number of stimulus conditions (i.e. output 
        labels) in the task. The keys defined in each dict should be a subset
        of the trial parameters used to generate trials, i.e., a subset of the 
        columns of the input `session` dataframe. For example, 
        
        task = [
            {'stimulus':0},
            {'stimulus':1}
            ]
        
        could be used to define a binary convex vs concave classification task.
        Alternatively, 
        
        task = [
            {'curvature':6},
            {'curvature':12},
            {'curvature':24}
            ]
        
        could be used to define a 3-way curvature classification task, or 

        task = [
            {'stimulus':0, 'curvature':6},
            {'stimulus':0,'curvature':12},
            {'stimulus':1, 'curvature':6},
            {'stimulus':1,'curvature':12},
            ]
        
        could be used to define a 4-way convexity X curvature task. 
        
    label_all_trials : boolean, optional
        Whether to assign a numeric label of -1 to trials that don't match any 
        of the conditions defined in `task` input. Otherwise trials that don't
        match any condition defined in `task` will receive label of nan. Set to
        True if dealing with downstream function that don't accept nan labels.

    Returns
    -------
    labels : numpy.ndarray
        t-element vector of condition labels, where t is the number of trials
        (rows) in `session` input dataframe. 

    """
    n_trials = session.shape[0]
    n_conditions = len(task)
    
    # Initialize label vector:
    labels = np.empty(n_trials)
    labels[:] = None
    
    # Iterate over conditions (i.e. output labels) for current task: 
    for cx, condition in enumerate(task):
        
        condition_filter = np.full(n_trials,True) # initialize boolean filter
        
        # Iterate over trial parameters defining current condition:
        for param in condition:
            param_filter = session[param]==condition[param]
            param_filter = np.array(param_filter)
            condition_filter = condition_filter & param_filter
        
        # Make sure we're not overwriting any previous condition labels; 
        # otherwise, it means some trials match more than one condition 
        # definition, in which case raise an error (maybe can add support for 
        # multiple labels per trial in future version, but keep it 1 label per
        # trial for now):
        if np.any(~np.isnan(labels[condition_filter])):
            raise AssertionError('At least one trial assigned more than one condition label; please check that condition definitions are mutually exclusive.'.format())
        
        # Assign label cx to trials matching parameters defined in `condition` dict:
        labels[condition_filter] = cx 
    
    # If label_all_trials is True, then assign label of -1 to all trials that 
    # don't match any other conditions:
    if label_all_trials:
        other = [l not in np.arange(n_conditions) for l in labels]
        labels[other] = -1
    
    return labels
    

    
def session2feature_array(session, field='features'):
    """
    Extract simulated whisker contact and angle data from session dataframe.

    Parameters
    ----------
    session : pandas.core.frame.DataFrame
        Dataframe of simulated session data and parameters. Should be same 
        format as output of simulate_session() function.
    
    field: 'features' | 'features_bins_summed'
        Data to convert to array. Can be either raw features include full 
        spatiotemporal patter of contacts/angles ('features') or angles summed
        across time ('features_bins_summed').

    Returns
    -------
    F : numpy.ndarray
        t-by-g matrix, where t is the number of trials and g is the total 
        number of features per trial. g itself equals b*f, where b is the 
        number of time bins per trial, and f is the number of features per time
        bin (whisker contacts, whisker angles, etc).

    """
    F = np.array([np.reshape(x,-1) for x in session[field]])
    return F        
    
    

def binarize_contacts(features, operation='median'):
    """
    Binarize whisker contacts.

    Parameters
    ----------
    features : array-like
        t-by-g matrix, where t is the number of trials and g is the total 
        number of features per trial (same format as output of
        session2feature_array()). 
        
    operation : str, optional
        Operation to use for binarizing integer number of contacts. Possible 
        operations are as follows:
            
                median: threshold contacts based on the median number of contacts 
                    across trials for each whisker. If the number of contacts 
                    for a given whisker on a given trial is above the median 
                    for that whisker, then count as 1; otherwise, count as 0.

    Returns
    -------
    binarized_features : numpy.ndarray
        Same shape as input, except all entries are binary.

    """
    if operation=='median':
        features_noise=(features+np.random.normal(0,0.001,np.shape(features)))
        meds=np.median(features_noise, 0)
        binarized_features=(features_noise>meds)
    
    return binarized_features



def load_task_def(path):
    """
    Load task definition from JSON file. 

    Parameters
    ----------
    path : str
        Path to JSON file encoding task definition. Should be formatted as a 
        JSON object with a single key called "task". The value of this "task"
        field should itself be a list of JSON objects, each one corresponding
        to a single output class for the task. Each of these inner JSON objects
        defines a number of keys corresponding to trial parameters that define
        the corresponding output class. 

    Returns
    -------
    task : list
        List of dicts, where each dict defines one output class for the task.

    """
    contents = json.load(open(path, 'r'))
    task = contents['task']
    return task



def plot_perf_v_curv(perf_m, perf_sem, rad_vec, lab_vec=None):
    """
    Plot decoder performance vs stimulus curvature.

    Parameters
    ----------
    perf_m : numpy.ndarray
        c-by-m-by-2 array, where c is the number of curvatures and m is the 
        number of models tested. The i,j,0-th element is the mean performance 
        of the j-th model on the i-th curvature on training data; the i,j,1-th 
        element is the mean performance of the j-th model on the i-th curvature
        on held-out test data.
        
    perf_sem : numpy.ndarray
        Same as perf_m, but for SEM instead of mean.
        
    rad_vec : numpy.ndarray
        Array of curvatures tested. Number of elements must equal c (see 
        description of perf_m).
        
    lab_vec : TYPE, optional
        Labels for different models tested. Number of elements must equal m
        (see description of perf_m).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of decoding performance vs stimulus curvature. Plot includes 
        separate curve for each model. 

    """

    num_models=perf_m.shape[1]
    if lab_vec==None:
        lab_vec = ['model{}'.format(x) for x in range(num_models)]
    
    fig=plt.figure(figsize=(2,2))
    ax=fig.add_subplot(111)

    perf_sem[np.isnan(perf_sem)]=0 # convert any nan to 0 for plotting purposes

    for j in range(num_models):
        if j==0:
            plt.errorbar(rad_vec,perf_m[:,j,1],yerr=perf_sem[:,j,1],color='orange',label=lab_vec[j])
        else:
            plt.errorbar(rad_vec,perf_m[:,j,1],yerr=perf_sem[:,j,1],color='green',alpha=(j+1)/num_models,label=lab_vec[j])
    plt.plot(rad_vec,0.5*np.ones(len(rad_vec)),color='black',linestyle='--')
    plt.xscale('log')
    plt.xlabel('Curvature (Rad)')
    plt.ylabel('Performance')
    plt.legend(loc='best')
    plt.ylim([0.4,1])
    plt.show()

    return fig    



def plot_model_performances(perf_m, perf_sem, perf_summed_m=None, perf_summed_sem=None, plot_train=False, split_by_curvature=True):
    """
    Plot bar graphs of decoder model performance.

    Parameters
    ----------
    perf_m : numpy.ndarray
        Same as for plot_perf_v_curv.
        
    perf_sem : numpy.ndarray
        Same as for plot_perf_v_curv.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Bar graph of decoder performance for different models.

    """
    
    if split_by_curvature:
        n_models_idx=1
    else:
        n_models_idx=0
    num_models=perf_m.shape[n_models_idx]  
    min_alph=0.4
    max_alph=1.0
    alph_step = (max_alph-min_alph)/(num_models-1)
    alpha_vec=np.arange(min_alph, max_alph+alph_step, alph_step)
    width=0.15

    fig=plt.figure(figsize=(2,2))
    ax=fig.add_subplot(111)
    #functions_miscellaneous.adjust_spines(ax,['left','bottom'])
    
    #plt.xticks(width*np.arange(len(models_vec))-1.5*width,model_labels,rotation='vertical')
    for j in range(num_models):
        if split_by_curvature:
            ax.bar(j*width-1.5*width,perf_m[0,j,1],yerr=perf_sem[0,j,1],color='green',width=width,alpha=alpha_vec[j])
            if plot_train:
                ax.scatter(j*width-1.5*width,perf_m[0,j,0],color='green',alpha=alpha_vec[j])
        else:
            ax.bar(j*width-1.5*width,perf_m[j,1],yerr=perf_sem[j,1],color='green',width=width,alpha=alpha_vec[j])
            if plot_train:
                ax.scatter(j*width-1.5*width,perf_m[j,0],color='green',alpha=alpha_vec[j])
        #ax.scatter(j*width-1.5*width+p+np.random.normal(0,std_n,3),perf[:,p,j,1],color='black',alpha=alpha_vec[j],s=4)
    #ax.bar(-1.5*width,lr_m[0,1],yerr=lr_sem[0,1],color='green',width=width,alpha=alpha_vec[0])
    
    # If plotting performance for contacts summed across bins:
    if perf_summed_m is not None and perf_summed_sem is not None: 
        xright=(3.5+num_models)*width
        for j in range(num_models):
            if split_by_curvature:
                ax.bar((j+num_models)*width-1.5*width,perf_summed_m[0,j,1],yerr=perf_summed_sem[0,j,1],color='orange',width=width,alpha=alpha_vec[j])
                if plot_train:
                    ax.scatter((j+num_models)*width-1.5*width,perf_summed_m[0,j,0],color='orange',alpha=alpha_vec[j])    
            else:
                ax.bar((j+num_models)*width-1.5*width,perf_summed_m[j,1],yerr=perf_summed_sem[j,1],color='orange',width=width,alpha=alpha_vec[j])
                if plot_train:
                    ax.scatter((j+num_models)*width-1.5*width,perf_summed_m[j,0],color='orange',alpha=alpha_vec[j])
    else:
        xright=3.5*width
    ax.plot([-3.5*width,xright],0.5*np.ones(2),color='black',linestyle='--')
    ax.set_ylim([0.4,1.0])
    #ax.set_xlim([-3.5*width,3.5*width])
    ax.set_ylabel('Decoding Performance')
    
    return fig
    
    
    
def plot_summed_contacts(session, task, colors=None, save_output=False, output_directory=None):
    
    # Extract feature matrix:
    F=session2feature_array(session,field='features_bins_summed')
    F=F[:,np.arange(0,F.shape[1],2)] # keep only contacts (even cols.); exclude angle (odd cols)
    
    # Generate trial labels:
    labels=session2labels(session,task)
    
    # Plot data:
    fig,ax=plot_2d_inpt(F,labels,colors=colors)
        
    # Make X and Y lims equal:
    xl=ax.get_xlim()
    yl=ax.get_ylim()
    maxmax=max(xl[1],yl[1])
    minmin=max(xl[0],yl[0])
    ax.set_xlim(minmin, maxmax)
    ax.set_ylim(minmin, maxmax)
    
    # Add axis labels:
    plt.xlabel('whisker 1 summed contacts')
    plt.ylabel('whisker 2 summed contacts')
        
    # Save output if requested:
    if save_output:

        if output_directory == None:
            output_directory = os.getcwd()

        # If requested output directory does not exist, create it:        
        if not os.path.exists(output_directory):
            pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        fig_path = os.path.join(output_directory,'whisker_contacts.png')
        fig.savefig(fig_path,dpi=500,bbox_inches='tight')
    
        # Write metadata if module available:
        if 'analysis_metadata' in sys.modules:
            M=Metadata()
            M.add_output(fig_path)
            metadata_path = os.path.join(output_directory, 'plot_contacts_metadata.json')
            write_metadata(M, metadata_path)
            
            
    
def plot_2d_inpt(dat, labels, colors=None):
    # TODO: raise warning if dat is more than 2 columns
    # TODO: verify that len(labels)=dat.shape[0]
    # TODO: verify that len(colors)=dat.shape[0] if colors is not None
    
    # Initialize figure:
    fig=plt.figure(figsize=(3,3))
    ax=fig.add_subplot(111)
    
    # Iterate over conditions:
    unique_labels=np.unique(labels)
    for bx, b in enumerate(unique_labels):
        if colors is not None:
            curr_color=colors[bx]
        else:
            curr_color=None
        curr_dat=dat[labels==b]
        ax.scatter(curr_dat[:,0], curr_dat[:,1],c=curr_color,alpha=0.1)
        
    return fig, ax
        


def load_sim_params(hparams):
    """
    Load simulated whisker task parameters and decoder hyperparameters.    

    Parameters
    ----------
    hparams : str | dict
        Simulation parameters and decoder hyperparameters. If str, should be 
        path to a JSON file encoding relevant variables; if dict, should define 
        one key for each parameter/hyperparameter. See example_hparams.json 
        file in this repo for example. TODO: add documentation for specific 
        params/hyperparams. 

    Returns
    -------
    h : dict
        Dict of simulated whisker task parameters and decoder hyperparameters.

    """

    # Load/define hyperparameters:
    # If no hyperparameters provided, use defaults:    
    if hparams==None: 
        h=generate_default_sim_params() 
    # If hparams is a dict, just use it directly:
    elif type(hparams)==dict:
        h=hparams
    # If hparams is a path to a JSON file:
    elif type(hparams)==str: 
        # Make sure the hyperparameter file actually exists:
        if not os.path.exists(hparams):
            raise FileNotFoundError('Hyperparameter file {} not found; please make sure that file exists and path is specified correctly.'.format(hparams))
        else:
            h = json.load(open(hparams,'r')) # TODO: add function validating all necessary hyperparameters are defined
    
    return h



def load_mlp_hparams(hparams):
    """
    Load MLP hyperparameters.    

    Parameters
    ----------
    hparams : str | dict
        MLP hyperparameters. If str, should be path to a JSON file encoding 
        relevant variables; if dict, should define one key for each 
        hyperparameter. See example_mlp_hparams.json file in this repo for 
        example. TODO: add documentation for specific params/hyperparams. 

    Returns
    -------
    h : dict
        Dict of MLP hyperparameters.

    """

    # Load/define hyperparameters:
    # If no hyperparameters provided, use defaults:    
    if hparams==None: 
        h=generate_default_mlp_hparams() 
    # If hparams is a dict, just use it directly:
    elif type(hparams)==dict:
        h=hparams
    # If hparams is a path to a JSON file:
    elif type(hparams)==str: 
        # Make sure the hyperparameter file actually exists:
        if not os.path.exists(hparams):
            raise FileNotFoundError('Hyperparameter file {} not found; please make sure that file exists and path is specified correctly.'.format(hparams))
        else:
            h = json.load(open(hparams,'r')) # TODO: add function validating all necessary hyperparameters are defined
    
    return h



def define_model_labels(models_vec):
    """
    Define labels for a set of models.

    Parameters
    ----------
    models_vec : list
        List of lists and scalars. Each element corresponds to a model. An 
        empty list means a linear model, a scalar means an MLP with one hidden
        layer, and for all n>1, an n-element list means an MLP with n hidden
        layers.

    Returns
    -------
    labels_vec : list
        List of strings, each one a label for an input model.
    """
    
    labels_vec = []
    for m in models_vec:
        if np.isscalar(m):
            labels_vec.append('Nonlin1')
        elif type(m)==list:
            if len(m)==0:
                labels_vec.append('Lin')
            else:
                labels_vec.append('Nonlin{}'.format(len(m)))            
            
    return labels_vec



def load_simulation(session_in):
    
    # If session_in is str, assume path to pickle containing dataframe of simulated session:
    if type(session_in)==str:
        session_out=pkl.load(open(session_in, 'rb'))
    
    # if session_in is dataframe, just return it:
    elif type(session_in)==pd.core.frame.DataFrame:
        session_out=session_in
    
    # otherwise raise error:
    else:
        raise TypeError('session_in not of recognized type; please ensure session_in is either a pandas dataframe or a path to a pickled pandas dataframe.')
    
    return session_out


    
class manager(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-s', '--save', action='store_true')
        parser.add_argument('-v', '--verbose', action='store_true')
        parser.add_argument('-w', '--sim_params', type=str, action='store', default=None)
        parser.add_argument('-m', '--mlp_hparams', type=str, action='store', default=None)
        parser.add_argument('-o', '--output_directory', type=str, action='store', default=None)
        args = parser.parse_args(sys.argv[2:])
        
        sim_params=args.sim_params
        mlp_hparams=args.mlp_hparams
        save=args.save
        output_directory=args.output_directory
        #print('output_directory={}'.format(output_directory))
        #print('output_directory={}'.format(output_directory))
        verbose=args.verbose
    
        compare_stim_decoders(sim_params=sim_params, mlp_hparams=mlp_hparams, save_figs=save, output_directory=output_directory, verbose=verbose)


if __name__ == '__main__':
    manager()
