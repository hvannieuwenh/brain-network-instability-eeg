import os
import numpy as np
import glob
import re
import math
import sklearn
import covar
import csv
import mat73
import scipy as sp
from scipy import io
from sklearn.covariance import LedoitWolf
from os.path import exists

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

# set paths, import lead field
main_dir = '/shared/home/jolien/python_stuff/'
leadfield = sp.io.loadmat(main_dir + 'mean_leadfield_svd.mat')

windowsizes = [0.75,1.25,2.5,5,7.5,10,15,20,25,30]

for coupling in ['035_dense_sample']:
    coup_dir='/shared/datasets/private/instability_outputs_LB/coupling_{}/'.format(coupling)
    if os.path.isdir(coup_dir) == False:
        os.mkdir(coup_dir)
    for param_var in ['eca','ek','ena','gca','gk','gna','aee','aei']:
        lb_data_dir='/shared/datasets/private/LB_model_outputs/jolien_GFS/coupling_variations/coupling_{}/{}/'.format(coupling,param_var)
        output_dir = '/shared/datasets/private/instability_outputs_LB/coupling_{}/{}/'.format(coupling,param_var)
        if os.path.isdir(output_dir) == False:
            os.mkdir(output_dir)

        for mat_file in glob.glob(lb_data_dir + '*.mat'):
            print(mat_file)
            sampling_freq = 1000
            lb_mat = mat73.loadmat(mat_file)
            eeg_signals = np.dot(leadfield['leadfield'],lb_mat['excitatory_signals'])
            #eeg_signals = np.dot(leadfield['leadfield'],lb_mat['signals'])
            time = np.arange(0,len(eeg_signals[0]))/sampling_freq
            for band in ['allfreq']:
                if os.path.isdir(output_dir + 'instab_alltau_' + band) == False:
                    os.mkdir(output_dir + 'instab_alltau_' + band)
                if band == 'delta':
                    l_freq = 1
                    h_freq = 3.5
                if band == 'theta':
                    l_freq = 4
                    h_freq = 7.5
                if band == 'alpha':
                    l_freq = 8
                    h_freq = 13
                if band == 'beta':
                    l_freq = 14
                    h_freq = 30
                if band == 'lowgamma':
                    l_freq = 30
                    h_freq = 40 
                if band == 'broadband':
                    l_freq = 1
                    h_freq = 40 
                for windowsizesec in windowsizes:
                    if os.path.isdir(output_dir + 'instab_alltau_' + band + '/instab_alltau_' \
                                     + band + '_winsize' + str(windowsizesec)) == False:
                        os.mkdir(output_dir + 'instab_alltau_' + band + '/instab_alltau_' \
                                 + band + '_winsize' + str(windowsizesec))
                        
                    print('New run')
                    print('Sampling freq = ',sampling_freq, ' Hz')

                    ###  Instability time! 

                    # input some information about the data which allows for calculation of window size
                    TR = eeg_raw.times[1] - eeg_raw.times[0]            # time of repetition of data (aka sampling period) in units of seconds (s)
                    total_scan_time = (len(time)*TR)/60
                    numtimepts = len(ROI_amp[0])   #stores number of time points sampled in time series
                    durationTS = numtimepts*TR       #stores length of time series in units of seconds (s)
                    maxtau = math.floor(durationTS/windowsizesec) #stores maximum value of "tau" that can be examined using given information/data, rounds down
                    windowsizetimepts = round(windowsizesec/TR)      #calculates desired window size in units of # of timepoints, rounded to nearest whole number since # of timepoints should be integer 
                    w = windowsizetimepts
                    print('TR = ',TR,' s')
                    print('Total scan time = ',total_scan_time,' min')
                    print('Duration of time series = ',durationTS, ' s')
                    print('Window size in seconds = ', windowsizesec, ' s')
                    print('Window size in time points = ',w)
                    print('Max tau =',maxtau)
                    print('Number of time points in time series = ',numtimepts)


                    # this splices the ROI time series into windows of size "w" and creates a 3-d array containing 
                    #i: tau'th window, j: ROI, k: timepts
                    temp = []
                    temp2 = []
                    #print('length of eeg_signals=',len(eeg_signals))
                    for x in range(0,maxtau):
                        for y in range(0,len(eeg_signals)):
                            chunk = eeg_signals[y,(w*x):(w*(x+1))].tolist()
                            temp.append(chunk)
                        temp2.append(temp)
                        temp = []
                    eeg_signals_windowed = np.array(temp2)

                    temp3 = []

                    #compute matrices of pearson correlations between ROIs for each individual time window, stored in one 3d array 
                    #(tau_corr) with each i'th matrix being one time windows' correlation

                    for x in range(0,maxtau):
                        real_cov = np.cov(eeg_signals_windowed[x])
                        np.random.seed(0)
                        X = np.random.multivariate_normal(mean=np.zeros(len(real_cov)),cov=real_cov,size=50)
                        LW_cov = LedoitWolf().fit(X)
                        corr = correlation_from_covariance(LW_cov.covariance_)
                        temp2 = corr.tolist()
                        temp3.append(temp2)
                    tau_corr_bot = np.array(temp3)


                    #finally, calculate instabilities.
                    temp=[]
                    all_instab=[]
                    for tau in range(1,maxtau):
                        for leftwin in range(0,maxtau):
                            for rightwin in range(0,maxtau):
                                if rightwin-leftwin == tau:
                                    diff = np.subtract(tau_corr_bot[rightwin],tau_corr_bot[leftwin])
                                    instab = (np.linalg.norm(diff))/(math.sqrt((diff.size) - len(eeg_signals)))
                                    temp.append(instab)
                        all_instab.append(temp)
                        temp = []

                    #store and print average of one run's instability for smallest time window (tau=1)
                    tau1_instab_all = all_instab[0]
                    tau1_instab_avg = np.average(all_instab[0])
                    tau1_instab_sd = np.std(all_instab[0])                 # should this be standard deviation or SEM?
                    print('Tau=1 average instability = \n',tau1_instab_avg)
                    print('Tau=1 SD instability = \n', tau1_instab_sd)

                    out_run = re.sub('.mat','',re.sub(lb_data_dir,'',mat_file))
                    output_file = output_dir + 'instab_alltau_' + band + '/instab_alltau_' \
                    + band + '_winsize' + str(windowsizesec) + '/' + out_run + '_instability.csv'

                    with open(output_file, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(all_instab)