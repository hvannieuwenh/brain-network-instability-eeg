import os
import numpy as np
import mne
import glob
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
#import pypfopt
import sklearn
#import covar
import scipy as sp
import seaborn as sns
import csv

from mpl_toolkits import mplot3d
from scipy import stats, signal
#from pypfopt.risk_models import risk_matrix
from sklearn.covariance import LedoitWolf
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,corrmap)
from numpy import savetxt
from numpy.polynomial.polynomial import polyfit
from datetime import datetime
from os.path import exists

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

main_dir = '/shared/home/jolien/metabolic_eeg_instability/'
which_prep = '1-40Hz_'+eyes+'asr40'
data_dir = '/shared/datasets/private/eeg_metabolic/prep_data_'+which_prep+'/'

subjects=['001','002']
sessions=['glu','bhb']
windowsizes=[30,25,20,15,10,7.5,5,2.5,1.25,0.75]
eyes='EO'
runs=['1','2']
task='rest'
bands=['allfreq','delta','theta','alpha','beta','lowgamma']

all_subject_list,all_avg_instability,all_sd_instability = [],[],[]


# make needed output directories 
if os.path.isdir(main_dir + 'instab_'+ which_prep) == False:
    os.mkdir(main_dir + 'instab_'+ which_prep)
for band in bands:
    if os.path.isdir(main_dir + 'instab_'+ which_prep + '/instab_alltau_' + band) == False:
        os.mkdir(main_dir + 'instab_'+ which_prep + '/instab_alltau_' + band)
    for windowsizesec in windowsizes:
        if os.path.isdir(main_dir + 'instab_'+ which_prep + '/instab_alltau_' + band + '/instab_alltau_' \
                         + band + '_winsize' + str(windowsizesec)) == False:
            os.mkdir(main_dir + 'instab_'+ which_prep + '/instab_alltau_' + band + '/instab_alltau_' \
                     + band + '_winsize' + str(windowsizesec))

for windowsizesec in windowsizes:
    for band in bands:
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
        if band == 'allfreq':
            h_freq = 40
        for subj in subjects:
            for session in sessions:
                for runnum in runs:
                    check_path=main_dir + 'instab_'+ which_prep + '/instab_alltau_' + band + '/instab_alltau_' \
                                         + band + '_winsize' + str(windowsizesec) +'/'+'sub-'\
                                         +subj+'_ses-'+session+'_task-'+task+eyes+'_run-'\
                                         +runnum+'.csv'
                    print('Now trying:')
                    print(check_path)
                    
                    if exists(check_path) == True:
                        print('*** Subject already done!')
                        continue
                        
                    else:
                        print('New subject')
                        now=datetime.now()
                        print('*** TIME =',now.strftime("%H:%M:%S"))
                        
                        try:
                            if task=='rest':
                                data_path = data_dir + 'sub-' + subj + '_ses-' + session \
                                + '_task-' + task + eyes+'_run-' + runnum + '.set'
                                output_file = main_dir + 'instab_'+ which_prep + '/instab_alltau_' + band + '/instab_alltau_' + band + '_winsize' + str(windowsizesec) + '/' + 'sub-' + subj + \
                                '_ses-' + session + '_task-' + task + eyes + '_run-' + runnum + '.csv'

                            eeg_raw = mne.io.read_raw_eeglab(data_path,eog=['EXG1'],preload=True)
                            channels = eeg_raw.ch_names
                            sampling_freq = eeg_raw.info['sfreq']
                            if band == 'allfreq':
                                print('Applying {} LP filter'.format(band))
                                eeg_raw = eeg_raw.filter(None,h_freq)
                            else:
                                print('Applying {} BP filter'.format(band))
                                eeg_raw = eeg_raw.filter(l_freq,h_freq)
                            all_channel_amp, all_times = eeg_raw.get_data(return_times=True,picks='eeg')
                            print('Sampling freq = ',sampling_freq, ' Hz')

                            ###  Instability time! 

                            ROI_amp = [];
                            for i in range(0,len(channels)):
                                #print(channels[i])
                                # beginning and end of desired sample in timepts
                                beginning = 0
                                end = int((np.shape(all_channel_amp))[1])
                                selection = eeg_raw[i,beginning:end]
                                time = selection[1]
                                amp = (selection[0][0].T).tolist()
                                #plt.plot(time,amp)
                                #plt.show()
                                ROI_amp.append(amp)

                            ROI_amp = np.array(ROI_amp)

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
                            print('maxtau=',maxtau-1)
                            print('Number of time points in time series = ',numtimepts)


                            # this splices the ROI time series into windows of size "w" and creates a 3-d array containing 
                            #i: tau'th window, j: ROI, k: timepts
                            temp = []
                            temp2 = []
                            #print('length of ROI_amp=',len(ROI_amp))
                            for x in range(0,maxtau):
                                for y in range(0,len(ROI_amp)):
                                    chunk = ROI_amp[y,(w*x):(w*(x+1))].tolist()
                                    temp.append(chunk)
                                temp2.append(temp)
                                temp = []
                            ROI_amp_windowed = np.array(temp2)

                            temp3 = []

                            #compute matrices of pearson correlations between ROIs for each individual time window, stored in one 3d array 
                            #(tau_corr) with each i'th matrix being one time windows' correlation

                            for x in range(0,maxtau):
                                real_cov = np.cov(ROI_amp_windowed[x])
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
                                            instab = (np.linalg.norm(diff))/(math.sqrt((diff.size) - len(channels)))
                                            temp.append(instab)
                                all_instab.append(temp)
                                temp = []

                            #store and print average of one subjects' instability for smallest time window (tau=1)
                            tau1_instab_all = all_instab[0]
                            tau1_instab_avg = np.average(all_instab[0])
                            tau1_instab_sd = np.std(all_instab[0])                 
                            print('Tau=1 average instability = \n',tau1_instab_avg)
                            print('Tau=1 SD instability = \n', tau1_instab_sd)

                            all_subject_list.append(subj)
                            all_avg_instability.append(tau1_instab_avg)
                            all_sd_instability.append(tau1_instab_sd)

                            with open(output_file, 'w', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(all_instab)

                            print('Successfully saved {}'.format(output_file))

                        except Exception as e: 
                            print('***ERROR')
                            print(e)
                            np.savetxt(output_file + 'FAILED',[0])

                
now=datetime.now()
print('*** FINISH time = ',now.strftime("%H:%M:%S"))