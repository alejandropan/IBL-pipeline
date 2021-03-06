# @alejandropan 2019

#Import some general packages

import time, re, datetime, os, glob
from datetime import timedelta
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from IPython import embed as shell
from scipy import stats

## CONNECT TO datajoint

import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'

import datajoint as dj
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis
from load_mouse_data_datajoint import *  # this has all plotting functions


#Collect all alyx data
allsubjects = pd.DataFrame.from_dict((subject.Subject() * subject.SubjectLab & 'sex!="U"' & 'subject_birth_date>"2018-10-15"' ).fetch(as_dict=True, order_by=[ 'subject_nickname']))



allsubjects = pd.DataFrame.from_dict(
    ((subject.Subject - subject.Death) * subject.SubjectLab & 'sex!="U"' &
     action.Weighing & action.WaterAdministration).fetch(
as_dict=True, order_by=['lab_name', 'subject_nickname']))

if allsubjects.empty:
    raise ValueError('DataJoint seems to be down, please try again later')
#Drop double entries
allsubjects =  allsubjects.drop_duplicates('subject_nickname')
#Drop transgenics
allsubjects.loc[(allsubjects['subject_line'] == 'C57BL/6J') |(allsubjects['subject_line'] == None)]

#Add learning rate columns
allsubjects['training_status'] =np.nan
allsubjects['days_to_trained'] = np.nan
allsubjects['trials_to_trained'] = np.nan

#Add bias (level2) columns
allsubjects['average_bias08'] =np.nan
allsubjects['average_bias02'] =np.nan
allsubjects['average_threshold'] =np.nan
allsubjects['average_lapse_high'] =np.nan
allsubjects['average_lapse_low'] =np.nan



users  =  allsubjects['lab_name'].unique()

for labname in users:
    for mouse in allsubjects['subject_nickname']:
            try:
                # TRIAL COUNTS AND SESSION DURATION
                behav = get_behavior(mouse, labname)
                # check whether the subject is trained based the the lastest session
                subj = subject.Subject & 'subject_nickname="{}"'.format(mouse)
                last_session = subj.aggr(behavior.TrialSet, session_start_time='max(session_start_time)')
                training_status = (behavior_analysis.SessionTrainingStatus & last_session).fetch1('training_status')
                
                if training_status in ['trained', 'ready for ephys']:
                    first_trained_session = subj.aggr(behavior_analysis.SessionTrainingStatus & 'training_status="trained"', first_trained='min(session_start_time)')
                    first_trained_session_time = first_trained_session.fetch1('first_trained')
                    # convert to timestamp
                    trained_date = pd.DatetimeIndex([first_trained_session_time])[0]
                    # how many days to training?
                    days_to_trained = sum(behav['date'].unique() < trained_date.to_datetime64())
                    # how many trials to trained?
                    trials_to_trained = sum(behav['date'] < trained_date.to_datetime64())
                    #average threshold
                    pars = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate.PsychResults * subject.Subject * subject.SubjectLab & \
                                         'subject_nickname="%s"'%mouse & 'lab_name="%s"'%labname).fetch(as_dict=True))
                    average_threshold  = pars.loc[(pars['prob_left'] == 0.5) & (pars['session_date'] \
                                                >= first_trained_session_time.date()), 'threshold'].mean()
                    average_lapse_high  = pars.loc[(pars['prob_left'] == 0.5) & (pars['session_date'] \
                                                >= first_trained_session_time.date()), 'lapse_high'].mean()
                    average_lapse_low  = pars.loc[(pars['prob_left'] == 0.5) & (pars['session_date'] \
                                                >= first_trained_session_time.date()), 'lapse_low'].mean()
                else:   
                    days_to_trained = np.nan
                    trials_to_trained = np.nan
                    average_threshold = np.nan
                    average_lapse_high = np.nan
                    average_lapse_low = np.nan
    
                if training_status == 'ready for ephys':
                    #Only counting from ready to ephys status
                    first_ephystrained_session = subj.aggr(behavior_analysis.SessionTrainingStatus & \
                                                           'training_status="ready for ephys"', first_ephystrained='min(session_start_time)')
                    first_ephystrained_session_time = first_ephystrained_session.fetch1('first_ephystrained')
                    pars = pd.DataFrame((behavior_analysis.BehavioralSummaryByDate.PsychResults * \
                                         subject.Subject * subject.SubjectLab & 'subject_nickname="%s"'%mouse & \
                                         'lab_name="%s"'%labname).fetch(as_dict=True))
                    average_bias_08  = pars.loc[(pars['prob_left'] == 0.8) & (pars['session_date'] \
                                                >= first_ephystrained_session_time.date()), 'bias'].mean()
                    average_bias_02  = pars.loc[(pars['prob_left'] == 0.2) & (pars['session_date'] \
                                                >= first_ephystrained_session_time.date()), 'bias'].mean()
                    
                else:
                    average_bias_08 = np.nan
                    average_bias_02= np.nan
                    
                # keep track
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['days_to_trained']] = days_to_trained
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['trials_to_trained']] = trials_to_trained
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['training_status']] = training_status
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_threshold']] = average_threshold
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_lapse_high']] = average_lapse_high
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_lapse_low']] = average_lapse_low
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_bias08']] = average_bias_08
                allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['average_bias02']] = average_bias_02
                
            except:
                pass
        
#Star plotting
#Make sublist with labs that have trained males and female
#TODO dectect this condition automatically
allsubjects['sex of the experimenter'] = "F"
allsubjects.loc[((allsubjects['lab_name']== 'cortexlab') | (allsubjects['lab_name']== 'wittenlab')|(allsubjects['lab_name']=='angelakilab') | (allsubjects['lab_name']=='danlab')), ['sex of the experimenter']] = "M"
subjects_mixed = allsubjects.loc[((allsubjects['lab_name']== 'churchlandlab')|(allsubjects['lab_name']=='angelakilab') | (allsubjects['lab_name']=='cortexlab'))]


##Plots per session
#Total - day
sns.set()
total_day = plt.figure(figsize=(10,6))
sns.boxplot(x="sex", y="days_to_trained", data=allsubjects )
sns.swarmplot(x="sex", y="days_to_trained", data=allsubjects,hue="lab_name", edgecolor="white")

#Per Lab - day
lab_day = plt.figure(figsize=(10,6))
sns.boxplot(x="lab_name", y="days_to_trained", hue="sex",
            data=allsubjects)

#Interaction - day
interaction_day = plt.figure(figsize=(10,6))
sns.catplot(x="sex", y="days_to_trained",col='sex of the experimenter', data=subjects_mixed, kind="box")

##Plots per trial
#Total - trial
total_trial = plt.figure(figsize=(10,6))
sns.boxplot(x="sex", y="trials_to_trained", data=allsubjects )
sns.swarmplot(x="sex", y="trials_to_trained", data=allsubjects,hue="lab_name", edgecolor="white")

#Per Lab - trial
lab_trial = plt.figure(figsize=(10,6))
sns.boxplot(x="lab_name", y="trials_to_trained", hue="sex",
            data=allsubjects)

#Interaction - day
interaction_trial = plt.figure(figsize=(10,6))
interaction_trial = sns.catplot(x="sex", y="trials_to_trained",col='sex of the experimenter', data=subjects_mixed, kind="box")

###############################################################################
#Analysis of psychometrics
###############################################################################


##Plot average psychometric measures
#Stats:
measures = pd.DataFrame (index = ['trials_to_trained', 'average_bias08', \
                                  'average_bias02', 'average_threshold', \
                                  'average_lapse_high', 'average_lapse_low']\
                                    , columns = ['s','p'])

for measure in measures.index.values:
    _, p_norm_M = stats.normaltest(allsubjects.loc[(allsubjects['sex'] == 'M'), \
                                             measure].dropna())
    _, p_norm_F = stats.normaltest(allsubjects.loc[(allsubjects['sex'] == 'F'), \
                                             measure].dropna())
    if p_norm_F >0.05 and p_norm_M >0.05:
        s ,  p = stats.ttest_ind(allsubjects.loc[(allsubjects['sex'] == 'M'), \
                                                 measure].dropna(), allsubjects.loc[(allsubjects['sex'] == 'F'), \
                                                        measure].dropna())
    else:
        s ,  p = stats.mannwhitneyu(allsubjects.loc[(allsubjects['sex'] == 'M'), \
                                                 measure].dropna(), allsubjects.loc[(allsubjects['sex'] == 'F'), \
                                                        measure].dropna())
    measures.loc[measure, 's'] = s
    measures.loc[measure, 'p'] = p * len(measures.index.values)  #Simple Bonferroni Correction
   
#Ceil p values to 1
measures.loc[(measures['p'] > 1), 'p'] =1
   
#plotting
psychometric_measures, ax = plt.subplots(2,3,figsize=(10,6))
sns.boxplot(x="sex", y="average_threshold", data=allsubjects, ax=ax[0,0])
sns.swarmplot(x="sex", y="average_threshold", data=allsubjects,hue="lab_name", edgecolor="white", ax=ax[0,0])
ax[0,0].legend_.remove()
sns.boxplot(x="sex", y="average_lapse_high", data=allsubjects, ax=ax[0,1])
sns.swarmplot(x="sex", y="average_lapse_high", data=allsubjects,hue="lab_name", edgecolor="white", ax=ax[0,1])
ax[0,1].legend_.remove()
sns.boxplot(x="sex", y="average_lapse_low", data=allsubjects, ax=ax[0,2] )
sns.swarmplot(x="sex", y="average_lapse_low", data=allsubjects,hue="lab_name", edgecolor="white", ax=ax[0,2])
ax[0,2].legend_.remove()
sns.boxplot(x="sex", y="average_bias08", data=allsubjects, ax=ax[1,0])
sns.swarmplot(x="sex", y="average_bias08", data=allsubjects,hue="lab_name", edgecolor="white", ax=ax[1,0])
ax[1,0].legend_.remove()
sns.boxplot(x="sex", y="average_bias02", data=allsubjects, ax=ax[1,1] )
sns.swarmplot(x="sex", y="average_bias02", data=allsubjects,hue="lab_name", edgecolor="white", ax=ax[1,1])
ax[1,1].legend_.remove()
plt.tight_layout()


##Save figs
total_day.savefig("total_day.pdf", bbox_inches='tight')
lab_day.savefig("lab_day.pdf", bbox_inches='tight')
interaction_day.savefig("interaction_day.pdf", bbox_inches='tight')
total_trial.savefig("total_triL.pdf", bbox_inches='tight')
lab_trial.savefig("lab_trial.pdf", bbox_inches='tight')
interaction_trial.savefig("interaction_trial.pdf")
psychometric_measures.savefig("psychometric_measures.pdf")