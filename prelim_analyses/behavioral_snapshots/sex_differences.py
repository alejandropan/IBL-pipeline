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

## CONNECT TO datajoint

import datajoint as dj
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis
from load_mouse_data_datajoint import *  # this has all plotting functions


#Collect all alyx data
allsubjects = pd.DataFrame.from_dict(((subject.Subject() - subject.Death()) & 'sex!="U"').fetch(as_dict=True, order_by=['lab_name', 'subject_nickname']))
#allsubjects = pd.DataFrame.from_dict((subject.Subject() & 'sex!="U"' & 'subject_birth_date>"2018-10-15"' ).fetch(as_dict=True, order_by=['lab_name', 'subject_nickname']))
allsubjects = pd.DataFrame.from_dict((subject.Subject() & 'sex!="U"' & 'subject_birth_date>"2018-10-15"' ).fetch(as_dict=True, order_by=['lab_name', 'subject_nickname']))


if allsubjects.empty:
    raise ValueError('DataJoint seems to be down, please try again later')
#Drop double entries
allsubjects =  allsubjects.drop_duplicates('subject_nickname')
#Add learning rate columns
allsubjects['training_status'] =np.nan
allsubjects['days_to_trained'] = np.nan
allsubjects['trials_to_trained'] = np.nan

users = allsubjects['lab_name'].unique()


for lab in users:
    # take mice from this lab only
    subjects = allsubjects.loc[allsubjects['lab_name']=="%s" % lab]
    for mouse in subjects['subject_nickname']:
        try:
            # TRIAL COUNTS AND SESSION DURATION
            behav = get_behavior(mouse, lab)
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
            else:
                days_to_trained = np.nan
                trials_to_trained = np.nan
            # keep track
            allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['days_to_trained']] = days_to_trained
            allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['trials_to_trained']] = trials_to_trained
            allsubjects.loc[allsubjects['subject_nickname'] == mouse, ['training_status']] = training_status
        except:
            pass
        
#Star plotting
#Make sublist with labs that have trained males and female
#TODO dectect this condition automatically
allsubjects['sex of the experimenter'] = "F"
allsubjects.loc[((allsubjects['lab_name']== 'wittenlab')|(allsubjects['lab_name']=='angelakilab') | (allsubjects['lab_name']=='danlab')), ['sex of the experimenter']] = "M"
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

#Per Lab - day
lab_trial = plt.figure(figsize=(10,6))
sns.boxplot(x="lab_name", y="trials_to_trained", hue="sex",
            data=allsubjects)

#Interaction - day
interaction_trial = plt.figure(figsize=(10,6))
interaction_trial = sns.catplot(x="sex", y="trials_to_trained",col='sex of the experimenter', data=subjects_mixed, kind="box")

##Save figs
total_day.savefig("total_day.pdf", bbox_inches='tight')
lab_day.savefig("lab_day.pdf", bbox_inches='tight')
interaction_day.savefig("interaction_day.pdf", bbox_inches='tight')
total_trial.savefig("total_triL.pdf", bbox_inches='tight')
lab_trial.savefig("lab_trial.pdf", bbox_inches='tight')
interaction_trial.savefig("interaction_trial.pdf")