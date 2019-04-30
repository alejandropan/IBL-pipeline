# @alejandropan 2019

#Import some general packages
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

## CONNECT TO datajoint

import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'

from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis
from load_mouse_data_datajoint import *  # this has all plotting functions
from ibl_pipeline.utils import psychofit as psy
import seaborn as sns


#TODO add restriction for untrainable #2,793,497 trials #89,442 from witten trained
key = ((subject.Subject()) * (behavior.TrialSet() & 'n_trials > 100') * (subject.SubjectLab() & 'lab_name="wittenlab"') * (behavior_analysis.SessionTrainingStatus() & 'training_status="trained"  ')).fetch('KEY')

trials  = behavior.TrialSet.Trial & key

t_info = trials.proj('trial_response_choice', signed_contrast='trial_stim_contrast_right - trial_stim_contrast_left')
q = dj.U('signed_contrast').aggr(t_info, n='count(*)', n_right='sum(trial_response_choice="CCW")')
result = q.proj('n', 'n_right', 'signed_contrast', prop_right='n_right / n')

right_trials, total_trials, prop_right_trials, signed_contrasts = result.fetch('n_right', 'n', 'prop_right', 'signed_contrast')

fig, ax = plt.subplots(1, 1, dpi=150)
ax.plot(signed_contrasts * 100, prop_right_trials)
ax.set_xlabel('Signed Contrast (%)')
ax.set_ylabel('P(right choice)')

#Psychometric - A bit redudant but getting and error if not doing this

choice, cont_left, cont_right = (behavior.TrialSet.Trial & key).fetch('trial_response_choice', 
                                                                      'trial_stim_contrast_left', 
                                                                      'trial_stim_contrast_right')
signed_contrasts = cont_right - cont_left
right_choices = choice == 'CCW'
unique_signed_contrasts = np.unique(signed_contrasts)

total_trials = []
right_trials = []

for cont in unique_signed_contrasts:
    matching = (signed_contrasts == cont)
    total_trials.append(np.sum(matching))
    right_trials.append(np.sum(right_choices[matching]))

prop_right_trials = np.divide(right_trials, total_trials)

pars, L = psy.mle_fit_psycho(
        np.vstack([unique_signed_contrasts * 100, total_trials, prop_right_trials]),
        P_model='erf_psycho_2gammas',
        parstart=np.array([np.mean(unique_signed_contrasts), 20., 0.05, 0.05]),
        parmin=np.array([np.min(unique_signed_contrasts), 0., 0., 0.]),
        parmax=np.array([np.max(unique_signed_contrasts), 100., 1, 1]))

x = np.linspace(-100, 100)
y = psy.erf_psycho_2gammas(pars, x)

fig, ax = plt.subplots(1, 1, dpi=150)
ax.plot(unique_signed_contrasts * 100, prop_right_trials)
ax.plot(x, y)
ax.set_xlabel('Signed Contrast (%)')
ax.set_ylabel('P(right choice)')


#By mouse in lab

def psy_by_mouse_local (unique_signed_contrasts,labname):
        
        labname  = labname
        mice = pd.DataFrame.from_dict((subject.Subject()) 
            * (subject.SubjectLab() & 'lab_name="{}"'.format(labname)) 
            * (behavior_analysis.SessionTrainingStatus() & 'training_status="trained"  '))     
        
        psy_df = pd.DataFrame(columns = [unique_signed_contrasts])
        
        for row , mouse in enumerate(mice.subject_nickname.unique()):
            
            key = ((subject.Subject() & 'subject_nickname = "{}"'.format(mouse)) 
            * (behavior.TrialSet() & 'n_trials > 600') * (subject.SubjectLab() & 'lab_name="{}"'.format(labname)) 
            * (behavior_analysis.SessionTrainingStatus() & 'training_status="trained"  ')).fetch('KEY')
            
            choice, cont_left, cont_right = (behavior.TrialSet.Trial & key).fetch('trial_response_choice', 
                                                                                  'trial_stim_contrast_left', 
                                                                                  'trial_stim_contrast_right')
            signed_contrasts = cont_right - cont_left
            right_choices = choice == 'CCW'
            
            total_trials = []
            right_trials = []
            
            for cont in unique_signed_contrasts:
                matching = (signed_contrasts == cont)
                total_trials.append(np.sum(matching))
                right_trials.append(np.sum(right_choices[matching]))
            
            prop_right_trials = np.divide(right_trials, total_trials)
            psy_df.loc[row,:] = prop_right_trials
            
        return psy_df
        
#By mouse total
def psy_by_mouse (unique_signed_contrasts):
   
    
      mice = pd.DataFrame.from_dict((subject.Subject()) 
            * (subject.SubjectLab()) 
            * (behavior_analysis.SessionTrainingStatus() & 'training_status="trained"  '))     
        
      psy_df = pd.DataFrame(columns = [unique_signed_contrasts])
      
      for row , mouse in enumerate(mice.subject_nickname.unique()):
            
            key = ((subject.Subject() & 'subject_nickname = "{}"'.format(mouse)) 
            * (behavior.TrialSet() & 'n_trials > 100') * (subject.SubjectLab()) 
            * (behavior_analysis.SessionTrainingStatus() & 'training_status="trained"  ')).fetch('KEY')
            
            choice, cont_left, cont_right = (behavior.TrialSet.Trial & key).fetch('trial_response_choice', 
                                                                                  'trial_stim_contrast_left', 
                                                                                  'trial_stim_contrast_right')
            signed_contrasts = cont_right - cont_left
            right_choices = choice == 'CCW'
            
            total_trials = []
            right_trials = []
            
            for cont in unique_signed_contrasts:
                matching = (signed_contrasts == cont)
                total_trials.append(np.sum(matching))
                right_trials.append(np.sum(right_choices[matching]))
            
            prop_right_trials = np.divide(right_trials, total_trials)
            psy_df.loc[row,:] = prop_right_trials
        
      return psy_df
