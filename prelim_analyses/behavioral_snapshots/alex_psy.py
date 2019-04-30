#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:11:57 2019

@author: Alejandro
"""

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