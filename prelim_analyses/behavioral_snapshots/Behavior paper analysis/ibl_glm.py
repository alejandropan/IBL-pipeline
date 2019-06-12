#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 01:49:23 2019
TODO : Divide by sex of the animal
prepare ibl trial dataframe for IBL
@author: ibladmin
"""

import matplotlib.pyplot as plt
import pandas as pd
## CONNECT TO datajoint
import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, behavior
from load_mouse_data_datajoint import *  # this has all plotting functions
import seaborn as sns
from glm import *




key = ((subject.Subject()  & 'sex!="U"') * (behavior.TrialSet() & 'n_trials > 100') * (subject.SubjectLab()) * (behavior_analysis.SessionTrainingStatus() & 'training_status="ready for ephys"  ')).fetch('KEY')
trials_ibl = pd.DataFrame.from_dict((subject.Subject() * behavior.TrialSet.Trial & key).fetch(as_dict=True))

trials_ibl['signed_contrasts'] = trials_ibl['trial_stim_contrast_right'] - trials_ibl['trial_stim_contrast_left']

##Rename for GLM function
trials_ibl = trials_ibl.rename(index=str, columns={"session_start_time": "ses", 
                                      "subject_uuid": "mouse_name", 
                                      "trial_feedback_type": "feedbackType", 
                                      "trial_response_choice":"choice"})

#Rename choices
trials_ibl.loc[(trials_ibl['choice']=='CW'),'choice'] = -1
trials_ibl.loc[(trials_ibl['choice']=='CCW'), 'choice'] = 1
trials_ibl.loc[(trials_ibl['choice']=='No Go'), 'choice'] = 0

#Remove 0.5 block?


psy_df =  trials_ibl.loc[(trials_ibl['trial_stim_prob_left'] == 0.8) | (trials_ibl['trial_stim_prob_left'] == 0.2)]

mresult, fresult, mr2, fr2  = glm_logit(psy_df)

mresults  =  pd.DataFrame({"Predictors": mresult.model.exog_names , "Coef" : mresult.params.values,\
                          "SEM": mresult.bse.values, "Sex": "M"})
mresults = mresults.reindex(mresults.Coef.abs().sort_values(ascending=False).index)
fresults  =  pd.DataFrame({"Predictors": fresult.model.exog_names , "Coef" : fresult.params.values,\
                          "SEM": mresult.bse.values, "Sex": "F"}).reindex(mresults.Coef.abs().sort_values().index)
fresults = fresults.reindex(fresults.Coef.abs().sort_values(ascending=False).index)
results  = pd.concat([mresults, fresults]) 


#Plotting
fig, ax = plt.subplots(figsize=(12, 9))
ax  = sns.barplot(x = 'Predictors', y = 'Coef', data=results, hue='Sex')    
ax.set_xticklabels( results['Predictors'], rotation=-90)
ax.set_ylabel('coef')
ax.axhline(y=0, linestyle='--', color='black', linewidth=2)
fig.suptitle ('GLM Biased Blocks')
fig.savefig("glm_sex_diff.pdf")

