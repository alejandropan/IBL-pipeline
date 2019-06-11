#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 01:49:23 2019
TODO : Divide by sex of the animal
prepare ibl trial dataframe for IBL
@author: ibladmin
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
## CONNECT TO datajoint
import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from load_mouse_data_datajoint import *  # this has all plotting functions





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

psy_df =  trials_ibl