# @alejandropan 2019

#Import some general packages
#Notes: Currently psychometric plot is taking into account ibl_witten_01 which is a bad dataset
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
from ibl_pipeline.utils import psychofit as psy
import seaborn as sns
from alex_psy import *

###Local
#TODO add restriction for untrainable #2,793,497 trials #89,442 from witten trained
#Need to make this a lot more function based
key = ((subject.Subject()) * (behavior.TrialSet() & 'n_trials > 100') * (subject.SubjectLab() & 'lab_name="wittenlab"') * (behavior_analysis.SessionTrainingStatus() & 'training_status="trained"  ')).fetch('KEY')
trials_princeton = pd.DataFrame.from_dict((behavior.TrialSet.Trial & key).fetch(as_dict=True))




#t_info = trials.proj('trial_response_choice', signed_contrast='trial_stim_contrast_right - trial_stim_contrast_left')
#q = dj.U('signed_contrast').aggr(t_info, n='count(*)', n_right='sum(trial_response_choice="CCW")')
#result = q.proj('n', 'n_right', 'signed_contrast', prop_right='n_right / n')

#right_trials, total_trials, prop_right_trials, signed_contrasts = result.fetch('n_right', 'n', 'prop_right', 'signed_contrast')

#fig, ax = plt.subplots(1, 1, dpi=150)
#ax.plot(signed_contrasts * 100, prop_right_trials)
#ax.set_xlabel('Signed Contrast (%)')
#ax.set_ylabel('P(right choice)')

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

psy_df_local = psy_by_mouse_local (unique_signed_contrasts,'wittenlab')
y_local = psy_df_local.mean()
x_local = unique_signed_contrasts
local_error =  psy_df_local.std()



#fig, ax = plt.subplots(1, 1, dpi=150)
#ax.plot(unique_signed_contrasts * 100, prop_right_trials)
#ax.plot(x, y)
#ax.set_xlabel('Signed Contrast (%)')
#ax.set_ylabel('P(right choice)')

##IBLWide
key = ((subject.Subject()) * (behavior.TrialSet() & 'n_trials > 100') * (subject.SubjectLab()) * (behavior_analysis.SessionTrainingStatus() & 'training_status="trained"  ')).fetch('KEY')
trials_ibl = pd.DataFrame.from_dict((behavior.TrialSet.Trial & key).fetch(as_dict=True))
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

x2 = np.linspace(-100, 100)
y2 = psy.erf_psycho_2gammas(pars, x)

#Psychometric by mouse using @alejandro functions

psy_df = psy_by_mouse(unique_signed_contrasts)
y_ibl = psy_df.mean()
x_ibl = unique_signed_contrasts 
ibl_error =  psy_df.std()


#Data for psychometrics per session
princeton_ses = trials_princeton.groupby(['session_start_time']).count()['trial_id']
ibl_ses = trials_ibl.groupby(['session_start_time']).count()['trial_id']
#Average performance across time

##IBLWide
key = ((subject.Subject()) * (behavior.TrialSet() & 'n_trials > 100') * (subject.SubjectLab()) * (behavior_analysis.SessionTrainingStatus() & 'training_status="training in progress"  ')).fetch('KEY')
trials_ibl_training = pd.DataFrame.from_dict((behavior.TrialSet.Trial & key).fetch(as_dict=True))

easy_trials  = trials_ibl_training.loc[(trials_ibl_training['trial_stim_contrast_left'] >=0.5) \
                                        | (trials_ibl_training['trial_stim_contrast_right']\
                                        >= 0.5)]
                                        
#Calculate normalized days from start of training (need to make into function)
easy_trials['norm_days'] = np.nan
for mouse  in easy_trials['subject_uuid'].unique():
   for sess in  easy_trials.loc[easy_trials['subject_uuid'] == mouse]['session_start_time'].unique():
       easy_trials['norm_days'].loc[(easy_trials['subject_uuid'] == mouse) & \
     (easy_trials['session_start_time']==sess)] = sum(easy_trials.loc[easy_trials['subject_uuid'] \
       == mouse]['session_start_time'].unique() < sess)
     #Make boolean of correct choices
easy_trials['correct'] =  easy_trials ['trial_feedback_type']==1  

    #easy_trials_by_training_day = easy_trials.groupby(['norm_days']).mean()
easy_trials_by_session =  easy_trials.groupby(['subject_uuid','norm_days'])['correct'].mean().reset_index() 

##Same for Princeton
key = ((subject.Subject()) * (behavior.TrialSet() & 'n_trials > 100') * (subject.SubjectLab() & 'lab_name="wittenlab"') * (behavior_analysis.SessionTrainingStatus() & 'training_status="training in progress"')).fetch('KEY')
trials_p_training = pd.DataFrame.from_dict((behavior.TrialSet.Trial & key).fetch(as_dict=True))
easy_trials  = trials_p_training.loc[(trials_ibl_training['trial_stim_contrast_left'] >=0.5) \
                                        | (trials_ibl_training['trial_stim_contrast_right']\
                                        >= 0.5)]
#here norm_days function when made will be called
                                        
per_com, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_title('Princeton')
ax1.set_xlabel('Training session')
ax1.set_ylabel('Performance (Fraction correct)')
ax2.set_title('IBL')
ax2.set_xlabel('Training session')
sns.lineplot(data = easy_trials_by_session, x ='norm_days', y='correct', color="#34495e")
sns.lineplot(data = easy_trials_by_session, x ='norm_days', y='correct', hue='subject_uuid', legend =False, palette="Blues_d",alpha=0.20)#, ax = ax2)

plt.savefig("training_com_average_p.svg", format="svg")

#Plotting

psy_com, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.set_title('Princeton')
ax1.set_xlabel('Signed Contrast (Fraction)')
ax1.set_ylabel('P(right choice)')
ax2.set_title('IBL')
ax2.set_xlabel('Signed Contrast (Fraction)')
sns.pointplot(data = psy_df, ax = ax2)
plt.savefig("psy_com.svg", format="svg")

#psychometric fit with two lapse rates #Princeton
psy_com_fit_local, (ax1) = plt.subplots()
plt.setp(ax1, xticks=unique_signed_contrasts, yticks = np.arange(min(x), max(x)+1, 0.1), )
ax1.plot(x/100,y, color='m') 
ax1.plot(x_local, y_local) 
ax1.errorbar(x_local, y_local, yerr=local_error)
ax1.set_title('Princeton')
ax1.set_xlabel('Signed Contrast (Fraction)')
ax1.set_ylabel('P(right choice)')
plt.savefig("psy_com_fit_local.svg", format="svg")





psy_com_fit_ibl, (ax1) = plt.subplots()
plt.setp(ax2, xticks=unique_signed_contrasts, yticks = np.arange(min(x2), max(x2)+1, 0.1), )
ax2.plot(x2/100,y2,color='m') 
ax2.plot(x_ibl, y_ibl) 
ax2.errorbar(x_ibl, y_ibl, yerr=ibl_error)
ax2.set_title('IBL')
ax2.set_xlabel('Signed Contrast (Fraction)')
ax2.set_ylabel('P(right choice)')
plt.savefig("psy_com_fit_ibl.svg", format="svg")


#Zoomed version
psy_com, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
plt.setp(ax1, xticks=unique_signed_contrasts, yticks = np.arange(min(x), max(x)+1, 0.1),\
         xlim=[-0.25, 0.25])
plt.setp(ax2, xticks=unique_signed_contrasts, xlim= [-0.25, 0.25])
ax1.plot(x/100,y, color='m') 

ax1.set_title('Princeton')
ax1.set_xlabel('Signed Contrast (Fraction)')
ax1.set_ylabel('P(right choice)')
ax2.set_title('IBL')
ax2.set_xlabel('Signed Contrast (Fraction)')
ax2.plot(x2/100,y2,color='m') 
plt.savefig("psy_com_fit_zoom.svg", format="svg")

#Stats on average number of trials  per session, bar plot IBL vs our lab

trial_ses_com, ax = plt.subplots()
sns.kdeplot(princeton_ses, shade=True, ax=ax, legend= False);
sns.kdeplot(ibl_ses, shade=True, ax=ax, legend= False);
ax.set_xlabel('Trials per session')
ax.set_ylabel('Probability Density')
ax.legend(('Princeton','IBL'))
plt.xlim([0,3000])
plt.savefig("trial_ses_com.svg", format="svg")


#Count trial number over same days
#Make extra df for witten  (trials df) 
#groupby['start_time'] for ibl and witten

