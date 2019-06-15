#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:55:17 2019


Predictors:  Rewarded choice and unrewarded choice
These two predictors include  correct choice and reward i.e

R (Right_Choice = True, Correct_choice = True, Reward = True) = 1
R (Right_Choice = True, Correct_choice = False, Reward = True) = 1 - This never happens
U(Right_Choice = True, Correct_choice = False, Reward = False) = 0
U(Right_Choice = True, Correct_choice = True, Reward = False) = 0

R (Right_Choice = False, Correct_choice = True, Reward = True) =  -1
R (Right_Choice = False, Correct_choice = False, Reward = True) =  -1 - This never happens
U(Right_Choice = False, Correct_choice = False, Reward = False) = 0 /
U(Right_Choice = False, Correct_choice = True, Reward = False) =  0 /


TODO:  invert choice vector so that negative value correspond to negative contrast in correct trials?

@author: ibladmin

TODO:  Description of psy_df

"""

import statsmodels.api as sm
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import time
import statsmodels.genmod.bayes_mixed_glm as  bayes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

 
def  glm_logit(psy_df, sex_diff = True):

   ##calculate useful variables
    
    #Calculate signed contrast
    if not 'signed_contrasts' in psy_df:
        psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
        psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
        psy_df.loc[:,'signed_contrasts'] =  (psy_df['contrastRight'] - psy_df['contrastLeft'])

        
    #Add sex if not present
    if not 'sex' in psy_df:
        psy_df.loc[:,'sex'] = np.empty([psy_df.shape[0],1])
        mice  = sorted(psy_df['mouse_name'].unique())
        for mouse in mice:
            sex = input('Sex of animal ')
            psy_df.loc[ psy_df['mouse_name'] == mouse, ['sex']]  = sex
    
    #make separate datafrme 
    data =  psy_df.loc[ :, ['sex', 'mouse_name', 'feedbackType', 'signed_contrasts', 'choice','ses']]
    
    ## Build predictor matrix
    
    #Rewardeded choices: 
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == -2) , 'rchoice']  = 0
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == -1) , 'rchoice']  = 0
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == 1) , 'rchoice']  = -1
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == 1) , 'rchoice']  = 1
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == -2) , 'rchoice']  = 0
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == -1) , 'rchoice']  = 0
    
    #Unrewarded choices: 
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == -2) , 'uchoice']  = -1
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == -1) , 'uchoice']  = -1
    data.loc[(data['choice'] == -1) & (data['feedbackType'] == 1) , 'uchoice']  = 0
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == 1) , 'uchoice']  = 0
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == -2) , 'uchoice']  = 1
    data.loc[(data['choice'] == 1) & (data['feedbackType'] == -1) , 'uchoice']  = 1
    
    #Drop nogos
    data = data.drop(data.index[data['feedbackType'] == 0],axis=0)
    ## Change -1 for 0 in choice 
    data.loc[(data['choice'] == -1), 'choice'] = 0
    
    #make Revidence and LEvidence
    data.loc[(data['signed_contrasts'] >= 0), 'Revidence'] = data.loc[(data['signed_contrasts'] >= 0), 'signed_contrasts'].abs()
    data.loc[(data['signed_contrasts'] <= 0), 'Revidence'] = 0
    data.loc[(data['signed_contrasts'] <= 0), 'Levidence'] = data.loc[(data['signed_contrasts'] <= 0), 'signed_contrasts'].abs()
    data.loc[(data['signed_contrasts'] >= 0), 'Levidence'] = 0
    
    
    #previous choices and evidence
    
    no_tback = 5 #no of trials back
    
    start = time.time()
    for date in sorted(data['ses'].unique()):
        for i in range(no_tback):
            data.loc[data['ses'] == date,'rchoice%s' %str(i+1)] =  data.loc[data['ses'] == date,'rchoice'].shift(i+1) #no point in 0 shift
            data.loc[data['ses'] == date,'uchoice%s' %str(i+1)] =  data.loc[data['ses'] == date,'uchoice'].shift(i+1) #no point in 0 shift
            data.loc[data['ses'] == date,'Levidence%s' %str(i+1)] =  data.loc[data['ses'] == date,'Levidence'].shift(i+1) #no point in 0 shift
            data.loc[data['ses'] == date,'Revidence%s' %str(i+1)] =  data.loc[data['ses'] == date,'Revidence'].shift(i+1) #no point in 0 shift
    end = time.time()
    print(end - start)
    
    
    #Remove first 5 trials from each ses
    #for date in sorted(data['ses'].unique()):
    #    data  = data.drop(data.index[data['ses'] == date][0:5] ,axis=0)
        
    #data = data.reset_index()
    #Drop unnecessary elements
    #data =  data.drop(columns  = ['feedbackType','sex', 'ses','signed_contrasts', 'index'])
    
    data =  data.dropna()
    
    mdata = data.loc[(data['sex'] == 'M')]
    fdata = data.loc[(data['sex'] == 'F')]
    ## construct our model, with contrast as a variable
    
    ##Bayeasian mixed effects #need to change ident and exog_VC to account for mixed effects
    if sex_diff==True:
        mresult, mr2 = load_regression(mdata)
        fresult, fr2 = load_regression(fdata)
        result, r2  =  load_regression(data)
        return mresult, fresult, result, mr2, fr2, r2
    
    
    else:  
        result, r2  =  load_regression(data)
        return result,  r2
    
    

def load_regression (data, mixed_effects  = False):
    
    endog  = pd.DataFrame(data['choice'])
    exog  = data[[ 'Revidence', 'Levidence',\
            'rchoice1', 'uchoice1', 'Levidence1',\
                  'Revidence1', 'rchoice2', 'uchoice2', 'Levidence2', 'Revidence2',\
                  'rchoice3', 'uchoice3', 'Levidence3', 'Revidence3', 'rchoice4',\
                  'uchoice4', 'Levidence4', 'Revidence4', 'rchoice5', 'uchoice5',\
                  'Levidence5', 'Revidence5']]
    
    #cols = list(exog.columns)
    
    #Normalising contrast
    for col in cols:
        col_zscore = col + '_zscore'
        exog[col_zscore] = (exog[col] - exog[col].mean())/exog[col].std(ddof=0)
    
    exog  =exog.drop(columns = [ 'Revidence', 'Levidence',\
            'rchoice1', 'uchoice1', 'Levidence1',\
                      'Revidence1', 'rchoice2', 'uchoice2', 'Levidence2', 'Revidence2',\
                      'rchoice3', 'uchoice3', 'Levidence3', 'Revidence3', 'rchoice4',\
                      'uchoice4', 'Levidence4', 'Revidence4', 'rchoice5', 'uchoice5',\
                      'Levidence5', 'Revidence5'] )
        
    
    ##Cross validation
    
    if mixed_effects == False :

    ##Frequentist wihtout mixed effects
        X_train, X_test, y_train, y_test = train_test_split(exog, np.ravel(endog), test_size=0.3)
        logit_model = sm.Logit(y_train,X_train)
        result=logit_model.fit()
        print(result.summary2())

        #crossvalidation
        
        
        cv = KFold(n_splits=10,  shuffle=False)
        logreg1 = LogisticRegression()
        r2  = cross_val_score(logreg1, exog, endog, cv=10)
        print( 'Accuracy  = ' , r2.mean())

       
    #cross validate  with sklearn 
    #NOTE: Currently cross validating without mixed effects
    
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        accu = logreg.score(X_test, y_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

    return result, r2


"""
TODO nmiced effects GLM

     X_train = X_train.drop(columns = 'mouse_name')
    X_test = X_test.drop(columns = 'mouse_name')
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    Mixed effects models no done yet
    y_train = pd.DataFrame(y_train)
    exog_vc = X_train
    for i, name in enumerate(sorted(X_train['mouse_name'].unique())):
        exog_vc.loc[(exog_vc['mouse_name'] == name)] = i
    ident =  np.ones(exog_vc.shape[1], dtype=np.int)
    X_train = X_train.drop(columns = 'mouse_name')
    X_test = X_test.drop(columns = 'mouse_name')
    exog_vc  = exog_vc.drop(columns = 'mouse_name')
    exog_vc = pd.DataFrame(exog_vc.iloc[:,1])
    exog_vc  = exog_vc + 1
    ident  = np.ones([1])  
    model1 = bayes.BinomialBayesMixedGLM(y_train, X_train, exog_vc,ident)
    result = model1.fit_map()
    result.summary() 
    
"""
    

#####
#To do model psychometric
####



    
    