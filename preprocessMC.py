'''
In the following lines I mount google drive and define working directories
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
import glob
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
data_dir =  '/content/gdrive/My Drive/Colab Notebooks/Data/MyAnalysis/'
data_dir3 = data_dir +'YacineModels/'
filepath = data_dir3+'Unified/' # just a procedure to align the filepath of the model checkpoint class to a target folder in google drive

'''
In the following lines I do implement the reading of the 2018 ultra legacy MC .parquet files and apply pre-processing cuts
'''
data_2018 = pd.read_parquet(data_dir + 'df_2018_gnninput.parquet') # first reading of the parquet file

'''
In the next line I remove the LO VBS events (I work only with NLO VBS), the InDYamc and HTDy Drell-Yan MC events (since I've chosen to work only with PTDy).
'''
data_2018 = data_2018[(data_2018['proc'] != 'ZZ_EWK_LO') & (data_2018['proc'] != 'IncDYamc') & (data_2018['proc'] != 'HTDY')]

'''
The next line selects the input features for the MLP model (note that ngood_bjets is used as a preselection cut feature, not as an input feature). ngood_jets, on the other hand, is an input feature, but it is 
binary, since it is in charge of specifying exactly whether the event has exactly 2 jets or at least 3 jets, providing that information to the MLP model. This is done to help the MLP not be confused with
the value 0.0 initialized in the third_jet_pt, third_jet_eta and third_jet_phi variables, when the event has exactly 2 jets.
'''
interest_columns = ['lead_jet_pt','lead_jet_eta','lead_jet_phi','trail_jet_pt','trail_jet_eta','trail_jet_phi','third_jet_pt','third_jet_eta','third_jet_phi',\
'leading_lep_pt','leading_lep_eta','leading_lep_phi','trailing_lep_pt', 'trailing_lep_eta','trailing_lep_phi','met_pt','met_phi','dijet_mass','dijet_deta','dilep_m','ngood_jets','ngood_bjets']

'''
The following values are used in the job of applying pre-processing cuts, which are applied only to remove huge outliers that can prejudice the MinMaxScalling of the input features
'''
up_threshold_values = [1200,7,np.pi,600,7,np.pi,270,7,np.pi,700,7,np.pi,300,7,np.pi,800,np.pi,3200.0,8.8,106.0,np.inf,1.0]

'''
The next line does the same as above, but defines the pré-processing cuts of the lower region for each input feature, in order to avoid the existence of outliers event and prejudice MinMaxScalling
'''
low_threshold_values = [30,-7,-np.pi,30,-7,-np.pi,-np.inf,-np.inf,-np.inf,20,-7,-np.pi,20,-7,-np.pi,30,-np.pi,0.0,0.5,76.0,1.0,-1]

'''
The next line in effect applies the pré-processing cuts
'''
for i in range(len(interest_columns)):
  data_2018 = data_2018[(data_2018[interest_columns[i]] < up_threshold_values[i]) & (data_2018[interest_columns[i]] > low_threshold_values[i])] 

bkgLabel = 0.0 # defines which target will be used as the background target value

sig = data_2018.loc[(data_2018['proc'] == 'ZZ_EWK_NLO_EE') | (data_2018['proc'] == 'ZZ_EWK_NLO_MuMu'),data_2018.columns] # The definition of the signal events dataframe
bkg = data_2018.loc[(data_2018['proc'] != 'ZZ_EWK_NLO_EE') & (data_2018['proc'] != 'ZZ_EWK_NLO_MuMu'),data_2018.columns] # The definition of the background events dataframe
sig['isSignal'] = np.ones((len(sig),)) # The definition of a column for us to identify each row of the formerly defined signal dataframe as a signal event tuple
bkg['isSignal'] = np.zeros((len(bkg),)) + bkgLabel #The definition of a column for us to identify each row of the formerly defined background dataframe as a signal event tuple
df_all = pd.concat([sig,bkg]) # Concatenate signal and background into an unique dataframe

df_allEta = pd.DataFrame(df_all[(df_all['ngood_jets'] >= 2)],copy=True) # this is applied only to select the events in the dataframes that have at least 2 good jets

'''
The next line in effect selects the input features that the ML model will use to train with
'''
df_allEta = df_allEta[['lead_jet_pt','lead_jet_eta','lead_jet_phi','trail_jet_pt','trail_jet_eta','trail_jet_phi','third_jet_pt','third_jet_eta','third_jet_phi','leading_lep_pt','leading_lep_eta','leading_lep_phi','trailing_lep_pt','trailing_lep_eta','trailing_lep_phi','met_pt','met_phi','dijet_mass','dijet_deta','dilep_m','ngood_jets','final_weight','isSignal',\
                       'proc']] 

'''
This next line is in charge of setting the value 3 for the number of good jets, in the events that have more than 2. This is done this way because Yacine told me that we can not trust the ngood_jets
variable to tell exactly the number of good jets the event has, but we can trust the information that the event has more than 2, every time we have the ngood_jets variable greater than 2
'''
df_allEta.loc[df_allEta['ngood_jets'] > 2,'ngood_jets'] = 3 

'''
The next 3 lines are important to give those quantities a value different than -99, every time the event has only 2 jets: since we must format the dataframe to contain all the events, no matter the number of good jets
they have, every time the event has only 2 good jets its -99 value squeezes the at least 3 jets events information to a very small region in the 0.0 to 1.0 interval. This is done as a pre-processing procedure to enhance
the MinMaxScalling.
'''
df_allEta.loc[df_allEta['ngood_jets'] == 2,'third_jet_pt'] = 30.0 # This is done as a pre-processing procedure to enhance the MinMaxScalling, the original value is -99 and prejudices MinMaxScalling

df_allEta.loc[df_allEta['ngood_jets'] == 2,'third_jet_eta'] = 0.0 # This is done with the same purpose: enhance MinMaxScalling of the feature, the original value is -99 and prejudices MinMaxScalling
df_allEta.loc[df_allEta['ngood_jets'] == 2,'third_jet_phi'] = 0.0 # This is done with the same purpose: enhance MinMaxScalling of the feature, the original value is -99 and prejudices MinMaxScalling

bkgSumWeight = df_allEta.loc[df_allEta['isSignal'] == bkgLabel ,'final_weight'].sum() # Expresses the number of background events of the Run II 2018
sigSumWeight = df_allEta.loc[df_allEta['isSignal'] == 1.0 ,'final_weight'].sum() # Expresses the number of NLO VBS, with final state in 2l 2nu events, for the Run II 2018

df_allEta.loc[df_allEta['isSignal'] == bkgLabel ,'final_weight'] /= bkgSumWeight # This lines normalizes the background number of events, maintaining the relative proportions between the different kinds of backgrounds
df_allEta.loc[df_allEta['isSignal'] == 1.0 ,'final_weight'] /= sigSumWeight # This line normalizes the NLO VBS with final state at 2l 2nu events.

from sklearn.preprocessing import MinMaxScaler

'''
The next line gets the values from the lead_jet_pt to ngood_jets parameters, they will be used as input features for the MLP, the ones it can rely upon to separate signal from background
'''
dataset = df_allEta.iloc[:,0:21].to_numpy()
dataset_W = df_allEta.iloc[:,21].to_numpy() # Defines a numpy array that represent normalized events weights, doesn't care if it's a background or a signal event
dataset_Y = df_allEta.iloc[:,22].to_numpy() # Defines the numpy array that stands for the targets of the events
scaler = MinMaxScaler() # Defines a MinMaxScaller
scaler.fit(dataset) # Get the scalling parameters for each column of the numpy 2-D array representing the input features of the ML model
dataset = scaler.transform(dataset) # transform the input features matrix, so that it's easier now for the ML model to converge when training and validating
