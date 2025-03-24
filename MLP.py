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

'''
In the following lines I do implement the reading of the 2018 ultra legacy MC .parquet files and apply pre-processing cuts
'''
data_2018 = pd.read_parquet(data_dir + 'df_2018_gnninput.parquet') # first reading of the parquet file

'''
In the next line I remove the LO VBS events (I work only with NLO VBS), the InDYamc and HTDy Drell-Yan MC events (since I've chosen to work only with PTDy).
'''
data_2018 = data_2018[(data_2018['proc'] != 'ZZ_EWK_LO') & (data_2018['proc'] != 'IncDYamc') & (data_2018['proc'] != 'HTDY')]

'''
The next line chooses which input features the MLP model will use as input features (note that ngood_bjets is not used as input feature, as well as ngood_jets which is used only for a specific purpose I'll explain later)
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

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras import utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback

filepath = data_dir3+'Unified/' # just a procedure to align the filepath of the model checkpoint class to a target folder in google drive

'''
The following class definition is the custom significance class I use to choose the best training epoch, by selecting the highest validation point, as well as an agreement of less than 1% between its values for the 
training and validation datasets. I've also used it to understand if a MLP model is already very huge and unnecessary for the task of separating VBS from background
'''
@tf.keras.utils.register_keras_serializable()
class globalSignificance(tf.keras.metrics.Metric):

  def __init__(self, name='globalSig',threshold = 0.5,sigSumWeight = None,bkgSumWeight = None, **kwargs):
    super(globalSignificance, self).__init__(name=name, **kwargs)
    self.sigPart = self.add_weight(name='sigPart', initializer='zeros')
    self.bkgPart = self.add_weight(name='bkgPart', initializer='zeros')
    self.sigPartTotal = self.add_weight(name='sigPartTotal', initializer='zeros')
    self.bkgPartTotal = self.add_weight(name='bkgPartTotal', initializer='zeros')
    self.threshold = threshold
    self.sigSumWeight = sigSumWeight.astype('float32')
    self.bkgSumWeight = bkgSumWeight.astype('float32')
	  
  '''
  The next method implements the update of state variables that record training metrics between batches
  '''
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool) # cast float to boolean array
    y_pred = tf.squeeze(y_pred) # This line corrects a bug I didn't noticed in 2 years: it changes the y_pred size from (batch_size,1) to (batch_size,). That corrected the wrong final result I was obtaining with the metric

    y_pred = tf.cast(tf.greater(y_pred,self.threshold), tf.bool) # Set a value to True whenever the model assigns a value greater than the self.threshold parameter to an element in the prediction array.
    isSig = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)) # Sets True only to the true positive array elements
    isSig = tf.cast(isSig, self.dtype) # Cast from boolean array to tf.float32 array
    isBkg = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)) # Sets True only to the false positive array elements
    isBkg = tf.cast(isBkg, self.dtype) # Cast from boolean array to tf.float32 array
    isTotalSig = y_true # Array that is true whenever the batch element comes from a signal event
    isTotalSig = tf.cast(isTotalSig, self.dtype) # cast from boolean array to tf.float32 array
    isTotalBkg = tf.logical_not(y_true) # Array that is true only if the target if from a background event
    isTotalBkg = tf.cast(isTotalBkg, self.dtype) # cast from boolean array to tf.float32 array
  
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype) # Cast sample_weight to tf.float32
      isSig = tf.multiply(isSig, sample_weight) # Normalized vector of true positives: if the element in the batch is from a true positive then it receives the normalized weight corresponding to it 
      isBkg = tf.multiply(isBkg, sample_weight) # Normalized vector of false positives: if the element is from a false positive then it receives the normalized weight corresponding to it
      isTotalSig = tf.multiply(isTotalSig, sample_weight) # Normalized vector of weights of all the signal events in the batch
      isTotalBkg = tf.multiply(isTotalBkg, sample_weight) # Normalized vector of weights of all the background events in the batch

    self.sigPart.assign_add(self.sigSumWeight*tf.reduce_sum(isSig)) # Set the self.sigPart variable with the physical world number of true positive events in the current batch (have to denormalize to calculate significance)
    self.bkgPart.assign_add(self.bkgSumWeight*tf.reduce_sum(isBkg)) # Set the self.bkgPart variable with the physical world number of false positive events in the current batch (have to denormalize to calculate significance)
    self.sigPartTotal.assign_add(self.sigSumWeight*tf.reduce_sum(isTotalSig)) # the physical world number of signal events in the batch
    self.bkgPartTotal.assign_add(self.bkgSumWeight*tf.reduce_sum(isTotalBkg)) # the physical world number of background events in the batch

  '''
  In the result method of the globalSignificance class I calculate the projected significance. Since I'am using only the 2018 MC it's important to scale it properly to predict how great the significance would be if I was
  using all the run II MC 1.513274 * [sqrt(2018 total number of events) / sqrt(total number of events assessed up to the current batch)] * [self.sigPart / sqrt(self.bkgPart)]. The value 1.513274 is due to the term
  sqrt(137.4/60), which is nothing more than a scale factor to prospect how great the significancy would be if the model was applied to the entire run II.
  '''
  def result(self):
    return 1.513274*tf.math.divide(tf.math.multiply(tf.math.sqrt(self.sigSumWeight+self.bkgSumWeight),self.sigPart),\
                          tf.math.multiply(tf.sqrt(self.bkgPartTotal + self.sigPartTotal),tf.math.sqrt(self.bkgPart)))

  '''
  The next method resets only the relevant state variables—the ones that need to be cleared or nullified at the end of a training epoch.
  '''
  def reset_state(self):
    self.sigPart.assign(0)
    self.bkgPart.assign(0)
    self.sigPartTotal.assign(0)
    self.bkgPartTotal.assign(0)

'''---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
'''
The next class is responsible for selecting the best model throughout the entire training and validation process
'''
class CustomModelCheckpoint(Callback):
    def __init__(self,metric,filepath, monitor, mode='auto', save_best_only=True, trainValidation_percent_threshold=10,minimum_accepted_val_metric=-np.inf,maximum_acceptable_loss = np.inf,\
                 trainingEpochs = np.inf):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath # diretório onde é salvo o melhor modelo escolhido
        self.monitor = monitor # This line sets the class's monitor variable, which determines the metric the callback will use to choose the best training epoch
        self.mode = mode # Tells if it is the maximum or the minimum of the metric that must have the attention, in the proccess of choosing the best training epoch
	''' The line below sets a boolean variable which tells the CustomModelCheckPoint class if it's relevant to save the model every time it fulfills the conditions
            to be chosen as worth of being selected or if we should save only the best performant model.
	'''
        self.save_best_only = save_best_only # Tell if we want to save only the best model, in the registered model's filepath, overwriting the former best model, or if we want to save the model everytime it improves
        self.trainValidation_percent_threshold = trainValidation_percent_threshold # This variable defines the maximum acceptable percentage discrepancy between the training and validation metric values for a model to be selected.

        self.best_val_metric = minimum_accepted_val_metric # This variable is the one that sets the minimum value, for the validation metric, that a model must have in order to be chosen. 

        self.minimum_acceptable_val_metric = minimum_accepted_val_metric #This does the same as before. The former line is important for the loop that choses the maximum, but this line definitely sets the minimum threshold
        self.maximum_acceptable_loss = maximum_acceptable_loss # Here, I set an additional condition for the chosen model: a maximum acceptable loss. However, I have been assigning its value as np.inf.
        self.trainingEpochs = trainingEpochs # The number of training epochs that was set in the model.fit keras method. I set it here for specific purposes inside the CustomModelCheckpoint Callback class.

    # The following method, of the CustomModelCheckpoint Callback class, refers to what it does when we achieve a training epoch end.
    def on_epoch_end(self, epoch, logs=None):
        current_val_metric = logs.get(f'val_{self.monitor}',None) # get the validation set value for the globalSignificance custom metric class in the end of the training epoch
        train_metric = logs.get(f'{self.monitor}', None) # # get the training set value for the globalSignificance custom metric class in the end of the training epoch
        current_val_loss = logs.get('val_loss') # Does the same thing but here the purpose is to get the validation loss
        '''
        The next condition is designed to break the endless training loop in the main part of the code (which is defined outside this class, in the main code) whenever we reach the final training epoch and obtain a valid
	model. There may be times when no model is obtained due to overly demanding conditions.
        '''
        if ((epoch == (self.trainingEpochs-1)) & (self.best_val_metric > self.minimum_acceptable_val_metric)):
          self.model.stop_training = True # This is done to stop the "while not self.model.stop_training" loop, in the main code, whenever we find a model that fulfills our demands 

	''' 
        The next conditionals exists to promote the logic necessary to register a valid machine learning model, during the training process, rejecting any model without the preset required conditions.
        '''
        if (self.mode == 'min' and (current_val_metric < self.best_val_metric)) or (self.mode == 'max' and (current_val_metric > self.best_val_metric)): # True conditional whenever the model reaches another high in validation
            if train_metric is not None:  # If the training metric value is None there will be no computation!
                if train_metric != np.nan: # This conditional is important to avoid numerical calculation errors.
                  '''
                  The next line is important to compute the percentual deviation between the training and validation metric performances: "percent_diff" of the model must be lower than a predefined maximum for the model
		  to be chosen.
                  '''
                  percent_diff = ((current_val_metric - train_metric) / (train_metric)) * 100

                else:
                  percent_diff = np.inf # When the training metric is np.nan this line guarantees the model will not be chosen.

                '''
                The next conditional is in charge of registering a model that sets a new high in the validation "globalSig" metric. But will only be accepted if the "percent_diff is smaller than
		"self.trainValidation_percent_threshold. In other words: no matter if we get a new high for the validation dataset. If the train dataset deviates too much from its performance, percentually, then the model
                will not be chosen. If the two mentioned conditions occur simultaneously, then a new best-performing model will be chosen and stored. It's noteworthy the relevance of the "self.maximum_acceptable_loss"
		parameter, even though I've always been setting it as "np.inf" and, due to this, it has had no relevance or outcome in the model choosing process.
                '''
                if (percent_diff <= self.trainValidation_percent_threshold) & (current_val_loss < self.maximum_acceptable_loss):
                    self.best_val_metric = current_val_metric
                    if self.save_best_only: # If this boolean variable is True, then the last best choen model will be overwriten by a new best model.
                        self.model.save(self.filepath+'best2.keras' , overwrite=True) # The line that effectively saves the model.
                        print('saved the model: ',current_val_metric,' ',train_metric,' ',percent_diff)
                    else: # This make sure that the model will not overwrite the last best performing model.
                        ''' 
                        Here i could've put a code to implement this specific situation of saving the model not overwriting any of the last, but I've prefered not to do it
                        '''
            else: # Error message that points that a valid training metric name was not supplied to the "CustomModelCheckpoint" callback class.
                print(f"Training metric 'train_{self.monitor}' not found. Model not saved.")

'''-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''

'''The next code cell is the main code cell'''

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

valTestSplit = 0.25 # 25% of the shuffled 2-D numpy array that represents our dataset will be assigned to as the validation split
epochs=200 # Number of training epochs

stopTraining = False # This variable is set to True every time the CustomModelCheckpoint class finds a model that fulfills all the preset conditions that a chosen model must reach. 
while (not stopTraining):
    ''' 
    The next line randomly splits the dataset into training and validation sets.
    '''
    dataset_train, dataset_test, dataset_train_W, dataset_test_W, dataset_train_Y, dataset_test_Y\
                                            = train_test_split(dataset,dataset_W,dataset_Y\
                                            ,test_size=valTestSplit,shuffle=True)
    '''
    The next 4 lines define the used machine learning architecture: shallow MLP with 200 neurons in the hidden layer, one input layer and one output layer with a single neuron 
    '''
    model = models.Sequential()
    model.add(layers.Input(shape=(21,)))
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    optim = Adam(learning_rate=0.001) # The optimizer choice
    globalSignif = globalSignificance(threshold=0.94,sigSumWeight=sigSumWeight.astype('float32'),bkgSumWeight=bkgSumWeight.astype(dtype='float32')) # Instantiation of the "globalSignificance" custom metric object and the threshold

    '''
    The next line compiles the model, sets weighted metric and optimizer. "run_eagerly" was/is only used for debugging purposes.
    '''
    model.compile(optimizer=optim, loss='binary_crossentropy', weighted_metrics=[globalSig3Jets],run_eagerly=False)

    '''
    The next line instantiates a CustomModelCheckpoint object to be passed to the model.fit method. As one can see in the next line, we set the preset requirements such as "minimum_acceptable_val_metric", which is the minimum
    acceptable validation metric that a model must have to be chosen, the "trainValidation_percent_threshold" of 1%, which is the largest percentual deviation the validation metric value must have w.r.t the training metric
    value, for a model to be chosen, the monitor is the "globalSig" custom metric and the mode is max, pointing out that we are searchig for the highest possible metric values. The number of training epochs is also important
    internally for the class, the name of the custom metric class object and "filepath" are also important
    '''
    modelCheckPoint = CustomModelCheckpoint(globalSignif,filepath,monitor='globalSig',mode='max',save_best_only=True,trainValidation_percent_threshold = 1.0,trainingEpochs = epochs\
                                            ,minimum_accepted_val_metric = 1.9)

    '''
    This next line sets the batch size, which I've prefired to set as a function of the number of batches that a dataset provides us. I've always procceed to prime factorize the "len(dataset_train)" quantity and try to set
    the number of batches as the integer divisor of "len(dataset_train)" that lies as close as possible to 1000. This is due to the appearance of negative loss values, in some training epochs, when I use a very high number
    of batches. The other issue that prevents me from using a batch size much larger than 1000 is the possibility of having bins with a negative number of events in the final histogram of the scores: this effect is due to
    the presence of events with negative weights in the 2018 MC. At the same time if I use a number of batches much smaller than 1000 I suffer from the problem of impairing the generalization capabilities of the model as
    well as taking the risk of converging to a local minimum in the optimization landscape.
    '''
    miniBatch = int(round(len(dataset_train)/2000))  
    

    '''
    The next line begins effectively stars the training and validation process, calling the CustomModelCheckpoint callback function and globalSignificance custom metric
    '''
    history = model.fit( dataset_train , dataset_train_Y,
                    epochs=epochs,
                    batch_size=miniBatch,
                    verbose=1,
                    validation_data=(dataset_test,dataset_test_Y,dataset_test_W),sample_weight = dataset_train_W,callbacks= [lr_scheduler,modelCheckPoint])
    
    stopTraining = model.stop_training # This line sets the stopTraining boolean variable to the model.stop_training that was set by the CustomModelCheckpoint callback class

'''
In the next two lines, I load the best model saved by the CustomModelCheckpoint callback class, allowing me to experiment with it and plot the output scores, ROC-AUC curve, or any other metrics I wish to explore.
'''
<chosenFileName = filepath+'best2.keras'
model.load_weights(chosenFileName)
