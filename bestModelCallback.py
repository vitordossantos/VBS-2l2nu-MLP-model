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
